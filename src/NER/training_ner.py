import time

import torch

from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import KFold

from src.NER.model_ner import NerModel


def save_checkpoint(epoch, model):
    ckp = model.module.state_dict()
    torch.save(ckp, "checkpoint.pt")
    print(f"Epoch {epoch} | Training checkpoint saved at NER/saves/checkpoint.pt")


def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


class TrainerNer:
    def __init__(self,
                 bert_model: dict,
                 train_in: TensorDataset,
                 train_out: TensorDataset,
                 epochs: int,
                 batch_size: int,
                 gpu_id: int,
                 save_every: int,
                 world_size: int
                 ) -> None:
        self.gpu_id = gpu_id
        self.bert_model = bert_model
        self.train_in = train_in
        self.train_out = train_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_evey = save_every
        self.world_size = world_size

    def _run_batch_ner(self, ids, masks, labels, model, optimizer, scheduler):

        model.train()
        optimizer.zero_grad()
        logits, entities_vector = model(ids, masks)
        loss_fun = torch.nn.CrossEntropyLoss().to(self.gpu_id)
        logits = torch.transpose(logits, dim0=1, dim1=2)
        loss = loss_fun(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        return torch.tensor(loss.item(), dtype=torch.float32, device=self.gpu_id)

    def _run_epoch_ner(self, train_in, train_out, epoch, model, optimizer, scheduler):

        b_sz = len(next(iter(train_in))[0])
        train_in.sampler.set_epoch(epoch)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch-size: {b_sz} | Steps {len(train_in)}")

        loss = torch.zeros(1, dtype=torch.float32, device=self.gpu_id)

        for (ids, masks), labels in zip(train_in, train_out):
            labels = labels[0]
            ids = ids.to(self.gpu_id)
            masks = masks.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            loss += (self._run_batch_ner(ids, masks, labels, model, optimizer, scheduler) / b_sz)

        return loss / len(train_in)

    def train_ner(self, train_in, train_out, model, optimizer, scheduler):

        epochs_loss_means = torch.empty(0, dtype=torch.float32, device=self.gpu_id)
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_loss = self._run_epoch_ner(train_in, train_out, epoch, model, optimizer, scheduler)
            epochs_loss_means = torch.cat([epochs_loss_means, epoch_loss], dim=0)
            if self.gpu_id == 0 and epoch % self.save_evey == 0:
                save_checkpoint(epoch, model)

        print("--- TRAINING TIME IN SECONDS: %s ---\n" % (time.time() - start_time))
        return epochs_loss_means

    def _validation_ner(self, val_in, val_out, model):

        b_sz = len(next(iter(val_in))[0])
        model.eval()
        loss_sum = 0

        for (ids, masks), labels in zip(val_in, val_out):
            labels = labels[0]
            ids = ids.to(self.gpu_id)
            masks = masks.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            logits, entities_vector = model(ids, masks)
            loss_fun = torch.nn.CrossEntropyLoss().to(self.gpu_id)
            logits = torch.transpose(logits, dim0=1, dim1=2)
            loss = loss_fun(logits, labels)
            loss_sum += loss.item() / b_sz

        return loss_sum / len(val_in)

    def kfold_cross_validation(self, k):

        kfold = KFold(n_splits=k, shuffle=True, random_state=0)

        results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.train_in)):
            print(f'--- FOLD {fold} ---\n')

            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            train_in_loader = DataLoader(self.train_in,
                                         batch_size=self.batch_size,
                                         pin_memory=True,
                                         shuffle=False,
                                         sampler=DistributedSampler(train_subsampler,
                                                                    num_replicas=self.world_size,
                                                                    rank=self.gpu_id))
            val_in_loader = DataLoader(self.train_in,
                                       batch_size=self.batch_size,
                                       pin_memory=True,
                                       shuffle=False,
                                       sampler=DistributedSampler(val_subsampler,
                                                                  num_replicas=self.world_size,
                                                                  rank=self.gpu_id))
            train_out_loader = DataLoader(self.train_out,
                                          batch_size=self.batch_size,
                                          pin_memory=True,
                                          shuffle=False,
                                          sampler=DistributedSampler(train_subsampler,
                                                                     num_replicas=self.world_size,
                                                                     rank=self.gpu_id))
            val_out_loader = DataLoader(self.train_out,
                                        batch_size=self.batch_size,
                                        pin_memory=True,
                                        shuffle=False,
                                        sampler=DistributedSampler(val_subsampler,
                                                                   num_replicas=self.world_size,
                                                                   rank=self.gpu_id))

            bert_model = self.bert_model['bert_model']
            len_labels = self.bert_model['len_labels']
            id_label = self.bert_model['id_label']
            label_id = self.bert_model['label_id']

            model = NerModel(bert_model, len_labels, id_label, label_id)
            model.apply(reset_parameters)
            optimizer = model.get_optimizer()
            scheduler = model.get_scheduler(self.epochs * len(train_subsampler) / self.batch_size)
            model.to(self.gpu_id)
            model = DDP(model, device_ids=[self.gpu_id])

            result = self.train_ner(train_in_loader, train_out_loader, model, optimizer, scheduler)
            print(result)

            # Saving the model
            save_path = f'model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            # Evaluation for this fold
            with torch.no_grad():
                results.append(self._validation_ner(val_in_loader, val_out_loader, model))

        print(f'K-FOLD CROSS VALIDATION RESULTS MEAN FOR {k} FOLDS: {sum(results) / len(results)}\n')
