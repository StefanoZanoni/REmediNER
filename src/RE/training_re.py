import time

import torch

from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import KFold

from src.RE.model_re import ReModel


def save_checkpoint(epoch, model):
    ckp = model.module.state_dict()
    torch.save(ckp, "checkpoint.pt")
    print(f"Epoch {epoch} | Training checkpoint saved at checkpoint.pt")


def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


class TrainerRe:
    def __init__(self,
                 model_ner,
                 context_mean_length: int,
                 entity_embeddings_length: int,
                 label_id: dict,
                 train_in: TensorDataset,
                 train_out: TensorDataset,
                 epochs: int,
                 batch_size: int,
                 gpu_id: int,
                 save_every: int,
                 world_size: int
                 ) -> None:
        self.gpu_id = gpu_id
        self.model_ner = model_ner
        self.context_mean_length = context_mean_length
        self.entity_embeddings_length = entity_embeddings_length
        self.label_id = label_id
        self.train_in = train_in
        self.train_out = train_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_evey = save_every
        self.world_size = world_size

    def _run_batch_re(self, ids, masks, label_id, train_output,
                      model_re, model_ner, optimizer):

        model_re.train()
        optimizer.zero_grad()
        with torch.no_grad():
            model_ner.eval()
            _, entities_vector, entities_context = model_ner(ids, masks)
        predicted_output = model_re(entities_vector, entities_context, label_id)
        loss = torch.nn.BCELoss(predicted_output[0][2], train_output).to(self.gpu_id)
        loss.backward()
        optimizer.step()

        return predicted_output, loss

    def _run_epoch_re(self, train_in, train_output, label_id,
                      model_re, model_ner, optimizer, epoch):

        b_sz = len(next(iter(train_in))[0])
        train_in.sampler.set_epoch(epoch)
        loss = torch.zeros(1, dtype=torch.float32, device=self.gpu_id)
        batch_re_output = []

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch-size: {b_sz} | Steps {len(train_in)}")

        for (ids, masks, _), out in zip(train_in, train_output):
            out = out[0]
            ids = ids.to(self.gpu_id)
            masks = masks.to(self.gpu_id)
            out = out.to(self.gpu_id)
            re_output, loss_batch = self._run_batch_re(ids, masks, label_id, out,
                                                       model_re, model_ner, optimizer)
            loss += loss_batch / b_sz
            batch_re_output.append(re_output)

        return batch_re_output, loss / len(train_in)

    def train_re(self, train_in, train_out, label_id, model_re, model_ner, optimizer):

        epoch_loss_means = torch.empty(0, dtype=torch.float32, device=self.gpu_id)
        re_output = []
        start_time = time.time()

        for epoch in range(self.epochs):
            batch_re_output, batch_loss = \
                self._run_epoch_re(train_in, train_out, label_id, model_re, model_ner, optimizer, epoch)
            epoch_loss_means = torch.cat([epoch_loss_means, batch_loss], dim=0)
            re_output.extend(batch_re_output)
            if self.gpu_id == 0 and epoch % self.save_evey == 0:
                save_checkpoint(epoch, model_re)

        print("--- training time in seconds: %s ---" % (time.time() - start_time))

        return re_output, epoch_loss_means

    def _validation_re(self, val_in, val_out, label_id, model_ner, model_re):

        b_sz = len(next(iter(val_in))[0])
        model_re.eval()
        loss_sum = 0

        for (ids, masks, labels), out in zip(val_in, val_out):
            out = out[0]
            ids.to(self.gpu_id)
            masks.to(self.gpu_id)
            labels.to(self.gpu_id)
            out = out.to(self.gpu_id)
            _, entities_vector, entities_context = model_ner(ids, masks, labels)
            predicted_output = model_re(entities_vector, entities_context, label_id)
            loss = torch.nn.BCELoss(predicted_output[2], out)
            loss_sum += loss / b_sz

        return loss_sum / len(val_in)

    def kfold_cross_validation(self, k):

        kfold = KFold(n_splits=k, shuffle=True, random_state=0)

        results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.train_in)):
            print(f'--- FOLD {fold} ---\n')

            train_in_subsampler = SubsetRandomSampler(train_idx)
            val_in_subsampler = SubsetRandomSampler(val_idx)
            train_in_loader = DataLoader(self.train_in,
                                         batch_size=self.batch_size,
                                         pin_memory=True,
                                         shuffle=False,
                                         sampler=DistributedSampler(train_in_subsampler,
                                                                    num_replicas=self.world_size,
                                                                    rank=self.gpu_id))
            val_in_loader = DataLoader(self.train_in,
                                       batch_size=self.batch_size,
                                       pin_memory=True,
                                       shuffle=False,
                                       sampler=DistributedSampler(val_in_subsampler,
                                                                  num_replicas=self.world_size,
                                                                  rank=self.gpu_id))
            train_out_loader = DataLoader(self.train_out,
                                          batch_size=self.batch_size,
                                          pin_memory=True,
                                          shuffle=False,
                                          sampler=DistributedSampler(train_in_subsampler,
                                                                     num_replicas=self.world_size,
                                                                     rank=self.gpu_id))
            val_out_loader = DataLoader(self.train_out,
                                        batch_size=self.batch_size,
                                        pin_memory=True,
                                        shuffle=False,
                                        sampler=DistributedSampler(val_in_subsampler,
                                                                   num_replicas=self.world_size,
                                                                   rank=self.gpu_id))

            model = ReModel(self.context_mean_length, self.entity_embeddings_length)
            model.apply(reset_parameters)
            optimizer = model.get_optimizer()
            model.to(self.gpu_id)
            model = DDP(model, device_ids=[self.gpu_id])

            re_output, loss = \
                self.train_re(train_in_loader, train_out_loader, self.label_id, model, self.model_ner, optimizer)
            print(loss)

            # Saving the model
            save_path = f'./model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            # Evaluation for this fold
            with torch.no_grad():
                results.append(self._validation_re(val_in_loader, val_out_loader, self.label_id, self.model_ner, model))

        print(f'K-FOLD CROSS VALIDATION RESULTS MEAN FOR {k} FOLDS: {sum(results) / len(results)}\n')

        return re_output
