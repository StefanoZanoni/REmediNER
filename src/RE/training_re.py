import os
import time

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import KFold

from src.RE.model_re import ReModel
from src.early_stopping import EarlyStopper


def save_checkpoint(epoch, model):
    if not os.path.exists('./RE/saves'):
        os.makedirs('./RE/saves')
    ckp = model.module.state_dict()
    torch.save(ckp, "./RE/saves/checkpoint.pt")
    print(f"Epoch {epoch} | Training checkpoint saved at RE/saves/checkpoint.pt")


def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


class TrainerRe:
    def __init__(self,
                 bert_name: str,
                 train_in: TensorDataset,
                 train_out: TensorDataset,
                 epochs: int,
                 batch_size: int,
                 gpu_id: int,
                 save_every: int,
                 world_size: int,
                 max_number_pos: int,
                 input_length: int
                 ) -> None:
        self.bert_name = bert_name
        self.gpu_id = gpu_id
        self.train_in = train_in
        self.train_out = train_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_evey = save_every
        self.world_size = world_size
        self.max_number_pos = max_number_pos
        self.input_length = input_length
        self.embedding = torch.nn.Embedding(self.max_number_pos, 768, padding_idx=0).to(self.gpu_id)

    def __run_batch_re(self, ids, masks, pos, train_output, model_re, optimizer):

        model_re.train()
        optimizer.zero_grad()
        effective_batch_size = list(ids.size())[0]
        predicted_output = model_re(ids, masks, pos, self.embedding, effective_batch_size)
        predicted_output = torch.transpose(predicted_output, dim0=1, dim1=2)
        loss_fun = torch.nn.CrossEntropyLoss(reduction='none').to(self.gpu_id)
        loss_masked = loss_fun(predicted_output, train_output)
        pad = -100
        loss_mask = train_output != pad
        loss = loss_masked.sum() / loss_mask.sum()
        loss.backward()
        optimizer.step()

        return predicted_output, loss.item()

    def __run_epoch_re(self, train_in, train_output, model_re, optimizer, epoch):

        b_sz = len(next(iter(train_in))[0])
        train_in.sampler.set_epoch(epoch)
        train_output.sampler.set_epoch(epoch)
        epoch_loss = 0
        epoch_re_output = []

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch-size: {b_sz} | Steps {len(train_in)}")

        for (ids, masks, pos), out in zip(train_in, train_output):
            out = out[0]
            ids = ids.to(self.gpu_id)
            masks = masks.to(self.gpu_id)
            pos = pos.to(self.gpu_id)
            out = out.to(self.gpu_id)
            batch_re_output, batch_loss = self.__run_batch_re(ids, masks, pos, out, model_re, optimizer)
            epoch_loss += batch_loss
            epoch_re_output.append(batch_re_output)

        return batch_re_output, epoch_loss / len(train_in)

    def __train_re(self, train_in, train_out, model_re, optimizer, val_in=None, val_out=None):

        train_loss_mean = []
        validation_loss_mean = []
        re_output = []
        stopper = EarlyStopper(patience=3, min_delta=0.005)

        start_time = time.time()

        for epoch in range(self.epochs):

            parameters_dict = model_re.state_dict()

            # training step
            batch_re_output, epoch_loss = \
                self.__run_epoch_re(train_in, train_out, model_re, optimizer, epoch)

            if val_in is not None or val_out is not None:
                # validation step
                with torch.no_grad():
                    validation_loss = self.__validation_re(val_in, val_out, model_re, epoch)

                validation_loss_mean.append(validation_loss)

                train_loss_mean.append(epoch_loss)
                re_output.extend(batch_re_output)

                # check for early stopping condition
                end, best_parameters_dict = stopper.early_stop(validation_loss, parameters_dict)
                if end:
                    # restore the best parameters found
                    model_re.load_state_dict(best_parameters_dict)
                    break
            else:
                train_loss_mean.append(epoch_loss)
                re_output.extend(batch_re_output)

            # save the checkpoint
            if self.gpu_id == 0 and epoch % self.save_evey == 0:
                save_checkpoint(epoch, model_re)

        print(f'---[GPU{self.gpu_id}] TRAINING TIME IN SECONDS: %s ---\n\n' % (time.time() - start_time))

        if val_in is not None or val_out is not None:
            return re_output, train_loss_mean, validation_loss_mean, epoch
        else:
            return re_output, train_loss_mean

    def __validation_re(self, val_in, val_out, model_re, epoch):

        val_in.sampler.set_epoch(epoch)
        val_out.sampler.set_epoch(epoch)
        model_re.eval()
        loss_sum = 0

        for (ids, masks, pos), out in zip(val_in, val_out):
            out = out[0]
            ids.to(self.gpu_id)
            masks.to(self.gpu_id)
            pos = pos.to(self.gpu_id)
            out = out.to(self.gpu_id)
            effective_batch_size = list(ids.size())[0]
            predicted_output = model_re(ids, masks, pos, self.embedding, effective_batch_size)
            predicted_output = torch.transpose(predicted_output, dim0=1, dim1=2)
            loss_fun = torch.nn.CrossEntropyLoss(reduction='none').to(self.gpu_id)
            loss_masked = loss_fun(predicted_output, out)
            pad = -100
            loss_mask = out != pad
            loss = loss_masked.sum() / loss_mask.sum()
            loss_sum += loss.item()

        return loss_sum / len(val_in)

    def kfold_cross_validation(self, k):

        kfold = KFold(n_splits=k, shuffle=True, random_state=0)
        train_means = []
        validation_means = []
        epochs = []

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

            model = ReModel(self.bert_name, self.batch_size, self.input_length)
            model.apply(reset_parameters)
            optimizer = model.get_optimizer()
            model.to(self.gpu_id)
            model = DDP(model, device_ids=[self.gpu_id])

            re_output, train_losses, validation_losses, max_epoch = \
                self.__train_re(train_in_loader, train_out_loader, model, optimizer, val_in_loader, val_out_loader)

            epochs.append(max_epoch)
            train_mean = sum(train_losses) / len(train_losses)
            validation_mean = sum(validation_losses) / len(validation_losses)
            train_means.append(train_mean)
            validation_means.append(validation_mean)

            # Saving the model
            save_path = f'./RE/saves/model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            if self.gpu_id == 0:
                plot_loss(train_losses, validation_losses, fold, 'RE')

        print(f'K-FOLD TRAIN RESULTS MEAN FOR {k} FOLDS:'f' {sum(train_means) / len(train_means)}\n\n')

        print(f'K-FOLD VALIDATION RESULTS MEAN FOR {k} FOLDS:'f' {sum(validation_means) / len(validation_means)}\n\n')

        return re_output, int(np.floor(sum(epochs) / k))

    def re_train(self, max_epoch):

        train_in_loader = DataLoader(self.train_in,
                                     batch_size=self.batch_size,
                                     pin_memory=True,
                                     shuffle=False,
                                     sampler=DistributedSampler(self.train_in,
                                                                num_replicas=self.world_size,
                                                                rank=self.gpu_id))
        train_out_loader = DataLoader(self.train_out,
                                      batch_size=self.batch_size,
                                      pin_memory=True,
                                      shuffle=False,
                                      sampler=DistributedSampler(self.train_out,
                                                                 num_replicas=self.world_size,
                                                                 rank=self.gpu_id))

        model = ReModel(self.bert_name, self.batch_size, self.input_length)
        model.apply(reset_parameters)
        optimizer = model.get_optimizer()
        model.to(self.gpu_id)
        model = DDP(model, device_ids=[self.gpu_id])

        self.epochs = max_epoch
        _, _ = self.__train_re(train_in_loader, train_out_loader, model, optimizer)

        return model
