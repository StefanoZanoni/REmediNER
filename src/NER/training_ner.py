import time

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import KFold

from src.NER.model_ner import NerModel
from src.early_stopping import EarlyStopper
from src.plot import plot_loss, plot_metrics

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


def clean_data(true_values, predicted_values):
    new_true = []
    new_predicted = []

    for i in range(true_values.shape[0]):
        true = true_values[i]
        predicted = predicted_values[i]
        predicted = np.delete(predicted, np.where(true == -100))
        true = np.delete(true, np.where(true == -100))
        new_true.append(true)
        new_predicted.append(predicted)

    return new_true, new_predicted


def scoring(true_values, predicted_values):
    true_values, predicted_values = clean_data(true_values, predicted_values)
    precisions = []
    recalls = []
    f1s = []
    batch_dim = len(true_values)
    for true, predicted in zip(true_values, predicted_values):
        precision = precision_score(true, predicted, average='micro')
        precisions.append(precision)
        recall = recall_score(true, predicted, average='micro')
        recalls.append(recall)
        f1 = f1_score(true, predicted, average='micro')
        f1s.append(f1)

    return sum(precisions) / batch_dim, sum(recalls) / batch_dim, sum(f1s) / batch_dim


def confusion_matrix(true_values, predicted_values):
    # Calculate confusion matrix
    cm = confusion_matrix(true_values, predicted_values)

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def save_checkpoint(epoch, model):
    ckp = model.module.state_dict()
    torch.save(ckp, "./NER/saves/checkpoint.pt")
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

    def __run_batch_ner(self, ids, masks, labels, model, optimizer, scheduler):

        model.train()
        optimizer.zero_grad()
        logits, entities_vector = model(ids, masks)
        loss_fun = torch.nn.CrossEntropyLoss().to(self.gpu_id)
        logits = torch.transpose(logits, dim0=1, dim1=2)
        loss = loss_fun(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        logits = torch.transpose(logits, dim0=1, dim1=2)
        predicted_output = torch.argmax(logits, dim=-1)
        predicted_labels = predicted_output.numpy(force=True)
        true_labels = labels.numpy(force=True)
        precision, recall, f1 = scoring(true_labels, predicted_labels)

        return loss.item(), precision, recall, f1

    def __run_epoch_ner(self, train_in, train_out, epoch, model, optimizer, scheduler):

        b_sz = len(next(iter(train_in))[0])
        train_in.sampler.set_epoch(epoch)
        train_out.sampler.set_epoch(epoch)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch-size: {b_sz} | Steps {len(train_in)}")

        epoch_loss = 0
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0
        train_dim = len(train_in)

        for (ids, masks), labels in zip(train_in, train_out):
            labels = labels[0]
            ids = ids.to(self.gpu_id)
            masks = masks.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            batch_loss, precision, recall, f1 = self.__run_batch_ner(ids, masks, labels, model, optimizer, scheduler)
            epoch_loss += batch_loss / b_sz
            epoch_precision += precision
            epoch_recall += recall
            epoch_f1 += f1

        return epoch_loss / train_dim, epoch_precision / train_dim, epoch_recall / train_dim, epoch_f1 / train_dim

    def __train_ner(self, train_in, train_out, val_in, val_out, model, optimizer, scheduler):

        train_loss_mean = []
        train_precision_mean = []
        train_recall_mean = []
        train_f1_mean = []
        validation_loss_mean = []
        validation_precision_mean = []
        validation_recall_mean = []
        validation_f1_mean = []
        stopper = EarlyStopper(patience=3, min_delta=0.005)

        start_time = time.time()

        for epoch in range(self.epochs):

            parameters_dict = model.state_dict()

            # training step
            train_loss, train_precision, train_recall, train_f1 = \
                self.__run_epoch_ner(train_in, train_out, epoch, model, optimizer, scheduler)
            # validation step
            with torch.no_grad():
                validation_loss, validation_precision, validation_recall, validation_f1 = \
                    self.__validation_ner(val_in, val_out, model, epoch)

            train_loss_mean.append(train_loss)
            train_precision_mean.append(train_precision)
            train_recall_mean.append(train_recall)
            train_f1_mean.append(train_f1)
            validation_loss_mean.append(validation_loss)
            validation_precision_mean.append(validation_precision)
            validation_recall_mean.append(validation_recall)
            validation_f1_mean.append(validation_f1)

            # check for early stopping condition
            end, best_parameters_dict = stopper.early_stop(validation_loss, parameters_dict)
            if end:
                # restore the best parameters found
                model.load_state_dict(best_parameters_dict)
                break

            if self.gpu_id == 0 and epoch % self.save_evey == 0:
                save_checkpoint(epoch, model)

        print("--- TRAINING TIME IN SECONDS: %s ---\n" % (time.time() - start_time))

        return train_loss_mean, train_precision_mean, train_recall_mean, train_f1_mean, \
            validation_loss_mean, validation_precision_mean, validation_recall_mean, validation_f1_mean

    def __validation_ner(self, val_in, val_out, model, epoch):

        b_sz = len(next(iter(val_in))[0])
        val_in.sampler.set_epoch(epoch)
        val_out.sampler.set_epoch(epoch)
        model.eval()
        loss_sum = 0
        precision = 0
        recall = 0
        f1 = 0
        val_dim = len(val_in)

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
            logits = torch.transpose(logits, dim0=1, dim1=2)
            predicted_output = torch.argmax(logits, dim=-1)
            predicted_labels = predicted_output.numpy(force=True)
            true_labels = labels.numpy(force=True)
            batch_precision, batch_recall, batch_f1 = scoring(true_labels, predicted_labels)
            precision += batch_precision
            recall += batch_recall
            f1 += batch_f1

        return loss_sum / val_dim, precision / val_dim, recall / val_dim, f1 / val_dim

    def kfold_cross_validation(self, k):

        kfold = KFold(n_splits=k, shuffle=True, random_state=0)
        train_means = []
        validation_means = []

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

            train_losses, train_precisions, train_recalls, train_f1s, \
                validation_losses, validation_precisions, validation_recalls, validation_f1s = \
                self.__train_ner(train_in_loader, train_out_loader,
                                 val_in_loader, val_out_loader,
                                 model, optimizer, scheduler)

            train_mean = sum(train_losses) / len(train_losses)
            validation_mean = sum(validation_losses) / len(validation_losses)
            train_means.append(train_mean)
            validation_means.append(validation_mean)

            # Saving the model
            save_path = f'./NER/saves/model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            plot_loss(train_losses, validation_losses, fold)
            plot_metrics(train_precisions, train_recalls, train_f1s,
                         validation_precisions, validation_recalls, validation_f1s, fold)

        print(f'K-FOLD TRAIN RESULTS MEAN FOR {k} FOLDS:'f' {sum(train_means) / len(train_means)}\n\n')

        print(f'K-FOLD VALIDATION RESULTS MEAN FOR {k} FOLDS:'f' {sum(validation_means) / len(validation_means)}\n\n')

        return model
