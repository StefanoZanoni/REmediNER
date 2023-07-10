import os
import time
import warnings

import numpy
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import KFold

from src.NER.model_ner import NerModel
from src.early_stopping import EarlyStopper
from src.plot import plot_loss, plot_metrics, plot_heat_map

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

id_label = {}
label_id = {}


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


def compute_metrics_mean(mean_dict, current_metrics_dict, dim=None):
    if dim is not None:
        for key in mean_dict:
            mean_dict[key]['f1-score'] /= dim
            mean_dict[key]['precision'] /= dim
            mean_dict[key]['recall'] /= dim
            mean_dict[key]['support'] = int(np.floor(mean_dict[key]['support'] / dim))
    else:
        for key in mean_dict:
            if key == 'micro avg' and key not in current_metrics_dict:
                mean_dict[key]['f1-score'] += current_metrics_dict['accuracy']
                mean_dict[key]['precision'] += current_metrics_dict['accuracy']
                mean_dict[key]['recall'] += current_metrics_dict['accuracy']
                mean_dict[key]['support'] += current_metrics_dict['macro avg']['support']
            else:
                mean_dict[key]['f1-score'] += current_metrics_dict[key]['f1-score']
                mean_dict[key]['precision'] += current_metrics_dict[key]['precision']
                mean_dict[key]['recall'] += current_metrics_dict[key]['recall']
                mean_dict[key]['support'] += current_metrics_dict[key]['support']


def create_mean_dict():
    mean_dict = {}
    for i in range(len(id_label)):
        mean_dict[id_label[i]] = {}
        mean_dict['micro avg'] = {}
        mean_dict['macro avg'] = {}
        mean_dict['weighted avg'] = {}

    for key in mean_dict:
        mean_dict[key]['f1-score'] = 0
        mean_dict[key]['precision'] = 0
        mean_dict[key]['recall'] = 0
        mean_dict[key]['support'] = 0

    return mean_dict


def get_classes_weight(true):
    unique = np.unique(true)
    classes = [id_label[unique[i]] for i in range(len(unique))]
    classes = [label_id[el] for el in classes]
    classes = np.array(classes)
    print(id_label, label_id)
    classes_weights = compute_class_weight(class_weight='balanced', classes=classes, y=true)
    label_weight = {el: classes_weights[i] for i, el in enumerate(classes)}
    for key in label_id:
        if label_id[key] not in label_weight:
            label_weight[label_id[key]] = 0
            classes_weights = numpy.insert(classes_weights, label_id[key], 0)

    return classes_weights


def scoring(true_values, predicted_values):
    true_values, predicted_values = clean_data(true_values, predicted_values)
    batch_dim = len(true_values)
    mean_dict = create_mean_dict()
    mean_cm = np.zeros((len(id_label), len(id_label)))
    for true, predicted in zip(true_values, predicted_values):
        metrics_dict = classification_report(true, predicted,
                                             target_names=[id_label[0], id_label[1], id_label[2],
                                                           id_label[3], id_label[4]],
                                             labels=[0, 1, 2, 3, 4],
                                             output_dict=True)
        compute_metrics_mean(mean_dict, metrics_dict)
        cm = confusion_matrix(true, predicted, labels=[0, 1, 2, 3, 4], normalize='all')
        mean_cm += cm

    compute_metrics_mean(mean_dict, metrics_dict, dim=batch_dim)
    mean_cm /= batch_dim

    return mean_dict, mean_cm


def save_checkpoint(epoch, model):
    if not os.path.exists('./NER/saves'):
        os.makedirs('./NER/saves')
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
                 world_size: int,
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
        metrics_dict, cm = scoring(true_labels, predicted_labels)

        return loss.item(), metrics_dict, cm

    def __run_epoch_ner(self, train_in, train_out, epoch, model, optimizer, scheduler):

        b_sz = len(next(iter(train_in))[0])
        train_in.sampler.set_epoch(epoch)
        train_out.sampler.set_epoch(epoch)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch-size: {b_sz} | Steps {len(train_in)}")

        epoch_loss = 0
        mean_dict = create_mean_dict()
        mean_cm = np.zeros((len(id_label), len(id_label)))
        train_dim = len(train_in)

        for (ids, masks), labels in zip(train_in, train_out):
            labels = labels[0]
            ids = ids.to(self.gpu_id)
            masks = masks.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            batch_loss, metrics_dict, cm = self.__run_batch_ner(ids, masks, labels, model, optimizer, scheduler)
            epoch_loss += batch_loss / b_sz
            compute_metrics_mean(mean_dict, metrics_dict)
            mean_cm += cm

        compute_metrics_mean(mean_dict, metrics_dict, dim=train_dim)

        return epoch_loss / train_dim, mean_dict, cm / train_dim

    def __train_ner(self, train_in, train_out, model, optimizer, scheduler, val_in=None, val_out=None, ):

        train_loss_mean = []
        train_metrics_mean = []
        validation_loss_mean = []
        validation_metrics_mean = []
        train_cm_mean = np.zeros((len(id_label), len(id_label)))
        val_cm_mean = np.zeros((len(id_label), len(id_label)))

        stopper = EarlyStopper(patience=3, min_delta=0.005)

        start_time = time.time()

        for epoch in range(self.epochs):

            parameters_dict = model.state_dict()

            # training step
            train_loss, train_metrics_dict, train_cm = self.__run_epoch_ner(train_in, train_out, epoch, model,
                                                                            optimizer,
                                                                            scheduler)
            if val_in is not None or val_out is not None:
                # validation step
                with torch.no_grad():
                    validation_loss, validation_metrics_dict, val_cm = self.__validation_ner(val_in, val_out, model,
                                                                                             epoch)

                validation_loss_mean.append(validation_loss)
                validation_metrics_mean.append(validation_metrics_dict)
                val_cm_mean += val_cm

                train_loss_mean.append(train_loss)
                train_metrics_mean.append(train_metrics_dict)
                train_cm_mean += train_cm

                # check for early stopping condition
                end, best_parameters_dict = stopper.early_stop(validation_loss, parameters_dict)
                if end:
                    # restore the best parameters found
                    model.load_state_dict(best_parameters_dict)
                    break
            else:
                train_loss_mean.append(train_loss)
                train_metrics_mean.append(train_metrics_dict)
                train_cm_mean += train_cm

            if self.gpu_id == 0 and epoch % self.save_evey == 0:
                save_checkpoint(epoch, model)

        print(f'---[GPU{self.gpu_id}] TRAINING TIME IN SECONDS: %s ---\n\n' % (time.time() - start_time))

        if val_in is not None or val_out is not None:
            return train_loss_mean, train_metrics_mean, validation_loss_mean, validation_metrics_mean, \
                train_cm_mean / self.epochs, val_cm_mean / self.epochs, epoch
        else:
            return train_loss_mean, train_metrics_mean, train_cm_mean / self.epochs

    def __validation_ner(self, val_in, val_out, model, epoch):

        b_sz = len(next(iter(val_in))[0])
        val_in.sampler.set_epoch(epoch)
        val_out.sampler.set_epoch(epoch)
        model.eval()
        loss_sum = 0
        mean_dict = create_mean_dict()
        mean_cm = np.zeros((len(id_label), len(id_label)))
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
            metrics_dict, cm = scoring(true_labels, predicted_labels)
            compute_metrics_mean(mean_dict, metrics_dict)
            mean_cm += cm

        compute_metrics_mean(mean_dict, metrics_dict, dim=val_dim)

        return loss_sum / val_dim, mean_dict, mean_cm / val_dim

    def kfold_cross_validation(self, k):

        kfold = KFold(n_splits=k, shuffle=True, random_state=0)
        train_means = []
        validation_means = []
        epochs = []

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
            global id_label, label_id
            id_label = self.bert_model['id_label']
            label_id = self.bert_model['label_id']

            model = NerModel(bert_model, len_labels, id_label, label_id)
            model.apply(reset_parameters)
            optimizer = model.get_optimizer()
            scheduler = model.get_scheduler(self.epochs * len(train_subsampler) / self.batch_size)
            model.to(self.gpu_id)
            model = DDP(model, device_ids=[self.gpu_id])

            train_losses, train_metrics, validation_losses, validation_metrics, train_cm, val_cm, max_epoch = \
                self.__train_ner(train_in_loader, train_out_loader, model, optimizer, scheduler,
                                 val_in_loader, val_out_loader)

            epochs.append(max_epoch)
            train_mean = sum(train_losses) / len(train_losses)
            validation_mean = sum(validation_losses) / len(validation_losses)
            train_means.append(train_mean)
            validation_means.append(validation_mean)

            # Saving the model
            save_path = f'./NER/saves/model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            if self.gpu_id == 0:
                plot_loss(train_losses, validation_losses, fold, 'NER')
                plot_metrics(train_metrics, validation_metrics, fold)
                plot_heat_map(val_cm, fold, train_cm)

        print(f'K-FOLD TRAIN RESULTS MEAN FOR {k} FOLDS:'f' {sum(train_means) / len(train_means)}\n\n')

        print(f'K-FOLD VALIDATION RESULTS MEAN FOR {k} FOLDS:'f' {sum(validation_means) / len(validation_means)}\n\n')

        return int(np.floor(sum(epochs) / k))

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

        bert_model = self.bert_model['bert_model']
        len_labels = self.bert_model['len_labels']
        global id_label
        id_label = self.bert_model['id_label']
        label_id = self.bert_model['label_id']

        model = NerModel(bert_model, len_labels, id_label, label_id)
        model.apply(reset_parameters)
        optimizer = model.get_optimizer()
        scheduler = model.get_scheduler(self.epochs * len(self.train_in) / self.batch_size)
        model.to(self.gpu_id)
        model = DDP(model, device_ids=[self.gpu_id])

        self.epochs = max_epoch
        _, _, _ = self.__train_ner(train_in_loader, train_out_loader, model, optimizer, scheduler)

        return model