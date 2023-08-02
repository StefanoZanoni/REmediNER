import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.NER.training_ner import create_mean_dict, scoring, compute_metrics_mean
from src.plot import plot_heat_map


def test_ner(test_input, test_output, model, batch_size, world_size, rank, id_label, loss_weights, input_length):
    test_in_loader = DataLoader(test_input,
                                batch_size=batch_size,
                                pin_memory=True,
                                shuffle=False,
                                sampler=DistributedSampler(test_input,
                                                           num_replicas=world_size,
                                                           rank=rank))
    test_out_loader = DataLoader(test_output,
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 shuffle=False,
                                 sampler=DistributedSampler(test_output,
                                                            num_replicas=world_size,
                                                            rank=rank))

    loss_weights = torch.tensor(loss_weights, dtype=torch.float)
    test_loss, test_metrics, test_cm = test(test_in_loader, test_out_loader, model, id_label, rank, loss_weights, input_length)
    print(f'------------------Test NER loss: {test_loss}------------------\n\n')
    print(f'------------------Test NER metrics: {test_metrics}------------------\n\n')
    plot_heat_map(test_cm)


def test(test_in, test_out, model, id_label, gpu_id, loss_weights, input_length):

    b_sz = len(next(iter(test_in))[0])

    model.eval()
    loss_sum = 0
    predicted = np.zeros((b_sz, input_length), dtype=int)
    true = np.zeros((b_sz, input_length), dtype=int)
    test_dim = len(test_in)

    for (ids, masks), labels in zip(test_in, test_out):
        labels = labels[0]
        ids = ids.to(gpu_id)
        masks = masks.to(gpu_id)
        labels = labels.to(gpu_id)
        effective_batch_size = list(ids.size())[0]

        logits = model(ids, masks, effective_batch_size)
        predicted_output = torch.argmax(logits, dim=-1)

        loss_fun = torch.nn.CrossEntropyLoss(reduction='none').to(gpu_id)
        logits = torch.transpose(logits, dim0=1, dim1=2)
        loss_masked = loss_fun(logits, labels)
        pad = -100
        loss_mask = labels != pad
        loss = loss_masked.sum() / loss_mask.sum()
        loss_sum += loss.item()

        predicted_labels = predicted_output.numpy(force=True)
        true_labels = labels.numpy(force=True)
        predicted = np.concatenate([list(predicted), list(predicted_labels)])
        true = np.concatenate([list(true), list(true_labels)])

    dropping_rows = list(range(b_sz))
    predicted = np.delete(predicted, dropping_rows, 0)
    true = np.delete(true, dropping_rows, 0)
    metrics, cm = scoring(true, predicted)

    return loss_sum / test_dim, metrics, cm
