import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.NER.training_ner import create_mean_dict, scoring, compute_metrics_mean, compute_batch_weights
from src.plot import plot_heat_map


def test_ner(test_input, test_output, model, batch_size, world_size, rank, id_label):
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

    test_loss, test_metrics, test_cm = test(test_in_loader, test_out_loader, model, id_label, rank)
    print(f'------------------Test NER loss: {test_loss}------------------\n\n')
    print(f'------------------Test NER metrics: {test_metrics}------------------\n\n')
    plot_heat_map(test_cm)


def test(test_in, test_out, model, id_label, gpu_id):
    model.eval()
    loss_sum = 0
    mean_dict = create_mean_dict()
    mean_cm = np.zeros((len(id_label), len(id_label)))
    test_dim = len(test_in)

    for (ids, masks), labels in zip(test_in, test_out):
        labels = labels[0]
        ids = ids.to(gpu_id)
        masks = masks.to(gpu_id)
        labels = labels.to(gpu_id)
        effective_batch_size = list(ids.size())[0]
        logits, entities_vector = model(ids, masks, effective_batch_size)
        class_weights = compute_batch_weights(labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        loss_fun = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none').to(gpu_id)
        logits = torch.transpose(logits, dim0=1, dim1=2)
        loss_masked = loss_fun(logits, labels)
        pad = -100
        loss_mask = labels != pad
        loss = loss_masked.sum() / loss_mask.sum()
        loss_sum += loss.item()
        logits = torch.transpose(logits, dim0=1, dim1=2)
        predicted_output = torch.argmax(logits, dim=-1)
        predicted_labels = predicted_output.numpy(force=True)
        true_labels = labels.numpy(force=True)
        metrics_dict, cm = scoring(true_labels, predicted_labels)
        compute_metrics_mean(mean_dict, metrics_dict)
        mean_cm += cm

    compute_metrics_mean(mean_dict, metrics_dict, dim=test_dim)

    return loss_sum / test_dim, mean_dict, mean_cm / test_dim
