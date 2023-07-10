import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.NER.training_ner import create_mean_dict, scoring, compute_metrics_mean
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
    b_sz = len(next(iter(test_in))[0])
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
        logits, entities_vector = model(ids, masks)
        loss_fun = torch.nn.CrossEntropyLoss().to(gpu_id)
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

    compute_metrics_mean(mean_dict, metrics_dict, dim=test_dim)

    return loss_sum / test_dim, mean_dict, mean_cm / test_dim
