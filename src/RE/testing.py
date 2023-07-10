import torch
from torch.utils.data import DataLoader, DistributedSampler


def test_re(test_input, test_output, model, batch_size, world_size, rank, max_number_pos):
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

    test_loss = test(test_in_loader, test_out_loader, model, max_number_pos, rank)
    print(f'------------------Test RE loss: {test_loss}------------------\n\n')


def test(test_in, test_out, model, max_number_pos, gpu_id):
    b_sz = len(next(iter(test_in))[0])
    model.eval()
    loss_sum = 0
    embedding = torch.nn.Embedding(max_number_pos, 768, padding_idx=0).to(gpu_id)

    for (ids, masks, pos), out in zip(test_in, test_out):
        out = out[0]
        ids.to(gpu_id)
        masks.to(gpu_id)
        pos = pos.to(gpu_id)
        out = out.to(gpu_id)
        effective_batch_size = list(ids.size())[0]
        predicted_output = model(ids, masks, pos, embedding, effective_batch_size)
        predicted_output = torch.transpose(predicted_output, dim0=1, dim1=2)
        loss_fun = torch.nn.CrossEntropyLoss().to(gpu_id)
        loss = loss_fun(predicted_output, out)
        loss_sum += loss.item() / b_sz

    return loss_sum / len(test_in)
