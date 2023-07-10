import torch
from torch.utils.data import DataLoader, DistributedSampler


def test_final(model, inputs, outputs, batch_size, world_size, rank):
    input_loader = DataLoader(inputs,
                              batch_size=batch_size,
                              pin_memory=True,
                              shuffle=False,
                              sampler=DistributedSampler(inputs,
                                                         num_replicas=world_size,
                                                         rank=rank))
    output_loader = DataLoader(outputs,
                               batch_size=batch_size,
                               pin_memory=True,
                               shuffle=False,
                               sampler=DistributedSampler(outputs,
                                                          num_replicas=world_size,
                                                          rank=rank))

    loss = test(input_loader, output_loader, model, rank)
    print(f'------------------Final Model loss: {loss}------------------\n\n')


def test(inputs, outputs, model, gpu_id):
    b_sz = len(next(iter(inputs))[0])
    model.eval()
    loss_sum = 0

    for (ids, masks), out in zip(inputs, outputs):
        out = out[0]
        ids.to(gpu_id)
        masks.to(gpu_id)
        out = out.to(gpu_id)
        predicted_output = model(ids, masks)
        predicted_output = torch.transpose(predicted_output, dim0=1, dim1=2)
        loss_fun = torch.nn.CrossEntropyLoss().to(gpu_id)
        loss = loss_fun(predicted_output, out)
        loss_sum += loss.item() / b_sz

    return loss_sum / len(inputs)
