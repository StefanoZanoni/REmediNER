import torch

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


def training_ner(train_d_loader, model, optimizer, device, scheduler):
    model.train()
    loss_results_train = 0
    for data in tqdm(train_d_loader):
        for k, v in data.items():
            data[k] = v.to(device)
        # backward propagation
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_results_train += loss.item()
    return model, loss_results_train / len(train_d_loader)


def validation_ner(val_d_loader, model, optimizer, device, scheduler):
    model.eval()  # for validation - not .train()
    loss_results_val = 0
    for data in tqdm(val_d_loader):
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss = model(**data)
        loss_results_val += loss.item()
    return loss_results_val / len(val_d_loader)


class TrainerNer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 gpu_id: int,
                 save_every: int
                 ) -> None:
        self.gpu_id = gpu_id
        self.model = DDP(model, device_ids=[gpu_id])
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_evey = save_every

    def _run_batch_ner(self, source):
        self.optimizer.zero_grad()
        loss = self.model(source)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def _run_epoch_ner(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch-size: {b_sz} | Steps {len(self.train_data)}")
        loss_batch = 0
        for source in self.train_data:
            source = source.to(self.gpu_id)
            loss_batch += self._run_batch_ner(source)
        return loss_batch / len(self.train_data)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"Epoch {epoch} | Training checkpoint saved at checkpoint.pt")

    def train_ner(self, max_epochs: int):
        epoch_loss_means = []
        for epoch in range(max_epochs):
            epoch_loss_mean = self._run_epoch_ner(epoch)
            epoch_loss_means.append(epoch_loss_mean)
            if self.gpu_id == 0 and epoch % self.save_evey == 0:
                self._save_checkpoint(epoch)
        return epoch_loss_means
