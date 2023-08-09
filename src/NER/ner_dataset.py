import torch


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, input_masks, labels):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'ids': self.input_ids[idx],
            'mask': self.input_masks[idx],
            'labels': self.labels[idx]
        }
