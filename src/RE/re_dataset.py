import torch


class REDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, input_masks, input_annotations):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.input_annotations = input_annotations

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'ids': self.input_ids[idx],
            'mask': self.input_masks[idx],
            'labels': self.input_annotations[idx]
        }
