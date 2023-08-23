import torch


class FinalDataset(torch.utils.data.Dataset):
    def __init__(self, ner_ids, ner_masks, re_labels):
        self.ner_ids = ner_ids
        self.ner_masks = ner_masks
        self.re_labels = re_labels

    def __len__(self):
        return len(self.ner_ids)

    def __getitem__(self, idx):
        return {
            'ids': self.ner_ids[idx],
            'mask': self.ner_masks[idx],
            'labels': self.re_labels[idx]
        }
