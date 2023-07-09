import torch


class FinalModel(torch.nn.Module):

    def __init__(self, ner, re):
        super(FinalModel, self).__init__()

        self.ner = ner
        self.re = re

    def forward(self, ids, masks, labels, id_label):
        _, entities = self.ner(ids, masks)

        output_re = self.re(output_ner)

        return output_re
