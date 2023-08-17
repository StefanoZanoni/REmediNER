import numpy as np
import torch
from transformers import AutoModelForTokenClassification


def model_bert(model_name, id_label, label_id):
    model = AutoModelForTokenClassification.from_pretrained(model_name, id2label=id_label, label2id=label_id)
    return model


class NerModel(torch.nn.Module):

    def __init__(self, model_name, input_size, id_label, label_id, loss_weights):
        super(NerModel, self).__init__()

        self.input_size = input_size
        self.bert = model_bert(model_name, id_label, label_id)
        self.loss_weights = loss_weights

    def forward(self, ids, mask, labels):
        bert_output = self.bert(ids, attention_mask=mask, return_dict=False)
        logits = bert_output[0]

        return {'logits': logits}
