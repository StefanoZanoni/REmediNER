import torch
from transformers import BertModel


def model_bert(model_name):
    model = BertModel.from_pretrained(model_name)
    return model


class ReModel(torch.nn.Module):

    def __init__(self, bert_name, input_size, loss_weights):
        super(ReModel, self).__init__()

        self.hidden_size = 768
        self.input_size = input_size
        self.loss_weights = loss_weights
        self.bert = model_bert(bert_name)

        # ------- BERT HEAD ------- #
        self.dropout = torch.nn.Dropout(0.4)

        # Bi-directional LSTM
        lstm_hidden_size = 128
        self.lstm = torch.nn.LSTM(input_size=self.hidden_size * 4, hidden_size=lstm_hidden_size, num_layers=1,
                                  batch_first=True, bidirectional=True)

        # Dimensionality reduction layer
        reduced_size = self.hidden_size // 2
        self.dim_reduction = torch.nn.Linear(in_features=self.input_size * lstm_hidden_size * 2,
                                             out_features=reduced_size)

        # Output layer
        self.final_linear = torch.nn.Linear(in_features=reduced_size, out_features=self.input_size * 5)
        self.gelu = torch.nn.GELU()

    def __bert_head(self, bert_output, effective_batch_size):
        bert_output = self.dropout(bert_output)
        bilstm_out = self.lstm(bert_output)[0]
        flatten_out = torch.nn.Flatten()(bilstm_out)
        reduced_out = self.dim_reduction(flatten_out)
        logits = self.final_linear(reduced_out)
        logits = self.gelu(logits)
        logits = torch.reshape(logits, shape=(effective_batch_size, self.input_size, 5))

        return logits

    def forward(self, ids, mask, labels):
        bert_output = self.bert(ids, attention_mask=mask, return_dict=False, output_hidden_states=True)

        # concatenate the last four hidden states
        bert_output = bert_output[2]
        num_hidden_states = len(bert_output)
        bert_output = [bert_output[num_hidden_states - 1 - 1], bert_output[num_hidden_states - 1 - 2],
                       bert_output[num_hidden_states - 1 - 3], bert_output[num_hidden_states - 1 - 4]]
        bert_output = torch.concat(bert_output, dim=-1)
        effective_batch_size = list(labels.size())[0]

        logits = self.__bert_head(bert_output, effective_batch_size)

        return {'logits': logits}
