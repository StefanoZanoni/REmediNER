import torch
from transformers import BertModel


def model_bert(model_name):
    model = BertModel.from_pretrained(model_name)

    # freeze bert parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


class ReModel(torch.nn.Module):

    def __init__(self, bert_name, batch_size, input_size):
        super(ReModel, self).__init__()

        self.hidden_size = 768
        self.batch_size = batch_size
        self.input_size = input_size

        self.bert = model_bert(bert_name)

        # bert head for RE
        lstm_hidden_size = 8
        self.lstm = torch.nn.LSTM(input_size=self.hidden_size * 5, hidden_size=lstm_hidden_size, num_layers=1,
                                  batch_first=True, bidirectional=True)

        self.lstm_flatten = torch.nn.Flatten()
        linear_in_size = self.input_size * lstm_hidden_size * 2
        self.linear = torch.nn.Linear(in_features=linear_in_size, out_features=self.input_size * 5)
        self.gelu = torch.nn.GELU()

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(ReModel.parameters(self),
                                      lr=1e-5,  # args.learning_rate - default is 5e-5
                                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                      )
        return optimizer

    def __bert_head(self, bert_output, pos, embedding, effective_batch_size):
        # bilstm computation
        pos_embedding = embedding(pos)
        lstm_input = torch.concat([bert_output, pos_embedding], dim=-1)
        bilstm_out = self.lstm(lstm_input)[0]
        flatten_out = self.lstm_flatten(bilstm_out)
        linear_out = self.linear(flatten_out)
        logits = self.gelu(linear_out)

        logits = torch.reshape(logits, shape=(effective_batch_size, self.input_size, 5))

        return logits

    def forward(self, ids, masks, pos, embedding, effective_batch_size):
        bert_output = self.bert(ids, attention_mask=masks, return_dict=False, output_hidden_states=True)

        # concatenate the last four hidden states
        bert_output = bert_output[2]
        num_hidden_states = len(bert_output)
        bert_output = [bert_output[num_hidden_states - 1 - 1], bert_output[num_hidden_states - 1 - 2],
                       bert_output[num_hidden_states - 1 - 3], bert_output[num_hidden_states - 1 - 4]]
        bert_output = torch.concat(bert_output, dim=-1)

        output = self.__bert_head(bert_output, pos, embedding, effective_batch_size)

        return output
