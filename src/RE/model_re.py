import torch
from transformers import BertModel


def model_bert(model_name):
    model = BertModel.from_pretrained(model_name)

    # freeze bert parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


class ReModel(torch.nn.Module):

    def __init__(self, bert_name, context_mean_length, batch_size, input_size):
        super(ReModel, self).__init__()

        self.context_mean_length = context_mean_length
        self.hidden_size = 768
        self.batch_size = batch_size
        self.input_size = input_size

        self.bert = model_bert(bert_name)

        # first piece of bert head: convolution + max pooling + convolution + max pooling + dense
        padding = (0, 0)
        dilation = (1, 1)
        kernel_size = (32, 512)
        stride = (1, 1)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size)
        self.conv_swish1 = torch.nn.SiLU()
        kernel_size = (4, 4)
        self.conv_max_pool1 = torch.nn.MaxPool2d(kernel_size=kernel_size)

        kernel_size = (8, 128)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size)
        self.conv_swish2 = torch.nn.SiLU()
        pool_h_out = 128
        self.conv_max_pool2 = torch.nn.AdaptiveMaxPool2d(output_size=(context_mean_length, pool_h_out))

        self.conv_flatten = torch.nn.Flatten()
        linear_in_size = int(pool_h_out * context_mean_length)
        self.conv_linear = torch.nn.Linear(in_features=linear_in_size, out_features=self.input_size * 5)
        self.conv_linear_relu = torch.nn.ReLU()

        # second piece of bert head: bilstm + dense

        # batch_first=True means batch should be our first dimension (Input Type 2)
        # otherwise if we do not define batch_first=True in RNN we need data in Input type 1 shape
        # (Sequence Length, Batch Size, Input Dimension).

        # LSTM layer
        lstm_hidden_size = 8
        self.lstm = torch.nn.LSTM(input_size=self.hidden_size * 5, hidden_size=lstm_hidden_size, num_layers=1,
                                  batch_first=True, bidirectional=True)

        self.lstm_flatten = torch.nn.Flatten()
        linear_in_size = self.input_size * lstm_hidden_size * 2
        self.lstm_linear = torch.nn.Linear(in_features=linear_in_size, out_features=self.input_size * 5)
        self.lstm_gelu = torch.nn.GELU()

        # final dense
        self.final_linear1 = torch.nn.Linear(in_features=self.input_size * 5 * 2, out_features=self.input_size * 5)
        self.final_linear1_elu = torch.nn.ELU()
        self.final_dropout = torch.nn.Dropout(p=0.3)
        self.final_linear2 = torch.nn.Linear(in_features=self.input_size * 5, out_features=self.input_size * 5)
        self.final_linear2_elu = torch.nn.ELU()
        self.final_softmax = torch.nn.Softmax(dim=-1)

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(ReModel.parameters(self),
                                      lr=1e-5,  # args.learning_rate - default is 5e-5
                                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                      )
        return optimizer

    def __bert_head(self, bert_output, pos, embedding, effective_batch_size):
        # conv computation
        bert_output_shape = list(bert_output.size())
        conv_input = torch.reshape(bert_output, (bert_output_shape[0], 1, bert_output_shape[1], bert_output_shape[2]))
        conv_output = self.conv1(conv_input)
        conv_output = self.conv_swish1(conv_output)
        conv_output = self.conv_max_pool1(conv_output)
        conv_output = self.conv2(conv_output)
        conv_output = self.conv_swish2(conv_output)
        conv_output = self.conv_max_pool2(conv_output)
        flatten_out = self.conv_flatten(conv_output)
        linear_out = self.conv_linear(flatten_out)
        conv_out = self.conv_linear_relu(linear_out)

        # bilstm computation
        pos_embedding = embedding(pos)
        lstm_input = torch.concat([bert_output, pos_embedding], dim=-1)
        bilstm_out = self.lstm(lstm_input)[0]
        bilstm_out = self.lstm_flatten(bilstm_out)
        linear_out = self.lstm_linear(bilstm_out)
        lstm_out = self.lstm_gelu(linear_out)

        # final computation
        final_input = torch.concat([conv_out, lstm_out], dim=-1)
        final_output1 = self.final_linear1(final_input)
        final_output1 = self.final_linear1_elu(final_output1)
        final_output1 = self.final_dropout(final_output1)
        final_output2 = self.final_linear2(final_output1)
        final_output2 = self.final_linear2_elu(final_output2)
        final_output2 = torch.reshape(final_output2, shape=(effective_batch_size, self.input_size, 5))
        re_output = self.final_softmax(final_output2)

        return re_output

    def forward(self, ids, masks, pos, embedding, effective_batch_size):
        bert_output = self.bert(ids, masks, output_hidden_states=True)

        # concatenate the last four hidden states
        bert_output = bert_output.hidden_states
        num_hidden_states = len(bert_output)
        bert_output = [bert_output[num_hidden_states - 1 - 1], bert_output[num_hidden_states - 1 - 2],
                       bert_output[num_hidden_states - 1 - 3], bert_output[num_hidden_states - 1 - 4]]
        bert_output = torch.concat(bert_output, dim=-1)

        output = self.__bert_head(bert_output, pos, embedding, effective_batch_size)

        return output
