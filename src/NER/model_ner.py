import numpy as np
import torch.nn
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup


def model_bert(model_name):
    model = BertModel.from_pretrained(model_name)

    # freeze the bert parameters, train just the classifier
    for param in model.parameters():
        param.requires_grad = False

    return model


class NerModel(torch.nn.Module):

    def __init__(self, model_name, input_size):
        super(NerModel, self).__init__()

        self.hidden_size = 768
        self.input_size = input_size
        self.bert = model_bert(model_name)

        # bert head for NER
        padding = (0, 0)
        dilation = (1, 1)
        kernel_size = (16, 128)
        stride = (2, 2)
        h_in = 768 * 4
        w_in = input_size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride)
        h_out = int(np.floor((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
        w_out = int(np.floor((w_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
        self.conv_swish1 = torch.nn.SiLU()
        kernel_size = (8, 64)
        stride = (1, 1)
        h_in = h_out
        w_in = w_out
        self.conv_max_pool1 = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        h_out = int(np.floor((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
        w_out = int(np.floor((w_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)

        kernel_size = (16, 128)
        stride = (2, 2)
        h_in = h_out
        w_in = w_out
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride)
        h_out = int(np.floor((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
        w_out = int(np.floor((w_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
        self.conv_swish2 = torch.nn.SiLU()
        kernel_size = (8, 64)
        stride = (1, 1)
        h_in = h_out
        w_in = w_out
        self.conv_max_pool2 = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        h_out = int(np.floor((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
        w_out = int(np.floor((w_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)

        kernel_size = (16, 128)
        stride = (2, 2)
        h_in = h_out
        w_in = w_out
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride)
        h_out = int(np.floor((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
        w_out = int(np.floor((w_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
        self.conv_swish3 = torch.nn.SiLU()
        h_in = h_out
        w_in = w_out
        self.conv_max_pool3 = torch.nn.AdaptiveMaxPool2d((35, 10))
        h_out = 10
        w_out = 35

        self.conv_flatten = torch.nn.Flatten()
        linear_in_size = h_out * w_out
        self.conv_linear = torch.nn.Linear(in_features=linear_in_size, out_features=self.input_size * 5)
        self.conv_linear_gelu = torch.nn.GELU()

        self.softmax = torch.nn.Softmax(dim=-1)

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(NerModel.parameters(self),
                                      lr=1e-2,  # args.learning_rate - default is 5e-5
                                      eps=1e-7  # args.adam_epsilon  - default is 1e-8.
                                      )
        return optimizer

    def get_scheduler(self, num_steps_training):
        scheduler = get_linear_schedule_with_warmup(
            self.get_optimizer(),
            num_warmup_steps=0,
            num_training_steps=num_steps_training
        )
        return scheduler

    def __bert_head(self, bert_output, effective_batch_size):

        bert_output_shape = list(bert_output.size())
        conv_input = torch.reshape(bert_output, (bert_output_shape[0], 1, bert_output_shape[1], bert_output_shape[2]))
        conv_output = self.conv1(conv_input)
        conv_output = self.conv_swish1(conv_output)
        conv_output = self.conv_max_pool1(conv_output)
        conv_output = self.conv2(conv_output)
        conv_output = self.conv_swish2(conv_output)
        conv_output = self.conv_max_pool2(conv_output)
        conv_output = self.conv3(conv_output)
        conv_output = self.conv_swish3(conv_output)
        conv_output = self.conv_max_pool3(conv_output)
        flatten_out = self.conv_flatten(conv_output)
        linear_out = self.conv_linear(flatten_out)
        logits = self.conv_linear_gelu(linear_out)

        logits = torch.reshape(logits, shape=(effective_batch_size, self.input_size, 5))
        entities_distribution = self.softmax(logits)
        entities_vector = torch.argmax(entities_distribution, dim=-1)

        return logits, entities_vector

    def forward(self, ids, mask, effective_batch_size):
        bert_output = self.bert(ids, attention_mask=mask, return_dict=False, output_hidden_states=True)

        # concatenate the last four hidden states
        bert_output = bert_output[2]
        num_hidden_states = len(bert_output)
        bert_output = [bert_output[num_hidden_states - 1 - 1], bert_output[num_hidden_states - 1 - 2],
                       bert_output[num_hidden_states - 1 - 3], bert_output[num_hidden_states - 1 - 4]]
        bert_output = torch.concat(bert_output, dim=-1)

        logits, entities_vector = self.__bert_head(bert_output, effective_batch_size)

        return logits, entities_vector
