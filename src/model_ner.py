import torch.nn
from transformers import AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup


def model_ner(model_name, len_labels, id_label, label_id):
    return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len_labels,
                                                           id2label=id_label, label2id=label_id)


class NerModel(torch.nn.Module):

    def __init__(self, model_name, len_labels, id_label, label_id):
        super(NerModel, self).__init__()

        self.ner = model_ner(model_name, len_labels, id_label, label_id)
        self.entities = torch.nn.Softmax(dim=-1)

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(NerModel.parameters(self),
                                      lr=1e-5,  # args.learning_rate - default is 5e-5
                                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                      )
        return optimizer

    def get_scheduler(self, num_steps_training):
        scheduler = get_linear_schedule_with_warmup(
            self.get_optimizer(),
            num_warmup_steps=0,
            num_training_steps=num_steps_training
        )
        return scheduler

    def forward(self, ids, mask, labels):
        bert_output = self.ner(ids, attention_mask=mask, labels=labels, return_dict=False, output_hidden_states=True)
        loss = bert_output[0]
        logits = bert_output[1]
        entities_vector = self.entities(logits)
        hidden_states = bert_output[2]
        num_hidden_states = len(hidden_states)
        last_4_hidden_states = [hidden_states[num_hidden_states - 1 - 1], hidden_states[num_hidden_states - 1 - 2],
                                hidden_states[num_hidden_states - 1 - 3], hidden_states[num_hidden_states - 1 - 4]]
        entities_context = torch.concat(last_4_hidden_states, dim=-1)
        return loss, entities_vector, entities_context
