import torch.nn
from transformers import AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup


def model_ner(model_name, len_labels, id_label, label_id):
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len_labels,
                                                            id2label=id_label, label2id=label_id)
    # freeze the bert parameters, train just the classifier
    for param in model.bert.parameters():
        param.requires_grad = False

    return model


class NerModel(torch.nn.Module):

    def __init__(self, model_name, len_labels, id_label, label_id):
        super(NerModel, self).__init__()

        self.ner = model_ner(model_name, len_labels, id_label, label_id)
        self.entities = torch.nn.Softmax(dim=-1)

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(NerModel.parameters(self),
                                      lr=1e-4,  # args.learning_rate - default is 5e-5
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

    def forward(self, ids, mask):
        bert_output = self.ner(ids, attention_mask=mask, return_dict=False, output_hidden_states=True)
        logits = bert_output[0]
        entities_distribution = self.entities(logits)
        entities_vector = torch.argmax(entities_distribution, dim=-1)

        return logits, entities_vector
