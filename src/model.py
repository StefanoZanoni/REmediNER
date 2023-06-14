from transformers import TFAutoModelForTokenClassification
from transformers import create_optimizer

import keras


def model_ner(model_name, labels):
    id_label = {i: label for i, label in enumerate(labels)}
    label_id = {label: i for i, label in enumerate(labels)}

    model = TFAutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels),
                                                              id2label=id_label, label2id=label_id)
    return model


def get_optimizer(num_epochs, train_steps):
    optimizer, _ = create_optimizer(
        # https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it
        # num_train_steps=train_steps, #training step is one gradient update
        init_lr=2e-5,
        weight_decay_rate=0.01,  # https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
        num_warmup_steps=0,
        # https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps
    )
    return optimizer
    # config the model with metrics and losses


def create_model(model_ner, optimizer):
    # model.compile(optimizer=optimizer)
    return
