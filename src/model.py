from transformers import TFAutoModelForTokenClassification
from transformers import create_optimizer

import tensorflow as tf

def model_ner(model_name, len_labels, id_label, label_id):
    return TFAutoModelForTokenClassification.from_pretrained(model_name, num_labels=len_labels,
                                                             id2label=id_label, label2id=label_id)


def get_optimizer():
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


def create_model(bert_model, len_labels, id_label, label_id, max_input_len):
    ner_model = model_ner(bert_model, len_labels, id_label, label_id)
    input_ids = tf.keras.Input(shape=(max_input_len,), dtype=tf.int32)
    input_mask = tf.keras.Input(shape=(max_input_len,), dtype=tf.int32)
    input_labels = tf.keras.Input(shape=(max_input_len,), dtype=tf.int32)
    ner_output = ner_model(input_ids, attention_mask=input_mask, labels=input_labels, return_dict=True)
    x = tf.keras.layers.Dense(10, activation='relu')(ner_output['last_hidden_state'])
    output = tf.keras.layers.Dense(2, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[input_ids, input_mask, input_labels], outputs=[output])
    optimizer = get_optimizer()
    model.compile(optimizer=optimizer)

    return model
