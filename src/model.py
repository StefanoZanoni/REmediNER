from transformers import TFAutoModelForTokenClassification
from transformers import create_optimizer

import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def model_ner(model_name, len_labels, id_label, label_id):
    return TFAutoModelForTokenClassification.from_pretrained(model_name, num_labels=len_labels,
                                                             id2label=id_label, label2id=label_id)


def get_optimizer():
    optimizer, _ = create_optimizer(
        # https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it
        num_train_steps=10,  # training step is one gradient update
        init_lr=2e-5,
        weight_decay_rate=0.01,  # https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
        num_warmup_steps=0,
        # https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps
    )
    return optimizer
    # config the model with metrics and losses


def create_model(bert_model, len_labels, id_label, label_id, num_drugs, num_effects, max_input_len=128):
    ner_model = model_ner(bert_model, len_labels, id_label, label_id)
    input_ids = tf.keras.Input(shape=(max_input_len,), dtype=tf.int32)
    input_mask = tf.keras.Input(shape=(max_input_len,), dtype=tf.int32)
    input_labels = tf.keras.Input(shape=(max_input_len,), dtype=tf.int32)
    ner_output = ner_model(input_ids, attention_mask=input_mask, labels=input_labels, return_dict=True)
    x = tf.keras.layers.Dense(20, activation='relu')(tf.concat(ner_output['hidden_states'][-4:], axis=-1))
    output_d = tf.keras.layers.Dense(num_drugs, activation='softmax')(x)
    output_d = tf.keras.layers.Lambda \
        (lambda x: tf.keras.backend.cast(tf.keras.backend.argmax(x), dtype='float32'), name='y_pred1')(output_d)
    output_e = tf.keras.layers.Dense(num_effects, activation='softmax')(x)
    output_e = tf.keras.layers.Lambda \
        (lambda x: tf.keras.backend.cast(tf.keras.backend.argmax(x), dtype='float32'), name='y_pred2')(output_e)
    model = tf.keras.models.Model(inputs=[input_ids, input_mask, input_labels], outputs=[output_d, output_e])
    # optimizer = get_optimizer()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.summary()

    return model
