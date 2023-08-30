import sys
import os
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import transformers
from torch.utils.data import TensorDataset
from transformers import BertTokenizerFast
from torchsummary import summary

from src.data_utilities import load_data, pre_process_texts, split_train_test, split_test, compute_weights

from src.NER.data_utilities_ner import (compute_iob, get_labels_id, tokenize_text_ner, get_ner_inputs,
                                        prepare_data_for_ner)
from src.NER.train_eval_ner import train_test_ner
from src.NER.ner_dataset import NERDataset

from src.RE.data_utilities_re import prepare_data_for_re, tokenize_text_re, get_re_inputs
from src.RE.re_dataset import REDataset
from src.RE.train_eval_re import train_test_re

from src.FINALMODEL.test_final import test_final


bert_name_ner = 'bert-base-cased'
bert_name_re = 'bert-base-cased'
tokenizer_ner = BertTokenizerFast.from_pretrained(bert_name_ner)
tokenizer_re = BertTokenizerFast.from_pretrained(bert_name_re)

transformers.utils.logging.set_verbosity_error()


def train_re(data, epochs, batch_size, input_length, train_indices, val_indices, test_indices):
    # RE data
    train_data = data.iloc[train_indices]
    val_data = data.iloc[val_indices]
    test_data = data.iloc[test_indices]
    train_in_re, train_out_re = tokenize_text_re(train_data, tokenizer_re)
    val_in_re, val_out_re = tokenize_text_re(val_data, tokenizer_re)
    test_in_re_final, test_out_re_final = tokenize_text_re(test_data, tokenizer_re)

    # training data
    train_re_ids, train_re_masks, train_re_annotations = \
        get_re_inputs(train_in_re, train_out_re, tokenizer_re, input_length)
    loss_weights_train = torch.tensor(compute_weights(train_re_annotations.numpy()), dtype=torch.float32)
    train_re_dataset = REDataset(train_re_ids, train_re_masks, train_re_annotations)

    # validation data
    val_re_ids, val_re_masks, val_re_annotations = \
        get_re_inputs(val_in_re, val_out_re, tokenizer_re, input_length)
    loss_weights_val = torch.tensor(compute_weights(val_re_annotations.numpy()), dtype=torch.float32)
    val_re_dataset = REDataset(val_re_ids, val_re_masks, val_re_annotations)

    # RE training
    re_model = train_test_re(bert_name_re, train_re_dataset, val_re_dataset, input_length,
                             batch_size, epochs, loss_weights_train, loss_weights_val)
    summary(re_model,
            input_size=[(batch_size, input_length), (batch_size, input_length), (batch_size, input_length)],
            dtypes=['torch.IntTensor', 'torch.IntTensor', 'torch.IntTensor'])

    # final test data
    _, _, test_re_annotations = \
        get_re_inputs(test_in_re_final, test_out_re_final, tokenizer_re, input_length)

    return re_model, test_re_annotations


def train_ner(data, epochs, batch_size, input_length, train_indices, val_indices, test_indices):
    # NER data
    id_label, label_id, len_labels = get_labels_id()
    train_data = data.iloc[train_indices]
    val_data = data.iloc[val_indices]
    test_data = data.iloc[test_indices]
    train_in_ner = train_data['text'].to_frame()
    train_out_ner = train_data['iob'].to_frame()
    val_in_ner = val_data['text'].to_frame()
    val_out_ner = val_data['iob'].to_frame()
    test_in_ner_final = test_data['text'].to_frame()
    test_out_ner_final = test_data['iob'].to_frame()

    # training data
    tokenized_texts_train_ner, tokenized_labels_train_ner = tokenize_text_ner(train_in_ner, train_out_ner,
                                                                              tokenizer_ner)
    ner_ids, ner_masks, ner_labels = \
        get_ner_inputs(tokenized_texts_train_ner, tokenized_labels_train_ner, tokenizer_ner, label_id, input_length)
    loss_weights_train = torch.tensor(compute_weights(ner_labels.numpy()), dtype=torch.float32)
    train_ner_dataset = NERDataset(ner_ids, ner_masks, ner_labels)

    # validation data
    tokenized_texts_val_ner, tokenized_labels_val_ner = tokenize_text_ner(val_in_ner, val_out_ner, tokenizer_ner)
    ner_ids, ner_masks, ner_labels = \
        get_ner_inputs(tokenized_texts_val_ner, tokenized_labels_val_ner, tokenizer_ner, label_id, input_length)
    loss_weights_val = torch.tensor(compute_weights(ner_labels.numpy()), dtype=torch.float32)
    val_ner_dataset = NERDataset(ner_ids, ner_masks, ner_labels)

    # NER training
    bert_model = {'bert_model': bert_name_ner,
                  'len_labels': len_labels,
                  'id_label': id_label,
                  'label_id': label_id}
    ner_model = (
        train_test_ner(bert_model, train_ner_dataset, val_ner_dataset, input_length, batch_size, epochs,
                       loss_weights_train, loss_weights_val))
    summary(ner_model,
            input_size=[(batch_size, input_length), (batch_size, input_length), (batch_size, input_length)],
            dtypes=['torch.IntTensor', 'torch.IntTensor', 'torch.IntTensor'])

    # final test data
    tokenized_texts_test_ner, tokenized_labels_test_ner = tokenize_text_ner(test_in_ner_final, test_out_ner_final,
                                                                            tokenizer_ner)
    ner_ids, ner_masks, _ = \
        get_ner_inputs(tokenized_texts_test_ner, tokenized_labels_test_ner, tokenizer_ner, label_id, input_length)

    return ner_model, ner_ids, ner_masks, id_label


def main(epochs=10, batch_size=32, ner_input_length=128, re_input_length=128):
    global bert_name_re
    global bert_name_ner

    # create the folder to store pre-processed data
    if not os.path.exists('../data'):
        os.makedirs('../data', exist_ok=True)

    # pre-process data for NER task
    if not os.path.exists("../data/ner.csv"):
        data_ner = load_data()
        pre_process_texts(data_ner)
        data_ner = prepare_data_for_ner(data_ner)
        compute_iob(data_ner)
        data_ner.to_csv("../data/ner.csv", index=False)
    # read already pre-processed data for NER task
    else:
        data_ner = pd.read_csv("../data/ner.csv", dtype={'drug': object, 'effect': object})
    # pre-process data for RE task
    if not os.path.exists("../data/re.csv"):
        data_re = load_data()
        pre_process_texts(data_re)
        data_re = prepare_data_for_re(data_re)
        data_re.to_csv("../data/re.csv", index=False)
    # read already pre-processed data for RE task
    else:
        data_re = pd.read_csv("../data/re.csv", converters={'annotated_text': literal_eval, 'pos_tags': literal_eval})

    # we split row indexes for create the same dataset splitting for both NER and RE.
    indices = np.arange(len(data_ner))
    train_indices, test_indices = split_train_test(indices)
    val_indices, test_indices = split_test(test_indices)

    # train the model for NER task
    ner_model, ner_ids, ner_masks, id_label = train_ner(data_ner, epochs, batch_size, ner_input_length,
                                                        train_indices, val_indices, test_indices)

    # train the model for RE task
    re_model, re_annotations = train_re(data_re, epochs, batch_size, re_input_length,
                                        train_indices, val_indices, test_indices)

    # test the final model
    test_final(ner_model, re_model, ner_ids, ner_masks, re_annotations,
               tokenizer_ner, id_label, re_input_length, batch_size)


if __name__ == '__main__':
    # read parameters from command line
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    ner_input_length = int(sys.argv[3])
    re_input_length = int(sys.argv[4])

    main(epochs, batch_size, ner_input_length, re_input_length)
