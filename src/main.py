import os
from ast import literal_eval

import pandas as pd
import torch
import torch.multiprocessing as mp
import transformers
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import TensorDataset
from transformers import BertTokenizerFast

from src.data_utilities import load_data, pre_process_texts
from src.NER.data_utilities_ner import compute_iob, get_labels_id, split_train_test_ner,\
    tokenize_text_ner, get_ner_inputs
from src.RE.data_utilities_re import prepare_data_for_re, compute_pos, compute_context_mean_length, tokenize_text_re, \
    split_train_test_re, get_re_inputs
from src.NER.training_ner import TrainerNer
from src.RE.training_re import TrainerRe

bert_model = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = BertTokenizerFast.from_pretrained(bert_model)

transformers.utils.logging.set_verbosity_error()


def ddp_setup(rank: int, world_size: int):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank, world_size, save_every, total_epochs=10, batch_size=32):

    ddp_setup(rank, world_size)

    if not os.path.exists("../data/ner.csv"):
        data = load_data()
        pre_process_texts(data)
        compute_iob(data)
        data.to_csv("../data/ner.csv", index=False)
    else:
        data = pd.read_csv("../data/ner.csv")

    if not os.path.exists("../data/re.csv"):
        data_re = prepare_data_for_re(data)
        compute_pos(data_re)
        data_re.to_csv("../data/re.csv", index=False)
    else:
        data_re = pd.read_csv("../data/re.csv", converters={"annotated_text": literal_eval, 'pos_tags': literal_eval})

    id_label, label_id, len_labels = get_labels_id(data)

    # NER data
    train_in_ner, test_in_ner, train_out_ner, test_out_ner = split_train_test_ner(data)

    # train data
    tokenized_texts_train_ner, tokenized_labels_train_ner = tokenize_text_ner(train_in_ner, train_out_ner, tokenizer)
    ner_ids, ner_masks, ner_labels = \
        get_ner_inputs(tokenized_texts_train_ner, tokenized_labels_train_ner, tokenizer, label_id)
    inputs_train_ner = TensorDataset(ner_ids, ner_masks)
    outputs_train_ner = TensorDataset(ner_labels)

    # test data
    # tokenized_texts_test_ner, tokenized_labels_test_ner = tokenize_text_ner(test_in_ner, test_out_ner, tokenizer)
    # inputs_test_ner = get_bert_inputs(tokenized_texts_test_ner, tokenized_labels_test_ner, tokenizer, label_id)
    # inputs_test_ner = TensorDataset(inputs_test_ner[0], inputs_test_ner[1])
    # outputs_test_ner = TensorDataset(inputs_test_ner[2])

    # NER training
    global bert_model
    bert_model = {'bert_model': bert_model,
                  'len_labels': len_labels,
                  'id_label': id_label,
                  'label_id': label_id}
    ner_trainer = TrainerNer(bert_model, inputs_train_ner, outputs_train_ner, total_epochs, batch_size, rank, save_every, world_size)
    model_ner = ner_trainer.kfold_cross_validation(k=2)

    # RE data
    tokenized_texts, tokenized_annotation, tokenized_pos = tokenize_text_re(data_re, tokenizer)
    train_in_re, test_in_re, train_out_re, test_out_re = split_train_test_re(tokenized_texts, tokenized_annotation)

    # train data
    train_re_ids, train_re_masks, train_re_annotations = get_re_inputs(train_in_re, train_out_re, tokenizer)
    inputs_train_re = TensorDataset(train_re_ids, train_re_masks)
    outputs_train_re = TensorDataset(train_re_annotations)

    # test data
    test_re_ids, test_re_masks, test_re_annotations = get_re_inputs(test_in_re, test_out_re, tokenizer)
    inputs_test_re = TensorDataset(test_re_ids, test_re_masks)
    outputs_test_re = TensorDataset(test_re_annotations)
    # context_mean_length = compute_context_mean_length(data_re)

    # RE training
    re_trainer = TrainerRe(model_ner, context_mean_length, 768, label_id, inputs_train_re, outputs_train_re,
                           total_epochs, batch_size, rank, save_every, world_size)
    re_trainer.kfold_cross_validation(k=2)

    destroy_process_group()


if __name__ == '__main__':
    import sys

    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    save_every = int(sys.argv[3])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, save_every, epochs, batch_size,), nprocs=world_size)
