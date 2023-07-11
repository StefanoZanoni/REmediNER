import os
from ast import literal_eval

import pandas as pd
import torch
import torch.multiprocessing as mp
import transformers
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import TensorDataset
from transformers import BertTokenizerFast
from torchsummary import summary

from src.NER.testing_ner import test_ner
from src.RE.testing_re import test_re
from src.data_utilities import load_data, pre_process_texts
from src.NER.data_utilities_ner import compute_iob, get_labels_id, split_train_test_ner, \
    tokenize_text_ner, get_ner_inputs, prepare_data_for_ner, split_test_ner
from src.RE.data_utilities_re import prepare_data_for_re, compute_pos, compute_context_mean_length, tokenize_text_re, \
    split_train_test_re, get_re_inputs, compute_pos_indexes, split_test_re
from src.NER.training_ner import TrainerNer
from src.RE.training_re import TrainerRe

# bert_name_ner = "d4data/biomedical-ner-all"
# bert_name_ner = "ukkendane/bert-medical-ner"
# bert_name_ner = 'emilyalsentzer/Bio_ClinicalBERT'
# bert_name_ner = "bert-base-uncased"
bert_name_ner = "bert-base-cased"
# bert_name_ner = "bert-large-uncased"
# bert_name_ner = "bert-large-cased"
bert_name_re = 'bert-base-cased'
tokenizer_ner = BertTokenizerFast.from_pretrained(bert_name_ner)
tokenizer_re = BertTokenizerFast.from_pretrained(bert_name_re)

transformers.utils.logging.set_verbosity_error()


def train_re(data_re, epochs, batch_size, rank, save_every, world_size, input_length):
    # RE data
    tokenized_texts, tokenized_annotation, tokenized_pos = tokenize_text_re(data_re, tokenizer_re)
    pos_indexes, max_number_pos = compute_pos_indexes(tokenized_pos, input_length)
    train_in_texts_re, train_in_pos_re, test_in_texts_re, test_in_pos_re, \
        train_out_re, test_out_re = split_train_test_re(tokenized_texts, pos_indexes, tokenized_annotation)
    test_in_texts_re, test_in_pos_re, test_in_texts_re_final, test_in_pos_re_final, test_out_re, test_out_re_final = \
        split_test_re(test_in_texts_re, test_in_pos_re, test_out_re)

    # train data
    train_re_ids, train_re_masks, train_re_annotations = \
        get_re_inputs(train_in_texts_re, train_out_re, tokenizer_re, input_length)
    train_in_pos_re = torch.tensor(train_in_pos_re, dtype=torch.int32)
    inputs_train_re = TensorDataset(train_re_ids, train_re_masks, train_in_pos_re)
    outputs_train_re = TensorDataset(train_re_annotations)

    # test data
    test_re_ids, test_re_masks, test_re_annotations = \
        get_re_inputs(test_in_texts_re, test_out_re, tokenizer_re, input_length)
    test_in_pos_re = torch.tensor(test_in_pos_re, dtype=torch.int32)
    inputs_test_re = TensorDataset(test_re_ids, test_re_masks, test_in_pos_re)
    outputs_test_re = TensorDataset(test_re_annotations)

    context_mean_length = compute_context_mean_length(data_re)

    # RE training
    re_trainer = TrainerRe(bert_name_re, inputs_train_re, outputs_train_re, epochs, batch_size, rank, save_every,
                           world_size, max_number_pos, input_length)
    re_output, max_epoch = re_trainer.kfold_cross_validation(k=5)
    # retrain on the whole development set
    re_model = re_trainer.re_train(max_epoch)
    summary(re_model,
            input_size=[(batch_size, input_length), (batch_size, input_length), (batch_size, input_length), 1,
                        batch_size],
            dtypes=['torch.IntTensor', 'torch.IntTensor', 'torch.IntTensor', 'Object', 'Int'])
    test_re(inputs_test_re, outputs_test_re, re_model, batch_size, world_size, rank, max_number_pos)

    # final test data
    _, _, test_re_annotations = \
        get_re_inputs(test_in_texts_re_final, test_out_re_final, tokenizer_re, input_length)
    outputs_test_re_final = TensorDataset(test_re_annotations)

    return re_model, outputs_test_re_final


def train_ner(data, epochs, batch_size, rank, save_every, world_size, input_length):
    # NER data
    id_label, label_id, len_labels = get_labels_id(data)
    train_in_ner, test_in_ner, train_out_ner, test_out_ner = split_train_test_ner(data)
    test_in_ner, test_in_ner_final, test_out_ner, test_out_ner_final = split_test_ner(test_in_ner, test_out_ner)

    # train data
    tokenized_texts_train_ner, tokenized_labels_train_ner = tokenize_text_ner(train_in_ner, train_out_ner,
                                                                              tokenizer_ner)
    ner_ids, ner_masks, ner_labels = \
        get_ner_inputs(tokenized_texts_train_ner, tokenized_labels_train_ner, tokenizer_ner, label_id, input_length)
    inputs_train_ner = TensorDataset(ner_ids, ner_masks)
    outputs_train_ner = TensorDataset(ner_labels)

    # test data
    tokenized_texts_test_ner, tokenized_labels_test_ner = tokenize_text_ner(test_in_ner, test_out_ner, tokenizer_ner)
    ner_ids, ner_masks, ner_labels = \
        get_ner_inputs(tokenized_texts_test_ner, tokenized_labels_test_ner, tokenizer_ner, label_id, input_length)
    inputs_test_ner = TensorDataset(ner_ids, ner_masks)
    outputs_test_ner = TensorDataset(ner_labels)

    # NER training
    bert_model = {'bert_model': bert_name_ner,
                  'len_labels': len_labels,
                  'id_label': id_label,
                  'label_id': label_id}
    ner_trainer = TrainerNer(bert_model, inputs_train_ner, outputs_train_ner,
                             epochs, batch_size, rank, save_every, world_size, input_length)
    max_epoch = ner_trainer.kfold_cross_validation(k=5)
    # retrain on the whole development set
    ner_model = ner_trainer.re_train(max_epoch)
    summary(ner_model,
            input_size=[(batch_size, input_length), (batch_size, input_length)],
            dtypes=['torch.IntTensor', 'torch.IntTensor'])
    test_ner(inputs_test_ner, outputs_test_ner, ner_model, batch_size, world_size, rank, id_label)

    # final test data
    tokenized_texts_test_ner, tokenized_labels_test_ner = tokenize_text_ner(test_in_ner_final, test_out_ner_final,
                                                                            tokenizer_ner)
    ner_ids, ner_masks, _ = \
        get_ner_inputs(tokenized_texts_test_ner, tokenized_labels_test_ner, tokenizer_ner, label_id, input_length)
    inputs_test_ner_final = TensorDataset(ner_ids, ner_masks)

    return ner_model, inputs_test_ner_final, id_label


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


def main(rank, world_size, save_every=10, epochs=10, batch_size=32, ner_input_length=128, re_input_length=128):
    global bert_name_re
    global bert_name_ner
    ddp_setup(rank, world_size)

    if not os.path.exists('../data'):
        os.makedirs('../data', exist_ok=True)

    if not os.path.exists("../data/ner.csv"):
        data_ner = load_data()
        pre_process_texts(data_ner)
        data_ner = prepare_data_for_ner(data_ner)
        compute_iob(data_ner)
        data_ner.to_csv("../data/ner.csv", index=False)
    else:
        data_ner = pd.read_csv("../data/ner.csv", dtype={'drug': object, 'effect': object})
    if not os.path.exists("../data/re.csv"):
        data_re = load_data()
        pre_process_texts(data_re)
        data_re = prepare_data_for_re(data_re)
        compute_pos(data_re)
        data_re.to_csv("../data/re.csv", index=False)
    else:
        data_re = pd.read_csv("../data/re.csv", converters={'annotated_text': literal_eval, 'pos_tags': literal_eval})

    ner_model, final_inputs, id_label = \
        train_ner(data_ner, epochs, batch_size, rank, save_every, world_size, ner_input_length)

    # re_model, final_outputs = \
    #     train_re(data_re, epochs, batch_size, rank, save_every, world_size, re_input_length)

    # final_model = FinalModel(ner_model, re_model, tokenizer, id_label, rank, re_input_length)
    # test_final(final_model, final_inputs, final_outputs, batch_size, world_size, rank)

    destroy_process_group()


if __name__ == '__main__':
    import sys

    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    save_every = int(sys.argv[3])
    ner_input_length = int(sys.argv[4])
    re_input_length = int(sys.argv[5])
    world_size = torch.cuda.device_count()
    mp.spawn(main,
             args=(world_size, save_every, epochs, batch_size, ner_input_length, re_input_length,),
             nprocs=world_size)
