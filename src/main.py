import os
import transformers

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import TensorDataset
from transformers import BertTokenizerFast

from src.data_utilities import load_data, split_train_test, tokenize_text, \
    compute_iob, pre_process_texts, get_labels_id, get_bert_inputs, get_re_outputs, count_drug_effects
from src.training import TrainerNer

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


def main(rank, world_size, save_every, total_epochs=1, batch_size=32):

    ddp_setup(rank, world_size)

    data = load_data()
    pre_process_texts(data)
    compute_iob(data)
    id_label, label_id, len_labels = get_labels_id(data)
    num_drugs, num_effects = count_drug_effects(data)

    train_in, test_in, train_out_ner, train_out_re, test_out_ner, test_out_re = split_train_test(data)

    tokenized_texts_train, tokenized_labels_train = tokenize_text(train_in, train_out_ner, tokenizer)
    tokenized_texts_test, tokenized_labels_test = tokenize_text(test_in, test_out_ner, tokenizer)
    inputs_train = get_bert_inputs(tokenized_texts_train, tokenized_labels_train, tokenizer, label_id)
    inputs_train = TensorDataset(inputs_train[0], inputs_train[1], inputs_train[2])
    # prepare the data to multi-gpu training
    # inputs_train = \
    #     DataLoader(inputs_train, batch_size=batch_size, pin_memory=True,
    #                shuffle=False, sampler=DistributedSampler(inputs_train))

    inputs_test = get_bert_inputs(tokenized_texts_test, tokenized_labels_test, tokenizer, label_id)
    train_out_re = get_re_outputs(train_out_re)
    test_out_re = get_re_outputs(test_out_re)

    global bert_model
    bert_model = {'bert_model': bert_model,
                  'len_labels': len_labels,
                  'id_label': id_label,
                  'label_id': label_id}
    ner_trainer = TrainerNer(bert_model, inputs_train, total_epochs, batch_size, rank, save_every, world_size)
    ner_trainer.kfold_cross_validation(k=2)

    destroy_process_group()


if __name__ == '__main__':
    import sys

    save_every = int(sys.argv[1])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, save_every,), nprocs=world_size)
