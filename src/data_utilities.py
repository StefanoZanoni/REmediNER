import re
import string

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from nltk.tokenize import TreebankWordTokenizer

import pandas as pd


def load_data():  # Noisy data? Duplicates?
    dataset = load_dataset("../ade_corpus_v2/ade_corpus_v2.py", 'Ade_corpus_v2_drug_ade_relation')
    dataframe = pd.DataFrame(dataset['train'])
    dataframe = dataframe.drop(columns='indexes')
    dataframe.drop_duplicates(inplace=True)  # Drop duplicates
    dataframe.dropna(inplace=True)
    return dataframe


def pre_process_texts(data):
    drugs = data['drug'].unique().tolist()
    effects = data['effect'].unique().tolist()
    exception_words = drugs + effects
    pattern = r"(?<![a-zA-Z])'s(?![a-zA-Z])|\b\d+(?:-\w+)*(?:'\w+)?\b|[^\w\s]".format("|".join(exception_words))
    data['text'] = data['text'].str.replace(pattern, ' ')


def iob_tagging(text, drug, effect, twt):
    start_d, end_d = re.search(re.escape(drug), text).span()
    span_list_d = twt.span_tokenize(text)
    start_e, end_e = re.search(re.escape(effect), text).span()
    span_list_e = twt.span_tokenize(text)

    iob_list = []
    for (start1, end1), (start2, end2) in zip(span_list_d, span_list_e):
        iob_tag = 'O'
        if start1 == start_d or start2 == start_e:
            iob_tag = 'B'
        elif (start_d < start1 or start_e < start2) and (end_d <= end1 or end_e <= end2):
            iob_tag = 'I'

        iob_list.append(iob_tag)

    return ' '.join(iob_list)


def get_row_iob(row, twt):
    return iob_tagging(row.text, row.drug, row.effect, twt)


def compute_iob(data):
    twt = TreebankWordTokenizer()
    data['iob'] = data.apply(lambda row: get_row_iob(row, twt), axis=1)


def split_train_test(data):
    train_data, test_data = train_test_split(data, test_size=0.1, shuffle=True, random_state=0)
    return train_data, test_data


def k_fold(data, n_splits=10):
    kf = KFold(n_splits=n_splits)
    return kf.split(data)


def get_input_output(data):
    input = data['text'].to_frame()
    output = data[['drug', 'effect']].apply(tuple, axis=1).to_frame()

    return input, output


def pre_processing(data, bert_model):
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    data = data['text'].to_list()
    encoded_input = tokenizer(data, padding=True, truncation=True, return_tensors='tf')
    return encoded_input
