import copy
import re
import ast
import pandas as pd
import tensorflow as tf

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from nltk.tokenize import TreebankWordTokenizer

i = 0


def load_data():  # Noisy data? Duplicates?
    dataset = load_dataset("../ade_corpus_v2/ade_corpus_v2.py", 'Ade_corpus_v2_drug_ade_relation')
    dataframe = pd.DataFrame(dataset['train'])
    dataframe = dataframe[:2]
    dataframe['indexes'] = dataframe['indexes'].astype(str)
    dataframe.drop_duplicates(inplace=True, ignore_index=True)  # Drop duplicates
    dataframe.dropna(inplace=True)
    dataframe['indexes'] = dataframe['indexes'].apply(lambda x: ast.literal_eval(x))
    return dataframe


def pre_process_texts(data):
    drugs = data['drug'].unique().tolist()
    effects = data['effect'].unique().tolist()
    exception_words = drugs + effects
    # remove all punctuations except genitive s
    pattern = r'(?!\b\w+\b)[^\w\s\']'.format(
        "|".join(exception_words))  # PROBLEM: digits are tokenized and seperated 2.27 -> 2 . 2 7
    data['text'] = data['text'].str.replace(pattern, ' ', regex=True)
    data['drug'], data['effect'] = data['drug'].str.replace(pattern, ' ', regex=True), \
        data['effect'].str.replace(pattern, ' ', regex=True)
    data['num_tokens_text'] = data['text'].apply(lambda x: len(str(x).split()))


def get_labels_id(data):
    labels = data['iob'].unique()
    entities = set()
    for l in labels:
        for el in l.split():
            entities.add(el)
    id_label = {i: label for i, label in enumerate(entities)}
    label_id = {label: i for i, label in enumerate(entities)}

    return id_label, label_id, len(entities)


def tokenize_text(data, tokenizer):
    texts = data['text'].to_list()
    labels = data['iob'].to_list()

    temp_tokenized_texts = []
    temp_tokenized_labels = []

    for text, text_labels in zip(texts, labels):
        tokenized_text = []
        tokenized_label = []
        for word, labels in zip(text.split(), text_labels.split()):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_text.append(tokenized_word)
            tokenized_label.append([labels] * n_subwords)

        # tokenized_text = list(map(' '.join, tokenized_text))
        # tokenized_label = list(map(' '.join, tokenized_label))
        temp_tokenized_texts.append(tokenized_text)
        temp_tokenized_labels.append(tokenized_label)

    tokenized_texts = []
    tokenized_labels = []

    for text in temp_tokenized_texts:
        tokenized_text = []
        for word in text:
            for token in word:
                tokenized_text.append(token)
        tokenized_texts.append(tokenized_text)

    for text_labels in temp_tokenized_labels:
        tokenized_text_labels = []
        for labels in text_labels:
            for label in labels:
                tokenized_text_labels.append(label)
        tokenized_labels.append(tokenized_text_labels)

    return tokenized_texts, tokenized_labels


def get_bert_outputs(data):
    return list(zip(data.drug, data.effect))


def get_bert_inputs(tokenized_texts, tokenized_labels, max_len, tokenizer, label_id):
    bert_inputs = []

    for text, labels in zip(tokenized_texts, tokenized_labels):
        tokenized_text = ["[CLS]"] + text + ["[SEP]"]
        labels = copy.copy(labels)
        labels.insert(0, "O")
        labels.insert(len(tokenized_text)-1, "O")
        # padding
        if len(tokenized_text) < max_len:
            tokenized_text = tokenized_text + ['[PAD]' for _ in range(max_len - len(tokenized_text))]
            labels = labels + ["O" for _ in range(max_len - len(labels))]

        attention_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_text]
        ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        label_ids = [label_id[label] for label in labels]

        input = (tf.constant(ids, dtype=tf.int32),
                 tf.constant(attention_mask, dtype=tf.int32),
                 tf.constant(label_ids, dtype=tf.int32))

        bert_inputs.append(input)

    return bert_inputs


def iob_tagging(text, drug, effect, twt):
    start_d, end_d = re.search(re.escape(drug), text).span()
    span_list_d = twt.span_tokenize(text)
    start_e, end_e = re.search(re.escape(effect), text).span()
    span_list_e = twt.span_tokenize(text)

    entities = ['Drug', 'Effect']

    iob_list = []
    i = 0
    for (start1, end1), (start2, end2) in zip(span_list_d, span_list_e):
        iob_tag = 'O'
        if start1 == start_d or start2 == start_e:
            iob_tag = 'B'
            if start1 == start_d:
                i = 0
            else:
                i = 1
        elif (start_d < start1 and end1 <= end_d) or (start_e < start2 and end2 <= end_e):
            iob_tag = 'I'

        if iob_tag != 'O':
            iob_tag += '-{}'.format(entities[i])

        iob_list.append(iob_tag)

    return ' '.join(iob_list)


def get_row_iob(row, twt):
    # global i
    # print(i)
    # i += 1
    return iob_tagging(row.text, row.drug, row.effect, twt)


def compute_iob(data):
    twt = TreebankWordTokenizer()
    data['iob'] = data.apply(lambda row: get_row_iob(row, twt), axis=1)


def split_train_test(inputs, outputs):
    return train_test_split(inputs, outputs, test_size=0.1, shuffle=True, random_state=0)


def k_fold(data_x, data_y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    return kf.split(data_x, data_y)
