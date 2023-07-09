import copy
import re

import numpy as np
import torch
from nltk import TreebankWordTokenizer
from sklearn.model_selection import train_test_split


def iob_tagging(text, drugs, effects, twt):
    ds = []
    es = []
    for drug in drugs:
        start, end = re.search(re.escape(drug), text).span()
        ds.append((start, end))
    for effect in effects:
        start, end = re.search(re.escape(effect), text).span()
        es.append((start, end))
    span_list = twt.span_tokenize(text)

    entities = ['Drug', 'Effect']

    iob_list = []

    i = 0
    for start, end in span_list:
        temp_iob_list = []
        for (start_d, end_d), (start_e, end_e) in zip(ds, es):
            iob_tag = 'O'
            if start == start_d or start == start_e:
                iob_tag = 'B'
                if start == start_d:
                    i = 0
                else:
                    i = 1
            elif (start_d < start and end <= end_d) or (start_e < start and end <= end_e):
                iob_tag = 'I'

            if iob_tag != 'O':
                iob_tag += '-{}'.format(entities[i])

            temp_iob_list.append(iob_tag)

        for j in range(len(temp_iob_list)):
            if 'B' in temp_iob_list[j]:
                iob_tag = temp_iob_list[j]
            elif 'I' in temp_iob_list[j]:
                iob_tag = temp_iob_list[j]

        iob_list.append(iob_tag)

    return ' '.join(iob_list)


def get_row_iob(row, twt):
    return iob_tagging(row.text, row.drug, row.effect, twt)


def compute_iob(data):
    twt = TreebankWordTokenizer()
    data['iob'] = data.apply(lambda row: get_row_iob(row, twt), axis=1)


def get_labels_id(data):
    labels = data['iob'].unique()
    entities = set()
    for l in labels:
        for el in l.split():
            entities.add(el)
    id_label = {i: label for i, label in enumerate(entities)}
    label_id = {label: i for i, label in enumerate(entities)}

    return id_label, label_id, len(entities)


def split_train_test_ner(data):
    input = data['text'].to_frame()
    output = data['iob'].to_frame()
    train_in, test_in, train_out, test_out = train_test_split(input, output, test_size=0.1, random_state=0)

    return train_in, test_in, train_out, test_out


def tokenize_text_ner(texts, labels, tokenizer):
    texts = texts['text'].to_list()
    labels = labels['iob'].to_list()

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


def get_ner_inputs(tokenized_texts, tokenized_labels, tokenizer, label_id, max_len=128):
    bert_ids = []
    bert_masks = []
    bert_labels = []

    for text, labels in zip(tokenized_texts, tokenized_labels):
        tokenized_text = ["[CLS]"] + text + ["[SEP]"]
        labels = copy.copy(labels)
        labels.insert(0, "O")
        labels.insert(len(tokenized_text) - 1, "O")

        # truncation
        if len(tokenized_text) > max_len:
            tokenized_text = tokenized_text[:max_len]
            labels = labels[:max_len]
        # padding
        if len(tokenized_text) < max_len:
            tokenized_text = tokenized_text + ['[PAD]' for _ in range(max_len - len(tokenized_text))]
            labels = labels + ['PAD' for _ in range(max_len - len(labels))]

        attention_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_text]
        ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        label_ids = [label_id[label] if label != 'PAD' else -100 for label in labels]

        bert_ids.append(ids)
        bert_masks.append(attention_mask)
        bert_labels.append(label_ids)

    ner_ids = torch.tensor(bert_ids, dtype=torch.long)
    ner_masks = torch.tensor(bert_masks, dtype=torch.long)
    ner_labels = torch.tensor(bert_labels, dtype=torch.long)

    return ner_ids, ner_masks, ner_labels


def concatenate_texts(texts, concat_number):
    result = ''
    for i in range(concat_number):
        result = result + ' ' + texts[i]

    return result


def concatenate_drugs_effects(drugs, effects, concat_number):
    concatenated_drugs = []
    concatenated_effects = []
    for i in range(concat_number):
        concatenated_drugs.append(drugs[i])
        concatenated_effects.append(effects[i])

    return concatenated_drugs, concatenated_effects


def convert_to_list(column, name):
    for i, row in column.iterrows():
        column.at[i, name] = [row[name]]


def prepare_data_for_ner(data):
    np.random.seed(0)
    new_data = copy.copy(data)
    convert_to_list(new_data['drug'].to_frame(), 'drug')
    convert_to_list(new_data['effect'].to_frame(), 'effect')
    concatenation_size = int(np.ceil(len(data) * 0.33))
    for concat_number in range(2, 5):
        for i in range(concatenation_size):
            random_row_indexes = [np.random.randint(low=0, high=len(data)) for _ in range(concat_number)]
            rows = data.iloc[random_row_indexes]
            texts = rows['text'].values.tolist()
            drugs = rows['drug'].values.tolist()
            effects = rows['effect'].values.tolist()
            concatenated_text = concatenate_texts(texts, concat_number)
            concatenated_drugs, concatenated_effects = concatenate_drugs_effects(drugs, effects, concat_number)
            new_data.loc[len(new_data)] = [concatenated_text, concatenated_drugs, concatenated_effects]

    return new_data
