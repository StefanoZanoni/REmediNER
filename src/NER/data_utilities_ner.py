import copy
import re

import torch
from nltk import TreebankWordTokenizer
from sklearn.model_selection import train_test_split


def iob_tagging(text, drug, effect, twt):
    start_d, end_d = re.search(re.escape(drug), text).span()
    start_e, end_e = re.search(re.escape(effect), text).span()
    span_list = twt.span_tokenize(text)

    entities = ['Drug', 'Effect']

    iob_list = []
    i = 0
    for start, end in span_list:
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
