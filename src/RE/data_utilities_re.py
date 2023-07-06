import copy
import re
import torch

import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split


def mask_texts(texts, drugs, effects, concatenation=False):
    annotations = []
    masked_texts = []

    annotation = 1

    for text, drug, effect in zip(texts, drugs, effects):
        masking = []
        new_sent = []
        sent = text.split()
        drug = drug.split()
        effect = effect.split()
        for idx, w in enumerate(sent):
            if w in drug:
                if "DRUG" not in new_sent:
                    new_sent.append("DRUG")
                    masking.append(annotation)
            elif w in effect:
                if "EFFECT" not in new_sent:
                    new_sent.append("EFFECT")
                    masking.append(annotation)
            else:
                new_sent.append(w)
                masking.append(0)

        if concatenation:
            annotation += 1

        annotations.append(masking)  # Sentences in str with annotation DRUG-EFFECT
        masked_texts.append(" ".join(new_sent))  # Masked sentences

    return annotations, masked_texts


def concatenate_texts(texts, concat_number):
    result = ''
    for i in range(concat_number):
        result = result + ' ' + texts[i]

    return result


def concatenate_annotations(annotations, concat_number):
    result = []
    for i in range(concat_number):
        result = result + annotations[i]

    return result


def add_concatenation(data, data_re, initial_data_size, concat_number):
    # concatenation of concat_number texts and relative labeling
    concatenation_size = int(np.ceil(initial_data_size * 0.33))
    for i in range(concatenation_size):
        random_row_indexes = [np.random.randint(low=0, high=len(data)) for _ in range(concat_number)]
        rows = data.iloc[random_row_indexes]
        texts = rows['text'].values.tolist()
        drugs = rows['drug'].values.tolist()
        effects = rows['effect'].values.tolist()
        annotation, masked_texts = mask_texts(texts, drugs, effects, concatenation=True)
        masked_texts = concatenate_texts(masked_texts, concat_number)
        annotation = concatenate_annotations(annotation, concat_number)
        data_re.loc[len(data_re)] = [masked_texts, annotation]


def prepare_data_for_re(data):
    # lowercasing vs no-lowercasing?
    texts = data['text'].values.tolist()
    drugs = data['drug'].values.tolist()
    effects = data['effect'].values.tolist()
    annotation, masked_texts = mask_texts(texts, drugs, effects, concatenation=False)

    data_re = pd.DataFrame()
    data_re['masked_text'] = masked_texts
    data_re['annotated_text'] = annotation

    initial_size = len(data)
    add_concatenation(data, data_re, initial_size, 2)
    add_concatenation(data, data_re, initial_size, 3)
    add_concatenation(data, data_re, initial_size, 4)

    return data_re


def compute_context_mean_length(data):
    function_word = ['AUX', 'CONJ', 'CCONJ', 'INTJ', 'PUNCT', 'SCONJ', 'X', 'SPACE']
    context_length = 0

    for index, row in data.iterrows():
        context_pos = row['pos_tags']
        context_pos = [pos for pos in context_pos if pos not in function_word]
        context_length += len(context_pos)

    return int(np.ceil(context_length / len(data)))


def compute_pos(data):
    nlp = spacy.load("en_core_web_sm")
    pos_tags = []
    for text in data['masked_text']:
        doc = nlp(text)
        pos = [token.pos_ for token in doc]
        pos_tags.append(pos)

    data['pos_tags'] = pos_tags


def tokenize_text_re(data, tokenizer):
    texts = data['masked_text'].to_list()
    texts_annotations = data['annotated_text'].to_list()
    pos_tags = data['pos_tags'].to_list()

    temp_tokenized_texts = []
    temp_tokenized_annotations = []
    temp_tokenized_tags = []

    for text, annotations, pos in zip(texts, texts_annotations, pos_tags):
        tokenized_text = []
        tokenized_annotation = []
        tokenized_pos = []
        for word, annotation, pos_tag in zip(text.split(), annotations, pos):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_text.append(tokenized_word)
            tokenized_annotation.append([annotation] * n_subwords)
            tokenized_pos.append([pos_tag] * n_subwords)

        temp_tokenized_texts.append(tokenized_text)
        temp_tokenized_annotations.append(tokenized_annotation)
        temp_tokenized_tags.append(tokenized_pos)

    tokenized_texts = []
    tokenized_annotations = []
    tokenized_pos = []

    for text in temp_tokenized_texts:
        tokenized_text = []
        for word in text:
            for token in word:
                tokenized_text.append(token)
        tokenized_texts.append(tokenized_text)

    for annotations in temp_tokenized_annotations:
        tokenized_annotation = []
        for annotation in annotations:
            for token in annotation:
                tokenized_annotation.append(token)
        tokenized_annotations.append(tokenized_annotation)

    for pos_tags in temp_tokenized_tags:
        tokenized_tag = []
        for pos_tag in pos_tags:
            for token in pos_tag:
                tokenized_tag.append(token)
        tokenized_pos.append(tokenized_tag)

    return tokenized_texts, tokenized_annotations, tokenized_pos


def split_train_test_re(tokenized_texts, tokenized_pos, output):
    input = []
    for i in range(len(tokenized_texts)):
        input.append((tokenized_texts[i], tokenized_pos[i]))
    train_in, test_in, train_out, test_out = train_test_split(input, output,
                                                              test_size=0.1, random_state=0)

    train_in_texts = []
    train_in_pos = []
    for el in train_in:
        train_in_texts.append(el[0])
        train_in_pos.append(el[1])

    test_in_texts = []
    test_in_pos = []
    for el in test_in:
        test_in_texts.append(el[0])
        test_in_pos.append(el[1])



    return train_in_texts, train_in_pos, test_in_texts, test_in_pos, train_out, test_out


def get_re_inputs(tokenized_texts, tokenized_annotations, tokenizer, max_len=512):
    bert_ids = []
    bert_annotations = []
    bert_masks = []

    for text, annotation in zip(tokenized_texts, tokenized_annotations):
        tokenized_text = ["[CLS]"] + text + ["[SEP]"]
        annotation = copy.copy(annotation)
        annotation.insert(0, 0)
        annotation.insert(len(tokenized_text) - 1, 0)

        # truncation
        if len(tokenized_text) > max_len:
            tokenized_text = tokenized_text[:max_len]
            annotation = annotation[:max_len]
        # padding
        if len(tokenized_text) < max_len:
            tokenized_text = tokenized_text + ['[PAD]' for _ in range(max_len - len(tokenized_text))]
            annotation = annotation + ['PAD' for _ in range(max_len - len(annotation))]

        attention_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_text]
        ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        annotation = [el if el != 'PAD' else -100 for el in annotation]

        bert_ids.append(ids)
        bert_masks.append(attention_mask)
        bert_annotations.append(annotation)

    re_ids = torch.tensor(bert_ids, dtype=torch.long)
    re_masks = torch.tensor(bert_masks, dtype=torch.long)
    re_annotations = torch.tensor(bert_annotations, dtype=torch.long)

    return re_ids, re_masks, re_annotations


def compute_pos_indexes(tokenized_pos):

    # compute pos indexes
    max_number_pos = set()
    for l in tokenized_pos:
        for pos in l:
            max_number_pos.add(pos)

    pos_indexes = {pos: i for i, pos in enumerate(max_number_pos, start=1)}

    indexes_global = []
    for l in tokenized_pos:
        indexes_local = [0]
        # CLS
        for pos in l:
            indexes_local.append(pos_indexes[pos])
        # SEP
        indexes_local.append(0)
        # PAD
        for i in range(512 - len(indexes_local)):
            indexes_local.append(0)

        indexes_global.append(indexes_local)

    return indexes_global, len(max_number_pos)
