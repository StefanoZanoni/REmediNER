import copy
import re

import numpy as np
import torch
from nltk import TreebankWordTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

np.random.seed(0)


def iob_tagging(text, drugs, effects, twt):
    # lists of all drugs and effects starting and ending indexes
    ds = []
    es = []

    # Find all starting and ending indexes of drugs and effects
    # if the same drug of effect occurs two or more times in the text,
    # start and end indexes refer to the first occurrence.
    # In this way we can associate one drug with multiple effects and vice versa
    for drug in drugs:
        start, end = re.search(re.escape(drug), text).span()
        ds.append((start, end))
    for effect in effects:
        start, end = re.search(re.escape(effect), text).span()
        es.append((start, end))

    # find all starting and ending indexes of all words in text
    span_list = twt.span_tokenize(text)

    entities = ['Drug', 'Effect']
    iob_list = []
    i = 0

    for start, end in span_list:
        temp_iob_list = []
        for (start_d, end_d), (start_e, end_e) in zip(ds, es):
            iob_tag = 'O'
            # if the beginning of a drug or an effect is found, set iob_tag to B
            if start == start_d or start == start_e:
                iob_tag = 'B'
                # check if it is the beginning of a drug or of an effect
                if start == start_d:
                    i = 0
                else:
                    i = 1
            # if the drug or the effect is split into multiple tokens, set iob_tag to I
            elif (start_d < start and end <= end_d) or (start_e < start and end <= end_e):
                iob_tag = 'I'

            # add '-Drug' or '-Effect' to iob_tag found so far
            if iob_tag != 'O':
                iob_tag += '-{}'.format(entities[i])

            temp_iob_list.append(iob_tag)

        # find multiple drugs or effects in the text prioritizing 'B' and 'I' over the 'O'
        for j in range(len(temp_iob_list)):
            if 'B' in temp_iob_list[j]:
                iob_tag = temp_iob_list[j]
            elif 'I' in temp_iob_list[j]:
                iob_tag = temp_iob_list[j]

        iob_list.append(iob_tag)

    return ' '.join(iob_list)


# compute iob tagging for a single text
def get_row_iob(row, twt):
    return iob_tagging(row.text, row.drug, row.effect, twt)


# compute iob tagging on the whole dataset
def compute_iob(data):
    twt = TreebankWordTokenizer()
    data['iob'] = data.apply(lambda row: get_row_iob(row, twt), axis=1)


def get_labels_id():
    label_id = {'O': 0, 'B-Drug': 1, 'I-Drug': 2, 'B-Effect': 3, 'I-Effect': 4}
    id_label = {0: 'O', 1: 'B-Drug', 2: 'I-Drug', 3: 'B-Effect', 4: 'I-Effect'}

    return id_label, label_id, 5


def split_train_test_ner(data):
    input = data['text'].to_frame()
    output = data['iob'].to_frame()
    train_in, test_in, train_out, test_out = \
        train_test_split(input, output, test_size=0.2, shuffle=True, random_state=0)

    return train_in, test_in, train_out, test_out


# this function is used to split the test data into test data for NER model testing
# and test data for FINAL model testing
def split_test_ner(input, output):
    test_in_ner, test_in_ner_final, test_out_ner, test_out_ner_final = \
        train_test_split(input, output, test_size=0.5, shuffle=True, random_state=0)

    return test_in_ner, test_in_ner_final, test_out_ner, test_out_ner_final


# this function is used to tokenize the text according to the bert tokenizer
# and duplicate the iob tagging for the sub-tokens
def tokenize_text_ner(texts, labels, tokenizer):
    texts = texts['text'].to_list()
    labels = labels['iob'].to_list()

    temp_tokenized_texts = []
    temp_tokenized_labels = []

    # tokenize text and labels
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

    # transform the list of lists of tokens into a list of tokens
    for text in temp_tokenized_texts:
        tokenized_text = []
        for word in text:
            for token in word:
                tokenized_text.append(token)
        tokenized_texts.append(tokenized_text)

    # transform the list of lists of tokens into a list of tokens
    for text_labels in temp_tokenized_labels:
        tokenized_text_labels = []
        for labels in text_labels:
            for label in labels:
                tokenized_text_labels.append(label)
        tokenized_labels.append(tokenized_text_labels)

    return tokenized_texts, tokenized_labels


# this function is used to prepare input in BERT format
def get_ner_inputs(tokenized_texts, tokenized_labels, tokenizer, label_id, max_len):
    bert_ids = []
    bert_masks = []
    bert_labels = []

    for text, labels in zip(tokenized_texts, tokenized_labels):
        tokenized_text = ["[CLS]"] + text + ["[SEP]"]
        labels = copy.copy(labels)
        labels.insert(0, 'PAD')
        labels.insert(len(tokenized_text) - 1, 'PAD')

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


# This function is used to compute data augmentation for NER task.
# Multiple texts will be concatenated to obtain texts with multiple drugs and effects.
def prepare_data_for_ner(data):
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


def get_missed_class(classes):
    missed_class = []
    total_classes = list(range(5))

    for el in total_classes:
        if el not in classes:
            missed_class.append(el)

    return missed_class


def compute_weights(data):
    class_weights = np.zeros(5)
    for labels in data:
        labels = np.delete(labels, np.where(labels == -100))
        classes = np.unique(labels)
        missed_class = get_missed_class(classes)
        weights = class_weight.compute_class_weight('balanced',
                                                    classes=classes,
                                                    y=labels)

        if missed_class:
            for missed in missed_class:
                if missed < len(weights):
                    weights = np.insert(weights, missed, np.max(weights) + np.mean(weights))
                else:
                    weights = np.append(weights, np.max(weights) + np.mean(weights))

        class_weights += weights

    return class_weights / len(data)
