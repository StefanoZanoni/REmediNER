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
    # dataframe = dataframe[:2] # for debugging
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
        "|".join(exception_words))  # PROBLEM: digits are tokenized and seperated 2.27 -> 2 27
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
    unique_drugs = data['drug'].unique()
    unique_effects = data['effect'].unique()
    drug_class = {label: i for i, label in enumerate(unique_drugs)}
    effect_class = {label: i for i, label in enumerate(unique_effects)}
    drug_classes = [drug_class[drug] for drug in data['drug']]
    effect_classes = [effect_class[effect] for effect in data['effect']]

    return [drug_classes, effect_classes], len(unique_drugs), len(unique_effects)


def get_bert_inputs(tokenized_texts, tokenized_labels, tokenizer, label_id, max_len=128):
    bert_ids = []
    bert_masks = []
    bert_labels = []

    for text, labels in zip(tokenized_texts, tokenized_labels):
        tokenized_text = ["[CLS]"] + text + ["[SEP]"]
        labels = copy.copy(labels)
        labels.insert(0, "O")
        labels.insert(len(tokenized_text) - 1, "O")
        # padding
        if len(tokenized_text) < max_len:
            tokenized_text = tokenized_text + ['[PAD]' for _ in range(max_len - len(tokenized_text))]
            labels = labels + ["O" for _ in range(max_len - len(labels))]

        attention_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_text]
        ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        label_ids = [label_id[label] for label in labels]

        bert_ids.append(ids)
        bert_masks.append(attention_mask)
        bert_labels.append(label_ids)

    return [bert_ids, bert_masks, bert_labels]


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
    return iob_tagging(row.text, row.drug, row.effect, twt)


def compute_iob(data):
    twt = TreebankWordTokenizer()
    data['iob'] = data.apply(lambda row: get_row_iob(row, twt), axis=1)


def compute_eager_tensor(data, text_len, num_texts, is_output=False):
    if not is_output:
        data = [item for sublist in data for item in sublist]
    data = tf.constant(data, dtype=tf.int32)
    data = tf.reshape(data, shape=(num_texts, text_len))

    return data


def split(input, output=None):
    if output is None:
        return train_test_split(input, test_size=0.1, shuffle=True, random_state=0)
    else:
        return train_test_split(input, output, test_size=0.1, shuffle=True, random_state=0)


def split_train_test(inputs, outputs):
    text_len = len(inputs[0][0])

    train_ids, test_ids, train_out_d, test_out_d = split(inputs[0], outputs[0])
    num_texts = len(train_ids)
    train_ids = compute_eager_tensor(train_ids, text_len, num_texts)
    train_out_d = compute_eager_tensor(train_out_d, 1, num_texts, is_output=True)
    num_texts = len(test_ids)
    test_ids = compute_eager_tensor(test_ids, text_len, num_texts)
    test_out_d = compute_eager_tensor(test_out_d, 1, num_texts, is_output=True)

    _, _, train_out_e, test_out_e = split(inputs[0], outputs[1])
    num_texts = len(train_ids)
    train_out_e = compute_eager_tensor(train_out_e, 1, num_texts, is_output=True)
    num_texts = len(test_ids)
    test_out_e = compute_eager_tensor(test_out_e, 1, num_texts, is_output=True)

    train_masks, test_masks = split(inputs[0])
    num_texts = len(train_masks)
    train_masks = compute_eager_tensor(train_masks, text_len, num_texts)
    num_texts = len(test_masks)
    test_masks = compute_eager_tensor(test_masks, text_len, num_texts)

    train_labels, test_labels = split(inputs[0])
    num_texts = len(train_labels)
    train_labels = compute_eager_tensor(train_labels, text_len, num_texts)
    num_texts = len(test_labels)
    test_labels = compute_eager_tensor(test_labels, text_len, num_texts)

    train_in = [train_ids, train_masks, train_labels]
    test_in = [test_ids, test_masks, test_labels]
    train_out = [train_out_d, train_out_e]
    test_out = [test_out_d, test_out_e]

    return train_in, test_in, train_out, test_out


def k_fold(data_x, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    return kf.split(data_x)
