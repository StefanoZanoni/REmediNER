import copy
import re
import ast
import pandas as pd
import torch
import spacy

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from nltk.tokenize import TreebankWordTokenizer


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

    data['text'] = data['text'].str.strip()
    data['drug'] = data['drug'].str.strip()
    data['effect'] = data['effect'].str.strip()

    # removes all punctuations, retains decimal numbers and patterns like 'z = 2.27'
    pattern = r"('s\b)|(?!(?:\b\w+\b|\d+(?:\.\d+)?|[a-zA-Z]=\d+(?:\.\d+)?))([^\w\s\'.=]|(?<!\d)\.(?!\d))".format(
        "|".join(exception_words))
    data['text'] = data['text'].str.replace(pattern, ' ', regex=True)
    data['drug'] = data['drug'].str.replace(pattern, ' ', regex=True)
    data['effect'] = data['effect'].str.replace(pattern, ' ', regex=True)

    # remove the remaining quotes
    data['text'] = data['text'].str.replace("'", '', regex=True)
    data['drug'] = data['drug'].str.replace("'", '', regex=True)
    data['effect'] = data['effect'].str.replace("'", '', regex=True)

    # remove all '.', also at the end of phrase if the last word is a number
    data['text'] = data['text'].str.replace(r'\.\s*$', '', regex=True)
    data['drug'] = data['drug'].str.replace(r'\.\s*$', '', regex=True)
    data['effect'] = data['effect'].str.replace(r'\.\s*$', '', regex=True)

    # remove double space
    data['text'] = data['text'].str.replace(r'\s+', ' ', regex=True)
    data['drug'] = data['drug'].str.replace(r'\s+', ' ', regex=True)
    data['effect'] = data['effect'].str.replace(r'\s+', ' ', regex=True)

    # remove with spaces in patterns like 'z = 2.27.' Works also for subsequent patterns
    data['text'] = data['text'].str.replace(r'(\b\w)\s*=\s*', r'\1=', regex=True)
    data['drug'] = data['drug'].str.replace(r'(\b\w)\s*=\s*', r'\1=', regex=True)
    data['effect'] = data['effect'].str.replace(r'(\b\w)\s*=\s*', r'\1=', regex=True)

    data['num_tokens_text'] = data['text'].apply(lambda x: len(str(x).split()))


def compute_pos(data):
    nlp = spacy.load("en_core_web_sm")
    pos_tags = []
    for text in data['text']:
        doc = nlp(text)
        pos = [token.pos_ for token in doc]
        pos_tags.append(pos)

    data['pos_tags'] = pos_tags


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind, ind + sll - 1


def compute_context_mean_length(data):
    function_word = ['AUX', 'CONJ', 'CCONJ', 'INTJ', 'PUNCT', 'SCONJ', 'X', 'SPACE']
    context_length = 0

    for index, row in data.iterrows():
        text = row['text'].split()
        drug = row['drug'].split()
        drug_indexes = find_sub_list(drug, text)
        effect = row['effect'].split()
        effect_indexes = find_sub_list(effect, text)
        if drug_indexes[0] < effect_indexes[0]:
            context_pos = row['pos_tags'][drug_indexes[1] + 1:effect_indexes[0]]
        else:
            context_pos = row['pos_tags'][effect_indexes[1] + 1:drug_indexes[0]]
        context_pos = [pos for pos in context_pos if pos not in function_word]
        context_length += len(context_pos)

    return (context_length / len(data)).__ceil__()


def count_drug_effects(data):
    unique_drugs = data['drug'].unique()
    unique_effects = data['effect'].unique()

    return len(unique_drugs), len(unique_effects)


def get_labels_id(data):
    labels = data['iob'].unique()
    entities = set()
    for l in labels:
        for el in l.split():
            entities.add(el)
    id_label = {i: label for i, label in enumerate(entities)}
    label_id = {label: i for i, label in enumerate(entities)}

    return id_label, label_id, len(entities)


def tokenize_text(texts, labels, tokenizer):
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


def get_re_outputs(data):
    unique_drugs = data['drug'].unique()
    unique_effects = data['effect'].unique()
    drug_class = {label: i for i, label in enumerate(unique_drugs)}
    effect_class = {label: i for i, label in enumerate(unique_effects)}
    drug_classes = [drug_class[drug] for drug in data['drug']]
    effect_classes = [effect_class[effect] for effect in data['effect']]

    return [drug_classes, effect_classes]


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

    bert_ids = torch.tensor(bert_ids, dtype=torch.long)
    bert_masks = torch.tensor(bert_masks, dtype=torch.long)
    bert_labels = torch.tensor(bert_labels, dtype=torch.long)

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


def split_train_test(data):
    input = data['text'].to_frame()
    output = data[['iob', 'drug', 'effect']]
    train_in, test_in, train_out, test_out = train_test_split(input, output, test_size=0.1, random_state=0)
    train_out_ner = train_out['iob'].to_frame()
    train_out_re = train_out[['drug', 'effect']]
    test_out_ner = test_out['iob'].to_frame()
    test_out_re = test_out[['drug', 'effect']]

    return train_in, test_in, train_out_ner, train_out_re, test_out_ner, test_out_re
