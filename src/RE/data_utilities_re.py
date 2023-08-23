import copy
import torch

import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split


def mask_texts(texts, drugs, effects, concatenation=False):
    annotations = []
    masked_texts = []

    annotation = 1
    founded_drugs = set()
    founded_effects = set()
    if concatenation:
        drug_associations = {}
        effect_associations = {}
        for idx, (drug, effect) in enumerate(zip(drugs, effects)):
            drug = drug.split()
            effect = effect.split()
            for el in drug:
                founded_drugs.add(el)
                drug_associations.setdefault(el, idx + 1)
            for el in effect:
                founded_effects.add(el)
                drug = drugs[idx]
                drug = drug.split()
                drug = drug[0]
                effect_associations.setdefault(el, drug_associations[drug])

    founded_drugs = set()
    founded_effects = set()
    for text, drug, effect in zip(texts, drugs, effects):
        masking = []
        new_sent = []
        sent = text.split()
        drug = drug.split()
        effect = effect.split()
        for idx, w in enumerate(sent):
            if w in drug:
                if "DRUG" not in new_sent and w not in founded_drugs:
                    new_sent.append("DRUG")
                    if concatenation:
                        masking.append(drug_associations[w])
                    else:
                        masking.append(annotation)
            elif w in effect:
                if "EFFECT" not in new_sent and w not in founded_effects:
                    new_sent.append("EFFECT")
                    if concatenation:
                        masking.append(effect_associations[w])
                    else:
                        masking.append(annotation)
            else:
                new_sent.append(w)
                masking.append(0)

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
    data['text'] = data['text'].str.lower()
    texts = data['text'].values.tolist()
    data['drug'] = data['drug'].str.lower()
    drugs = data['drug'].values.tolist()
    data['effect'] = data['effect'].str.lower()
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


def tokenize_text_re(data, tokenizer):
    texts = data['masked_text'].to_list()
    texts_annotations = data['annotated_text'].to_list()

    temp_tokenized_texts = []
    temp_tokenized_annotations = []

    for text, annotations in zip(texts, texts_annotations):
        tokenized_text = []
        tokenized_annotation = []
        for word, annotation in zip(text.split(), annotations):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_text.append(tokenized_word)
            tokenized_annotation.append([annotation] * n_subwords)

        temp_tokenized_texts.append(tokenized_text)
        temp_tokenized_annotations.append(tokenized_annotation)

    tokenized_texts = []
    tokenized_annotations = []

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

    return tokenized_texts, tokenized_annotations


def split_train_test_re(tokenized_texts, output):
    train_in, test_in, train_out, test_out = train_test_split(tokenized_texts, output,
                                                              test_size=0.2, shuffle=True, random_state=0)

    return train_in, test_in, train_out, test_out


def split_test_re(texts_input, output):
    test_in_re, test_in_re_final, test_out_re, test_out_re_final = train_test_split(texts_input, output,
                                                                                    test_size=0.5, shuffle=True,
                                                                                    random_state=0)

    return test_in_re, test_in_re_final, test_out_re, test_out_re_final


def get_re_inputs(tokenized_texts, tokenized_annotations, tokenizer, max_len):
    bert_ids = []
    bert_annotations = []
    bert_masks = []

    for text, annotation in zip(tokenized_texts, tokenized_annotations):
        tokenized_text = ["[CLS]"] + text + ["[SEP]"]
        annotation = copy.copy(annotation)
        annotation.insert(0, -100)
        annotation.insert(len(tokenized_text) - 1, -100)

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
