import copy
import torch

import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split


def mask_texts(texts, drugs, effects, concatenation=False):
    annotations = []
    masked_texts = []

    # Used to label a correlation between drug(s) and effect(s).
    # for example a correlated drug and effect will have the same value (1,2,3 etc.).
    annotation = 1

    found_drugs = set()
    found_effects = set()

    # If a text is concatenated, there could be the presence of a drug with multiple effects and vice-versa.
    # We have to ensure to associate them with the same index.
    # We therefore create two dictionaries that associate the drugs and effects with their index.
    if concatenation:
        drug_associations = {}
        effect_associations = {}
        for idx, (drug, effect) in enumerate(zip(drugs, effects)):
            drug = drug.split()
            effect = effect.split()
            for el in drug:
                found_drugs.add(el)
                # The index (idx) must be != 0. Zero is associated with words that are neither drugs nor effects.
                drug_associations.setdefault(el, idx + 1)
            for el in effect:
                found_effects.add(el)
                drug = drugs[idx]
                drug = drug.split()
                drug = drug[0]
                # We store the effect with the same index of the associated drug.
                effect_associations.setdefault(el, drug_associations[drug])

    found_drugs = set()
    found_effects = set()
    for text, drug, effect in zip(texts, drugs, effects):
        associations = []
        new_sent = []
        sent = text.split()
        drug = drug.split()
        effect = effect.split()
        for w in sent:
            if w in drug:
                # The idea is to maintain only one mask even though the drug is present multiple times.
                if w in found_drugs:
                    new_sent.append(w)
                    associations.append(0)
                # If the mask DRUG is already present in the masked sentence, it will be ignored.
                # Drugs with multiple words will be represented with just one mask (in this case DRUG).
                elif "DRUG" not in new_sent:
                    new_sent.append("DRUG")
                    if concatenation:
                        associations.append(drug_associations[w])
                    else:
                        associations.append(annotation)
                found_drugs.add(w)
            elif w in effect:
                if w in found_effects:
                    new_sent.append(w)
                    associations.append(0)
                elif "EFFECT" not in new_sent:
                    new_sent.append("EFFECT")
                    if concatenation:
                        associations.append(effect_associations[w])
                    else:
                        associations.append(annotation)
                found_effects.add(w)
            else:
                new_sent.append(w)
                associations.append(0)

        annotations.append(associations)  # Sentences in str with annotation DRUG-EFFECT
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
    # we mask texts individually, without concatenating.
    annotation, masked_texts = mask_texts(texts, drugs, effects, concatenation=False)

    data_re = pd.DataFrame()
    data_re['masked_text'] = masked_texts
    data_re['annotated_text'] = annotation

    initial_size = len(data)
    np.random.seed(0)
    # we concatenate the masked text
    add_concatenation(data, data_re, initial_size, 2)
    add_concatenation(data, data_re, initial_size, 3)
    add_concatenation(data, data_re, initial_size, 4)

    data_re['masked_text'] = data_re['masked_text'].apply(remove_double_spaces)
    return data_re


def remove_double_spaces(text):
    return ' '.join(text.split())


# this function is used to tokenize the text according to the bert tokenizer
# and duplicate the annotations for the sub-tokens
def tokenize_text_re(data, tokenizer):
    texts = data['masked_text'].to_list()
    texts_annotations = data['annotated_text'].to_list()

    temp_tokenized_texts = []
    temp_tokenized_annotations = []

    # tokenize text and annotations
    for text, annotations in zip(texts, texts_annotations):
        tokenized_text = []
        tokenized_annotation = []
        # We tokenize every single word in the sentence in order to get the number of subwords.
        # The aim is to extend the annotations to the number of subwords.
        for word, annotation in zip(text.split(), annotations):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_text.append(tokenized_word)
            tokenized_annotation.append([annotation] * n_subwords)

        temp_tokenized_texts.append(tokenized_text)
        temp_tokenized_annotations.append(tokenized_annotation)

    tokenized_texts = []
    tokenized_annotations = []

    # transform the list of lists of tokens into a list of tokens
    for text in temp_tokenized_texts:
        tokenized_text = []
        for word in text:
            for token in word:
                tokenized_text.append(token)
        tokenized_texts.append(tokenized_text)

    # transform the list of lists of tokens into a list of tokens
    for annotations in temp_tokenized_annotations:
        tokenized_annotation = []
        for annotation in annotations:
            for token in annotation:
                tokenized_annotation.append(token)
        tokenized_annotations.append(tokenized_annotation)

    return tokenized_texts, tokenized_annotations


# this function is used to prepare input in BERT format
def get_re_inputs(tokenized_texts, tokenized_annotations, tokenizer, max_len):
    bert_ids = []
    bert_annotations = []
    bert_masks = []

    for text, annotation in zip(tokenized_texts, tokenized_annotations):
        tokenized_text = ["[CLS]"] + text + ["[SEP]"]
        annotation = copy.copy(annotation)
        # We do not label CLS and SEP tokens.
        # We therefore add PAD in the annotations related to the positions of these two special tokens.
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
        # We assign the value -100 in the annotations that correspond to PAD
        # in order to ignore these indexes during the loss computation
        annotation = [el if el != 'PAD' else -100 for el in annotation]

        bert_ids.append(ids)
        bert_masks.append(attention_mask)
        bert_annotations.append(annotation)

    re_ids = torch.tensor(bert_ids, dtype=torch.long)
    re_masks = torch.tensor(bert_masks, dtype=torch.long)
    re_annotations = torch.tensor(bert_annotations, dtype=torch.long)

    return re_ids, re_masks, re_annotations
