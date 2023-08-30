import ast
import pandas as pd
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


def load_data():
    dataset = load_dataset("../ade_corpus_v2/ade_corpus_v2.py", 'Ade_corpus_v2_drug_ade_relation')
    dataframe = pd.DataFrame(dataset['train'])
    dataframe.drop(columns=['indexes'], inplace=True)
    dataframe.drop_duplicates(inplace=True, ignore_index=True)
    dataframe.dropna(inplace=True)

    return dataframe


def split_train_test(indices):
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=0)

    return train_indices, test_indices


def split_test(indices):
    val_indices, test_indices = train_test_split(indices, test_size=0.5, random_state=0)

    return val_indices, test_indices


# Dropping sentences with overlapping name in DRUG and EFFECT.
def drop_incorrect_sentences(data):
    # storing indexes (sentences) where overlapping occurs.
    find_double_index = list()

    for idx, item in data.iterrows():
        drug_list = list(item["drug"].split())
        effect_list = list(item["effect"].split())
        for d in drug_list:
            for e in effect_list:
                if d == e:
                    find_double_index.append(idx)

    # Dropping sentences based on the indexes.
    data.drop(index=find_double_index, inplace=True)
    data.reset_index(drop=True, inplace=True)


def pre_process_texts(data):
    drop_incorrect_sentences(data)

    drugs = data['drug'].unique().tolist()
    effects = data['effect'].unique().tolist()

    # Creation of a list with all drugs and effects, ensuring their integrity during preprocessing.
    # These terms will not be touched.
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

    # remove all '.', also at the end of sentence if the last word is a number
    data['text'] = data['text'].str.replace(r'\.\s*$', '', regex=True)
    data['drug'] = data['drug'].str.replace(r'\.\s*$', '', regex=True)
    data['effect'] = data['effect'].str.replace(r'\.\s*$', '', regex=True)

    # remove double space between words
    data['text'] = data['text'].str.replace(r'\s+', ' ', regex=True)
    data['drug'] = data['drug'].str.replace(r'\s+', ' ', regex=True)
    data['effect'] = data['effect'].str.replace(r'\s+', ' ', regex=True)

    # remove spaces in patterns like 'z = 2.27.' Works also for subsequent patterns
    data['text'] = data['text'].str.replace(r'(\b\w)\s*=\s*', r'\1=', regex=True)
    data['drug'] = data['drug'].str.replace(r'(\b\w)\s*=\s*', r'\1=', regex=True)
    data['effect'] = data['effect'].str.replace(r'(\b\w)\s*=\s*', r'\1=', regex=True)

    # lowercasing all drugs/effect in text
    for (i, _), (_, drug), (_, effect) in zip(data['text'].to_frame().iterrows(), data['drug'].to_frame().iterrows(),
                                              data['effect'].to_frame().iterrows()):
        drug = drug.drug
        effect = effect.effect
        data.at[i, 'text'] = data.at[i, 'text'].replace(drug, drug.lower())
        data.at[i, 'text'] = data.at[i, 'text'].replace(effect, effect.lower())

    # lowercasing all drugs/effects
    data['drug'] = data['drug'].str.lower()
    data['effect'] = data['effect'].str.lower()


def get_missed_class(classes):
    missed_class = []
    total_classes = list(range(5))

    for el in total_classes:
        if el not in classes:
            missed_class.append(el)

    return missed_class


# We try to give more importance to the classes related to the entities and less importance
# to the non-entity "O".
def compute_weights(data):
    class_weights = np.zeros(5)
    for labels in data:
        labels = np.delete(labels, np.where(labels == -100))
        classes = np.unique(labels)
        # There might be that not all the classes are present in all sentences.
        # We also give those missed classes a weighted value.
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