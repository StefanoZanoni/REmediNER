import ast
import pandas as pd

from datasets import load_dataset


def load_data():
    dataset = load_dataset("../ade_corpus_v2/ade_corpus_v2.py", 'Ade_corpus_v2_drug_ade_relation')
    dataframe = pd.DataFrame(dataset['train'])
    # dataframe = dataframe[:13]  # for debugging
    dataframe.drop(columns=['indexes'], inplace=True)
    dataframe.drop_duplicates(inplace=True, ignore_index=True)  # Drop duplicates
    dataframe.dropna(inplace=True)

    return dataframe


def drop_incorrect_sentences(data):
    find_double_index = list()

    for idx, item in data.iterrows():
        drug_list = list(item["drug"].split())
        effect_list = list(item["effect"].split())
        for d in drug_list:
            for e in effect_list:
                if d == e:
                    find_double_index.append(idx)

    data.drop(index=find_double_index, inplace=True)
    data.reset_index(drop=True, inplace=True)


def pre_process_texts(data):
    drop_incorrect_sentences(data)

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
