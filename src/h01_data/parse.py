# Note: Dataset 0 is PAD 1 is SOW and 2 is EOW
import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.word2vec import Word2VecInfo
from h01_data.asjp import AsjpInfo
from util import argparser as parser
from util import util


def read_src_data(ffolder):
    return AsjpInfo.get_df(ffolder)


def filter_word2vec(df, args):
    word2vec = Word2VecInfo.get_word2vec(ffolder=args.ffolder)
    n_removed = (~df['concept_id_train'].isin(word2vec)).sum()
    print('Words filtered because they are not in word2vec: %d' % (n_removed))
    if n_removed > 0:
        print('\tWarning: Shouldn\'t remove any word! All should be in word2vec')

    df = df[df['concept_id_train'].isin(word2vec)]
    return df


def get_languages(df):
    languages = df.language_id.unique()
    np.random.shuffle(languages)
    return languages


def split_data(df, families_split, family_column):
    df_split = {
        i: df[df[family_column].isin(family)]
        for i, family in enumerate(families_split)
    }

    return df_split


def get_families_split(df, n_folds, family_column):
    families = df[family_column].unique()
    assert '' not in families, \
        'Empty family should not be allowed'

    np.random.shuffle(families)
    n_families = families.shape[0]

    assert n_families >= n_folds, \
        'Number of families is smaller than of folds'

    return np.split(families, n_folds)


def separate_train(df, n_folds, family_column='macroarea'):
    languages = get_languages(df)
    families_split = get_families_split(df, n_folds, family_column)

    folds = split_data(df, families_split, family_column)

    lang_splits = [df_fold.language_id.unique() for df_fold in folds.values()]
    print_families_count(df, families_split, family_column)

    return folds, lang_splits, families_split


# def get_families_count(languages, families):
#     families_unique = set(families.values())
#     family_counts = {x: sum([1 for y in languages if x == families[y]]) for x in families_unique}
#     print('Family counts:', family_counts)
#     return family_counts


def print_families_count(df, families_split, family_column):
    folds = {
        family[0]: df[df[family_column].isin(family)]
        for family in families_split
    }

    family_counts = {x: df_fold.language_id.unique().shape[0] for x, df_fold in folds.items()}
    print('Family counts:', family_counts)
    return family_counts


# def split_families(families, languages):
#     families_list = sorted(list(families.keys()))
#     np.random.shuffle(families_list)

#     num_families = len(families)
#     num_entries = sum(families.values())
#     train_entries = int(num_entries * .8)
#     val_entries = int(num_entries * .1)

#     train_size = max(
#         min(
#             len([x for x in range(1, num_families)
#                  if sum([families[y] for y in families_list[:x]]) < train_entries]),
#             num_families - 2),
#         1)
#     val_size = max(
#         min(
#             len([x for x in range(train_size + 1, num_families)
#                  if sum([families[y] for y in families_list[train_size:x]]) < val_entries]),
#             num_families - train_size - 1),
#         1)
#     test_size = num_families - train_size - val_size

#     train_set = families_list[:train_size]
#     val_set = families_list[train_size:-test_size]
#     test_set = families_list[-test_size:]
#     return (train_set, val_set, test_set)


def get_tokens(df):
    tokens = set()
    for index, x in df.iterrows():
        tokens |= set(x['wordform'])

    tokens = sorted(list(tokens))
    token_map = {x: i + 3 for i, x in enumerate(tokens)}
    token_map['PAD'] = 0
    token_map['SOW'] = 1
    token_map['EOW'] = 2

    return token_map


def get_params_map(df):
    params_map = pd.Series(df.concept_name.values, index=df.index).to_dict()
    return params_map


def process_data(df_split, token_map, args):
    util.mkdir('%s/preprocess-%s/' % (args.ffolder, args.fsuffix))
    for fold, df_fold in df_split.items():
        process_data_mode(df_fold, token_map, fold, args)


def process_data_mode(df, token_map, fold, args):
    data = parse_data(df, token_map)
    save_data(data, fold, args.ffolder, fsuffix=args.fsuffix)


def parse_data(df, token_map):
    data = []

    for i, (index, x) in enumerate(df.iterrows()):
        instance = x['wordform']

        data += [{
            'wordform': instance,
            'x': [token_map['SOW']] + [token_map[z] for z in instance] + [token_map['EOW']],
            'idx': index,
            'concept_id': x['concept_id_train'],
            'concept_id_shuffled': [x['concept_shuffle_s-%d' % seed] for seed in range(25)],
            'family_weight': x['family_weight'],
        }]

    return data


def save_data(data, fold, ffolder, fsuffix=''):
    fname = '%s/preprocess-%s/data-all-%d.npy' % (ffolder, fsuffix, fold)
    util.write_data(fname, data)


def load_data(fold, ffolder, fsuffix=''):
    fname = '%s/preprocess-%s/data-all-%d.npy' % (ffolder, fsuffix, fold)
    data = util.read_data(fname)
    return data


def save_info(ffolder, fsuffix, token_map, n_concepts, data_split, families_split):
    info = {
        'token_map': token_map,
        'n_concepts': n_concepts,
        'data_split': data_split,
        'families_split': families_split,
    }
    fname = '%s/preprocess-%s/info.pckl' % (ffolder, fsuffix)
    util.write_data(fname, info)


def load_info(ffolder, fsuffix):
    fname = '%s/preprocess-%s/info.pckl' % (ffolder, fsuffix)
    info = util.read_data(fname)

    token_map = info['token_map']
    n_concepts = info['n_concepts']
    data_split = info['data_split']
    families_split = info['families_split']

    return token_map, n_concepts, data_split, families_split


def get_shuffled_concepts(df):
    for seed in range(25):
        shuffled_ids = df['concept_id_train'].values.copy()
        np.random.shuffle(shuffled_ids)
        df['concept_shuffle_s-%d' % seed] = shuffled_ids

    return df


def main(args):
    df = read_src_data(args.ffolder)
    df = filter_word2vec(df, args)
    df = get_shuffled_concepts(df)

    df_split, data_split, families_split = separate_train(df, args.n_folds, family_column=args.data_split)
    token_map = get_tokens(df)
    n_concepts = df.concept_id_train.max() + 1

    process_data(df_split, token_map, args)
    save_info(args.ffolder, args.fsuffix, token_map, n_concepts, data_split, families_split)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.data == 'asjp', 'this script should only be run with asjp data'
    main(args)
