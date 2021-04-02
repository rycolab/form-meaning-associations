import os
import sys
import math
from tqdm import tqdm
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.asjp import AsjpInfo
from h01_data.parse import load_info
from h02_learn.model.lstm import IpaLM
from h02_learn.train_cv import get_fold_split
from util import argparser
from util import constants
from util import util


def load_results(seed, context, args):
    results_file = '%s/cv/seed_%03d/losses__%s__%s.pckl' % (args.cfolder_base, seed, args.model, context)
    results = util.read_data(results_file)
    return results['results']


def get_per_word_results(result, context):
    data = {
        'ids': result['ids'],
        'length': result['lengths'],
        context + '_full': result['losses'].sum(-1),
        context: result['losses'].sum(-1) / result['lengths'],
    }

    return pd.DataFrame(data)


def group_recursevely(df):
    df_new = df.groupby(['language_id', 'family', 'macroarea', 'fold']).agg('mean').reset_index()
    df_new = df_new.groupby(['family', 'macroarea', 'fold']).agg('mean').reset_index()
    df_new = df_new.groupby(['macroarea', 'fold']).agg('mean').reset_index()

    return df_new


def load_results_as_df_seed(df_info, seed, context, args):
    results = load_results(seed, context, args)
    dfs = []

    for fold, result in enumerate(results):
        df = get_per_word_results(result, context)
        df['fold'] = fold
        dfs += [df]

    df = pd.concat(dfs).set_index('ids')
    df = df_info.join(df)

    # Drop loan words
    df = df[~df.Loan]

    return group_recursevely(df).set_index(['macroarea', 'fold'])


def load_results_as_df(df_info, context, args):
    dfs = []
    for seed in range(25):
        # print(context)
        df_seed = load_results_as_df_seed(df_info, seed, context, args)
        df_seed['seed'] = seed

        dfs += [df_seed]

    df = pd.concat(dfs).reset_index()
    df = df[['macroarea', 'seed', 'length', 'fold', context]]

    # Average seeds out
    return df.set_index(['macroarea', 'fold', 'seed'])


def permutation_test(macroarea, df, column1, column2, n_permutations=10000):
    values = df[column1] - df[column2]
    p_value = util.test_permutations(values, n_permutations)

    print(macroarea, p_value)
    return p_value


def welchs_test(macroarea, df, column1, column2):
    # get one sides welch's ttest p_value
    _, p_value = stats.ttest_ind(df[column1], df[column2], equal_var=False)
    p_value = p_value / 2

    print(macroarea, p_value)
    return p_value


def add_train_dev_columns(df, families_split):
    df['train'] = -1
    df['dev'] = -1
    df['test'] = -1

    for i, macroarea in enumerate(families_split):
        macroarea = macroarea[0]
        fold = (i + 1) % 4
        folds = get_fold_split(fold, 4)
        # df[df.macroarea]

        # import ipdb; ipdb.set_trace()
        df.loc[df['macroarea'] == macroarea, 'train'] = ', '.join([families_split[x][0] for x in folds[0]])
        df.loc[df['macroarea'] == macroarea, 'dev'] = ', '.join([families_split[x][0] for x in folds[1]])
        df.loc[df['macroarea'] == macroarea, 'test'] = ', '.join([families_split[x][0] for x in folds[2]])

    assert (df.test == df.macroarea).all(), 'Macroarea should match test area'
    return df


def get_mis(df):
    for context in ['onehot']:
        df['mi-' + context] = df['none'] - df[context]
        df['unc-' + context] = df['mi-' + context] / df['none']

        df['mi-' + context + '-shuffle'] = df[context + '-shuffle'] - df[context]
        df['unc-' + context + '-shuffle'] = df['mi-' + context + '-shuffle'] / df[context + '-shuffle']

    return df


def main():
    args = argparser.parse_args(csv_folder='cv')
    df_info = AsjpInfo.get_df(args.ffolder)
    _, _, _, families_split = load_info(args.ffolder, args.fsuffix)
    df = None

    for context in ['none', 'onehot', 'onehot-shuffle']:
        print(context)
        df_context = load_results_as_df(df_info, context, args)

        if df is None:
            df = df_context
        else:
            df_context = df_context[[context]]
            df = df.join(df_context)

    # Average seeds out
    df = df.reset_index()
    df_end = df.groupby(['macroarea']).agg('mean').reset_index()
    df_end = add_train_dev_columns(df_end, families_split)
    df_end.set_index('macroarea', inplace=True)
    df_end = get_mis(df_end)

    df_overall = df_end.copy()
    df_overall.loc['overall'] = df_end.mean()
    df_overall = df_overall.loc[['overall']]

    for context in ['onehot']:
        df_end['p_value-' + context] = -1
        print ('\nTest context %s' % context)
        for macroarea in df.macroarea.unique():
            df_macro = df[df['macroarea'] == macroarea]
            # permutation_test(macroarea, df_macro, 'none', 'onehot', n_permutations=10000)
            p_val = permutation_test(macroarea, df_macro, 'none', context, n_permutations=10000)
            df_end.loc[macroarea, 'p_value-' + context] = p_val

        p_val = permutation_test('Overall', df, 'none', context, n_permutations=10000)
        df_overall['p_value-' + context] = p_val
        # welchs_test(macroarea, df_macro, 'none', context)

    for context in ['onehot']:
        df_end['p_value-' + context + '-shuffle'] = -1
        print ('\nTest context %s' % context)
        for macroarea in df.macroarea.unique():
            df_macro = df[df['macroarea'] == macroarea]
            p_val = permutation_test(macroarea, df_macro, context + '-shuffle', context, n_permutations=10000)
            df_end.loc[macroarea, 'p_value-' + context + '-shuffle'] = p_val

        p_val = permutation_test('Overall', df, context + '-shuffle', context, n_permutations=10000)
        df_overall['p_value-' + context + '-shuffle'] = p_val

    del df_overall['seed']
    del df_end['seed']

    fname = os.path.join(args.rfolder_base, 'macroarea_results.tsv')
    df_end.to_csv(fname, sep='\t')

    fname = os.path.join(args.rfolder_base, 'overall_results.tsv')
    df_overall.to_csv(fname, sep='\t')


if __name__ == '__main__':
    main()
