import os
import sys
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
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


def load_results_as_df_seed(df_info, seed, context, args):
    results = load_results(seed, context, args)
    dfs = []

    for fold, result in enumerate(results):
        df = get_per_word_results(result, context)
        dfs += [df]

    df = pd.concat(dfs).set_index('ids')
    df = df_info.join(df)

    # Drop loan words
    df = df[~df.Loan]

    return df
    # return group_recursevely(df).set_index('macroarea')


def load_results_as_df(df_info, context, args):
    dfs = []
    for seed in range(25):
        # print(context)
        df_seed = load_results_as_df_seed(df_info, seed, context, args)
        df_seed['seed'] = seed

        dfs += [df_seed]

    df = pd.concat(dfs).reset_index()

  # 'glottocode', 'continent', 'ID',
    group_cols = ['item_id', 'concept_id', 'language_id', 'family', 'macroarea', 'iso_code',
                  'Latitude', 'Longitude', 'family_size', 'concept_name',
                  'wordform', 'macroarea_orig']
    df = df[group_cols + ['seed', 'length', context]]
    # Average seeds out
    df_mean = df.groupby(group_cols).agg('mean').reset_index()
    del df_mean['seed']

    return df_mean.set_index(['item_id'])


def get_mis(df):
    for context in ['onehot']:
        df['mi-' + context] = df['none'] - df[context]
        df['unc-' + context] = df['mi-' + context] / df['none']

    return df


def main():
    args = argparser.parse_args(csv_folder='cv')
    df_info = AsjpInfo.get_df(args.ffolder)

    df = None
    for context in ['none', 'onehot']:
        print(context)
        df_context = load_results_as_df(df_info, context, args)

        if df is None:
            df = df_context
        else:
            df_context = df_context[[context]]
            df = df.join(df_context)

    df = get_mis(df)

    fname = os.path.join(args.rfolder_base, 'avg_seed_results.tsv')
    df.to_csv(fname, sep='\t')


if __name__ == '__main__':
    main()
