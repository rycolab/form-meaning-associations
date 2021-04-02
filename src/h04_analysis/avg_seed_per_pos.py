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
# from h02_learn.dataset import get_data_loaders
from h02_learn.model.lstm import IpaLM
from h02_learn.train_cv import get_fold_split
from util import argparser
from util import constants
from util import util


def load_results(seed, context, args):
    results_file = '%s/cv/seed_%03d/losses__%s__%s.pckl' % (args.cfolder_base, seed, args.model, context)
    results = util.read_data(results_file)
    return results['results']


def get_per_token_results(result, context, token_map_inv):
    ids = result['ids'].numpy()
    lengths = result['lengths'].numpy()
    ys = result['ys'].numpy()
    losses = result['losses'].numpy()

    data = []
    for i in range(ys.shape[0]):
        tokens = [token_map_inv[x] for x in ys[i] if token_map_inv[x] != 'PAD']
        length = lengths[i]

        data += [{
            'ids': ids[i],
            'length': length,
            # 'position': j,
            'token_idx': ys[i, :length],
            'position': list(range(length)),
            'token': tokens,
            context: losses[i, :length],
        }]

    return pd.DataFrame(data)


def load_results_as_df_seed(df_info, seed, context, args):
    results = load_results(seed, context, args)
    token_map, _, _, _ = load_info(args.ffolder, args.fsuffix)
    token_map_inv = {idx: token for token, idx in token_map.items()}

    dfs = []

    for fold, result in enumerate(results):
        df = get_per_token_results(result, context, token_map_inv)
        # df = get_per_token_results(result, context)
        dfs += [df]

    df = pd.concat(dfs).set_index('ids')
    df = df_info.join(df)

    # Drop loan words
    df = df[~df.Loan]

    return df


def agg_seeds(x):
    losses = np.array((x.values.tolist()))
    return tuple(losses.mean(0).tolist())


def avg_seeds(df, context):
    df_new = df.copy()
    df_new.drop_duplicates('item_id', inplace=True)
    df_new.set_index('item_id', inplace=True)
    del df_new[context]
    del df_new['phoneme']

    # Aggregate seeds
    df = df.groupby('item_id').agg({context: agg_seeds})
    df = df_new.join(df).reset_index()

    # Explode results per token
    # 'glottocode',
    unexploded_cols = [
        'item_id', 'concept_id', 'family',
        'concept_name', 'language_id', 'wordform',
        'macroarea_orig', 'macroarea', 'family_size',
        'family_weight', 'concept_id_train', 'length'
    ]
    df = df[unexploded_cols + ['token_idx', 'token', 'position', context]]
    df = df.set_index(unexploded_cols).apply(pd.Series.explode).reset_index()

    return df


def load_results_as_df(df_info, context, args):
    dfs = []
    for seed in tqdm(range(25), total=25, desc='Getting seeds'):
        # print(context)
        df_seed = load_results_as_df_seed(df_info, seed, context, args)
        df_seed['seed'] = seed

        dfs += [df_seed]

    df = pd.concat(dfs).reset_index()

    df_mean = avg_seeds(df, context)

    return df_mean.set_index(['item_id', 'position'])


def get_mis(df):
    for context in ['onehot']:
        df['mi-' + context] = df['none'] - df[context]
        df['unc-' + context] = df['mi-' + context] / df['none']

    return df


def main():
    args = argparser.parse_args(csv_folder='cv')
    df_info = AsjpInfo.get_df(args.ffolder)

    df = None
    for context in tqdm(['none', 'onehot'], desc='Getting contexts'):
        tqdm.write(context)
        df_context = load_results_as_df(df_info, context, args)

        if df is None:
            df = df_context
        else:
            df_context = df_context[[context]]
            df = df.join(df_context)

    df = get_mis(df)

    fname = os.path.join(args.rfolder_base, 'avg_seed_results_per_pos.tsv')
    df.to_csv(fname, sep='\t')

if __name__ == '__main__':
    main()
