import os
import sys
import math
from tqdm import tqdm
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import argparser


def get_corrections(df, alpha):
    n_instances = df.shape[0]
    df.sort_values('p_value', inplace=True, ascending=True)
    df['range'] = range(1, n_instances + 1)
    df['threshold'] = df['range'] * alpha / n_instances
    df['significant'] = df['p_value'] < df['threshold']
    # df[df['significant']]
    last_significant = df[df['significant']].iloc[-1].range
    df.loc[df['range'] <= last_significant, 'significant'] = True

    df['significant-%.2f' % alpha] = df['significant']

    print('\tSignificant instances (%.2f): %d / %d' % (alpha, df['significant'].sum(), n_instances))
    del df['range']
    del df['threshold']
    del df['significant']
    return df


def add_corrections(folder, src_fname, tgt_fname, alphas):
    fname = os.path.join(folder, src_fname)
    if not os.path.isfile(fname):
        return

    df = pd.read_csv(fname, sep='\t')
    df['alphas'] = ', '.join([str(x) for x in alphas])
    for alpha in alphas:
        df = get_corrections(df, alpha=alpha)

    fname = os.path.join(folder, tgt_fname)
    df.to_csv(fname, sep='\t')


def add_token_corrections(folder, src_fname, tgt_fname, alphas):
    fname = os.path.join(folder, src_fname)
    if not os.path.isfile(fname):
        return

    df = pd.read_csv(fname, sep='\t')
    df.sort_values('p_value', inplace=True, ascending=True)
    df_mean = df.groupby(['concept_name', 'token']).agg('mean')
    df_sum = df.groupby(['concept_name', 'token']).agg('sum')
    # df_min = df.groupby(['concept_id', 'token_idx']).agg('min')
    # (df_min['n_instances'] < 20)

    df['alphas'] = ', '.join([str(x) for x in alphas])
    for alpha in alphas:
        df = get_corrections(df, alpha=alpha)

    df_all = df.groupby(['concept_name', 'token']).agg('all')

    df_mean['n_instances'] = df_sum['n_instances']
    for alpha in alphas:
        df_mean['significant-%.2f' % alpha] = df_all['significant-%.2f' % alpha]

    fname = os.path.join(folder, tgt_fname)
    df_mean.to_csv(fname, sep='\t')


def main():
    args = argparser.parse_args(csv_folder='cv')
    context = 'onehot'
    alphas = [0.01, 0.05, 0.1]

    print('Languages')
    add_corrections(args.rfolder_base, 'languages_results.tsv', 'languages_results--corrected.tsv', alphas=alphas)
    print('Concepts')
    add_corrections(args.rfolder_base, 'concepts_results.tsv', 'concepts_results--corrected.tsv', alphas=alphas)
    print('Tokens')
    add_token_corrections(args.rfolder_base, 'tokens_results.tsv', 'tokens_results--corrected.tsv', alphas=alphas)


if __name__ == '__main__':
    main()
