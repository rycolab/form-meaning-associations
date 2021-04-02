import os
import sys
import math
from tqdm import tqdm
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import argparser


def permutation_test(df, column, n_permutations=100000, batch_size=1000):
    # Get actual batch size
    batch_size = min(batch_size, n_permutations)

    # Get real avg
    real_avg = df[column].mean().item()
    values = df[column].values
    values_exp = np.expand_dims(values, axis=1).repeat(batch_size, axis=1)

    # Make n_permutations divisible per batch_size
    n_batches = math.ceil(n_permutations / batch_size)
    n_permutations = n_batches * batch_size

    # Get number of permutations above
    n = 0
    for _ in range(n_batches):
        permut = np.random.randint(0, 2, size=(len(values), batch_size)) * 2 - 1

        random_avgs = np.mean(values_exp * permut, axis=0)
        n += (random_avgs >= real_avg).sum()

    return n / n_permutations, n_permutations


def remove_unused_cols(df):
    del df['item_id']
    del df['position']
    del df['family_size']
    del df['family_weight']
    del df['length']

    return df


def get_macroarea_counts(df):
    df_count = df[['macroarea', 'concept_id', 'token_idx']].groupby(['concept_id', 'token_idx']).agg('count').reset_index()
    df_count['macroarea_count'] = df_count['macroarea']
    del df_count['macroarea']
    df = pd.merge(df, df_count, left_on=['concept_id', 'token_idx'], right_on=['concept_id', 'token_idx'])

    return df


def get_tokens_means(df):
    df_new = df.groupby(['language_id', 'family', 'macroarea', 'concept_id', 'concept_name', 'token', 'token_idx']).agg('mean').reset_index()
    df_new = df_new.groupby(['family', 'macroarea', 'concept_id', 'concept_name', 'token', 'token_idx']).agg('mean').reset_index()
    df_new = df_new.groupby(['macroarea', 'concept_id', 'concept_name', 'token', 'token_idx']).agg('mean').reset_index()

    df_new = get_macroarea_counts(df_new)
    return df_new


def main():
    args = argparser.parse_args(csv_folder='cv')
    context = 'onehot'

    fname = os.path.join(args.rfolder_base, 'avg_seed_results_per_pos.tsv')
    df = pd.read_csv(fname, sep='\t')
    remove_unused_cols(df)

    # df_concepts = df.groupby(['concept_id', 'concept_name']).agg('mean').reset_index()
    df_tokens = get_tokens_means(df)
    df_tokens.set_index(['macroarea', 'concept_id', 'token_idx'], inplace=True)
    df_tokens = df_tokens.sort_index()
    df_tokens = df_tokens[df_tokens.macroarea_count == 4]

    df_tokens['p_value'] = -1
    df_tokens['n_permutations'] = -1
    df_tokens['n_instances'] = -1
    for macroarea, concept_id, token_idx in tqdm(df_tokens.index.unique(), desc='Concept--token permutation tests'):
        idx = (macroarea, concept_id, token_idx)
        df_temp = df[(df.macroarea == macroarea) & (df.concept_id == concept_id) & (df.token_idx == token_idx)]
        p_val, n_permutations = permutation_test(df_temp, 'mi-' + context, n_permutations=100000)
        # p_val, n_permutations = permutation_test_recursive(df_temp, 'mi-' + context, n_permutations=100000)

        df_tokens.loc[idx, 'p_value'] = p_val
        df_tokens.loc[idx, 'n_permutations'] = n_permutations
        df_tokens.loc[idx, 'n_instances'] = df_temp.shape[0]

    fname = os.path.join(args.rfolder_base, 'tokens_results.tsv')
    df_tokens.to_csv(fname, sep='\t')


if __name__ == '__main__':
    main()
