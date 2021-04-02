import os
import sys
import math
from tqdm import tqdm
import pandas as pd
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import argparser


def recursive_mean(df):
    df_new = df.groupby(['language_id', 'family', 'macroarea']).agg('mean').reset_index()
    df_new = df_new.groupby(['family', 'macroarea']).agg('mean').reset_index()
    df_new = df_new.groupby(['macroarea']).agg('mean').reset_index()

    return df_new.mean()


def get_p_value_permutation(values, num_tests):
    real_avg = np.mean(values)

    permut = np.random.randint(0, 2, size=(len(values), num_tests)) * 2 - 1
    values_exp = np.expand_dims(values, axis=1).repeat(num_tests, axis=1)

    random_avgs = np.mean(values_exp * permut, axis=0)
    n = (random_avgs >= real_avg).sum()

    return n / num_tests


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


def main():
    args = argparser.parse_args(csv_folder='cv')
    context = 'onehot'

    fname = os.path.join(args.rfolder_base, 'avg_seed_results.tsv')
    df = pd.read_csv(fname, sep='\t', keep_default_na=False)

    df_languages = df.groupby(['language_id', 'macroarea']).agg('mean').reset_index()

    lang2iso = df[['language_id', 'iso_code']].drop_duplicates('language_id').set_index('language_id')['iso_code'].to_dict()
    df_languages['iso_code'] = df_languages.language_id.apply(lambda x: lang2iso[x])
    df_languages.set_index('language_id', inplace=True)

    del df['item_id']
    del df['concept_id']

    df_languages['p_value'] = -1
    df_languages['n_permutations'] = -1
    df_languages['n_concepts'] = -1
    for language_id in tqdm(df.language_id.unique(), desc='Language permutation tests'):
        df_temp = df[df.language_id == language_id]
        p_val, n_permutations = permutation_test(df_temp, 'mi-' + context, n_permutations=1000000)

        df_languages.loc[language_id, 'p_value'] = p_val
        df_languages.loc[language_id, 'n_permutations'] = n_permutations
        df_languages.loc[language_id, 'n_concepts'] = df_temp.shape[0]

    fname = os.path.join(args.rfolder_base, 'languages_results.tsv')
    df_languages.to_csv(fname, sep='\t')


if __name__ == '__main__':
    main()
