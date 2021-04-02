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


def permutation_test_recursive(df, column, n_permutations=100000, batch_size=1000):
    # Get rid of useless columns
    df = df[['language_id', 'family', 'macroarea', column]].copy()

    # Create columns for permut batch
    batch_size = min(batch_size, n_permutations)
    for i in range(batch_size):
        df['permut-%d' % i] = float('inf')
    permut_cols = ['permut-%d' % i for i in range(batch_size)]

    # Get real avg
    real_avg = recursive_mean(df)[column].item()
    values = df[column].values
    values_exp = np.expand_dims(values, axis=1).repeat(batch_size, axis=1)

    # Make n_permutations divisible per batch_size
    n_batches = math.ceil(n_permutations / batch_size)
    n_permutations = n_batches * batch_size

    # Get number of permutations above
    n = 0
    for _ in tqdm(range(n_batches), total=n_batches, desc='Specific concept batched tests'):
        permut = np.random.randint(0, 2, size=(len(values), batch_size)) * 2 - 1

        df[permut_cols] = values_exp * permut
        random_avgs = recursive_mean(df)

        n += (random_avgs[permut_cols] >= real_avg).sum()

    # Print results
    tqdm.write('N: %d\tPermuts: %d\tN: %.4f' % (n, n_permutations, n / n_permutations))
    return n / n_permutations, n_permutations


def get_concept_means(df):
    df_new = df.groupby(['language_id', 'family', 'macroarea', 'concept_id', 'concept_name']).agg('mean').reset_index()
    df_new = df_new.groupby(['family', 'macroarea', 'concept_id', 'concept_name']).agg('mean').reset_index()
    df_new = df_new.groupby(['macroarea', 'concept_id', 'concept_name']).agg('mean').reset_index()
    df_new = df_new.groupby(['concept_id', 'concept_name']).agg('mean').reset_index()

    del df_new['item_id']
    del df_new['Latitude']
    del df_new['Longitude']
    del df_new['family_size']

    return df_new


def main():
    args = argparser.parse_args(csv_folder='cv')
    context = 'onehot'

    fname = os.path.join(args.rfolder_base, 'avg_seed_results.tsv')
    df = pd.read_csv(fname, sep='\t')

    # df_concepts = df.groupby(['concept_id', 'concept_name']).agg('mean').reset_index()
    df_concepts = get_concept_means(df)
    df_concepts.set_index('concept_id', inplace=True)

    df_concepts['p_value'] = -1
    df_concepts['n_permutations'] = -1
    for concept_id in tqdm(df.concept_id.unique(), desc='Concept permutation tests'):
        df_temp = df[df.concept_id == concept_id]
        p_val, n_permutations = permutation_test_recursive(df_temp, 'mi-' + context, n_permutations=100000)

        df_concepts.loc[concept_id, 'p_value'] = p_val
        df_concepts.loc[concept_id, 'n_permutations'] = n_permutations

    fname = os.path.join(args.rfolder_base, 'concepts_results.tsv')
    df_concepts.to_csv(fname, sep='\t')


if __name__ == '__main__':
    main()
