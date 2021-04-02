import os
import sys
import math
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import argparser
from util import util


def read_data(args):
    fname = os.path.join(args.rfolder_base, 'tokens_results--corrected.tsv')
    df = pd.read_csv(fname, sep='\t')
    # df_p = pd.read_csv(
    #     'results/asjp/p_values-per_token_concept_and_area-n_tests_%d--corrections.tsv' % (num_tests),
    #     sep='\t')
    df['concept_lower'] = df.apply(lambda x: x.concept_name.lower(), axis=1)
    df = df.sort_values(['concept_lower', 'token'])

    return df


def get_table(df):
    concept_tokens = {}
    for _, item in df.iterrows():
        concept_tokens[item.concept_name] = \
            concept_tokens.get(item.concept_name, []) + [item.token]

    return concept_tokens


def print_table(concept_tokens):
    latex_str = []
    columns = 5
    for i, (concept, tokens) in enumerate(concept_tokens.items()):
        tokens = [x if x != 'EOW' else '$\\stringending$' for x in tokens]
        # print('%s\t[%s]' % (concept, ', '.join(tokens)))
        if i < len(concept_tokens) / columns:
            latex_str += ['%s & %s & ' % (concept, ' '.join(tokens))]
        else:
            row = int(i % math.ceil(len(concept_tokens) / columns))
            # print(row)
            if (len(concept_tokens) - i) > math.ceil(len(concept_tokens) / columns):
                latex_str[row] += '%s & %s & ' % (concept, ' '.join(tokens))
            else:
                latex_str[row] += '%s & %s \\\\ \n' % (concept, ' '.join(tokens))

    return ''.join(latex_str)


def main():
    args = argparser.parse_args(csv_folder='cv')
    context = 'onehot'
    alpha = 0.01

    df = read_data(args)
    df = df[df['significant-%.2f' % alpha]]
    print(df.shape)

    df = df[(df['significant-0.01']) & (df['n_instances'] > 1000)]

    # import ipdb; ipdb.set_trace()
    # df_p = read_p_values(num_tests)
    concept_tokens = get_table(df)
    latex_str = print_table(concept_tokens)

    print()
    print()
    print(latex_str)


if __name__ == '__main__':
    main()
