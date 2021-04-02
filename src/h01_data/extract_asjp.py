# Note: Dataset 0 is PAD 1 is SOW and 2 is EOW
import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.asjp import AsjpInfo
from util import argparser as parser
from util import constants
from util import util


def read_wordform_data(ffolder):
    fname = os.path.join(ffolder, 'listss19_formatted.tsv')
    df = pd.read_csv(fname, sep='\t', keep_default_na=False)

    df = df[df['pop'] > -2]
    df = df[df['wls_fam'] != 'Oth']

    df = concepts_to_rows(df)
    df = mark_loan_words(df)

    df['item_id'] = range(df.shape[0])
    df['concept_id'] = pd.factorize(df['concept_name'])[0]
    df['language_name'] = df['names']
    del df['names']
    df['language_id'] = pd.factorize(df['language_name'])[0]
    # df['language_id'] = pd.factorize(df['iso'])[0]

    del df['e']
    del df['hh']
    del df['wcode']

    return df


def concepts_to_rows(df):
    df = df.melt(id_vars=constants.ASJP_EXT_COLUMNS,
                 var_name="concept_name",
                 value_name="wordform")
    df = df[df['wordform'] != 'XXX'].copy()
    df["wordform"] = df["wordform"].str.split(", ")
    df = df.explode("wordform").reset_index(drop=True)

    return df


def mark_loan_words(df):
    # Wordforms starting in % are loan words
    df['wordform'] = df.wordform.apply(lambda x: str(x))
    df = df[df.wordform.apply(lambda x: len(x) != 0)].copy()
    df['Loan'] = df.wordform.apply(lambda x: x[0] == '%')
    df['wordform'] = df.wordform.apply(lambda x: x[1:] if x[0] == '%' else x)

    return df


def get_genealogy_info(df, ffolder, column=None):
    fname = os.path.join(ffolder, 'continents.csv')
    df_continent = pd.read_csv(fname, sep=',', keep_default_na=False, index_col=0)
    df_continent[df_continent == 'NA'] = None
    df_continent.drop_duplicates('ID', inplace=True)
    # df_continent = pd.read_csv(fname, sep=',', index_col=0)

    n_instances = df.shape[0]
    df = pd.merge(df, df_continent, left_on='language_name', right_on='ID', how='left')
    del df['ISO639P3code']
    del df['Glottocode']
    del df['Continent']
    del df['ID']
    del df['lat']
    del df['lon']

    df = df[~df['Family'].isna()]
    n_instances = df.shape[0]

    assert not df['Family'].isna().any()
    assert not df['Macroarea'].isna().any()

    assert n_instances == df.shape[0]

    # Remove Mixed Languages
    df = df[df['Family'] != 'MixedLanguage'].copy()

    return df


def rename_columns(df):
    # df['family'] = df['wls_gen']
    df['genus'] = df['wls_gen']
    del df['wls_gen']
    df['family'] = df['Family']
    del df['Family']
    df['macroarea'] = df['Macroarea']
    del df['Macroarea']
    df['iso_code'] = df['iso']
    del df['iso']

    return df


# Get macroarea where majority of family is
def get_family_macroarea(df):
    df_temp = df[['family', 'macroarea']]
    df_family = df_temp.groupby(['family']).agg(lambda x: x.value_counts().index[0])

    df['macroarea_orig'] = df['macroarea']
    del df['macroarea']

    df = pd.merge(df, df_family, left_on='family', right_on='family')
    return df


def get_family_sizes(df):
    df_family = df[['family', 'item_id']].groupby(['family']).agg('count')

    df_family['family_size'] = df_family['item_id']
    del df_family['item_id']
    df_family['family_weight'] = 1 / df_family['family_size']

    df = pd.merge(df, df_family, left_on='family', right_on='family')

    return df


def get_train_concept_idx(df):
    concepts = df['concept_id'].unique()
    concepts.sort()

    train_ids = {x: i for i, x in enumerate(concepts)}

    df['concept_id_train'] = df['concept_id'].apply(lambda x: train_ids[x])

    return df


def main(args):
    df = read_wordform_data(args.ffolder)
    df = get_genealogy_info(df, args.ffolder)

    df = rename_columns(df)

    df = get_family_macroarea(df)
    df = get_family_sizes(df)

    df = get_train_concept_idx(df)

    df.set_index('item_id', inplace=True)

    fname = '%s/extracted.tsv' % args.ffolder
    df.to_csv(fname, sep='\t')


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.data == 'asjp', 'this script should only be run with asjp data'
    main(args)
