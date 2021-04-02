import pandas as pd
import numpy as np
import json


class AsjpInfo(object):

    @classmethod
    def get_df(cls, ffolder):
        df = cls.read_src_data(ffolder)
        df['phoneme'] = df['wordform'].apply(lambda x: list(x))
        df.set_index('item_id', inplace=True)
        return df

    @classmethod
    def read_src_data(cls, ffolder):
        filename = '%s/extracted.tsv' % ffolder
        df = pd.read_csv(filename, sep='\t', keep_default_na=False)
        return df

    @staticmethod
    def read_form_data(ffolder):
        filename = '%s/forms.csv' % ffolder
        df = pd.read_csv(filename, sep=',', keep_default_na=False)
        df = df[['ID', 'Language_ID', 'Parameter_ID', 'Form', 'Loan']]
        assert df.shape[0] == df.dropna().shape[0]
        return df

    @staticmethod
    def read_parameters_data(ffolder):
        filename = '%s/parameters.csv' % ffolder
        df = pd.read_csv(filename, sep=',', keep_default_na=False)
        df = df[['ID', 'Name']]
        assert df.shape[0] == df.dropna().shape[0]
        return df

    @staticmethod
    def get_language_iso(ffolder):
        filename = '%s/languages.csv' % ffolder
        df = pd.read_csv(filename, sep=',', keep_default_na=False)
        df['iso_code'] = df['ISO639P3code']
        df['Language_ID'] = df['ID']

        df = df[['Language_ID', 'iso_code']]
        assert df.shape[0] == df.dropna().shape[0]
        df.drop(df[df['iso_code'] == ''].index, inplace=True)

        return df

    @staticmethod
    def get_continent_df(ffolder):
        filename = '%s/families/continents.csv' % ffolder
        df = pd.read_csv(filename, sep=',', keep_default_na=False)

        df['Language_ID'] = df['ID']
        del df['ID']
        df['Language_Name'] = df['Name']
        del df['Name']
        del df['Macroarea']
        del df['Unnamed: 0']

        return df

    @classmethod
    def get_glottocode_families(cls, ffolder):
        filename = '%s/families/glottolog4.csv' % ffolder
        df = pd.read_csv(filename, sep=',', keep_default_na=False)
        df['Glottocode'] = df['id']
        df = df[['Glottocode', 'family']]

        return df

    @classmethod
    def get_idx_to_glottocode_families(cls, ffolder):
        df = cls.get_glottocode_family_df()
        idx2families = df['family'].to_dict()

        return idx2families

    @classmethod
    def get_glottocode_family_sizes(cls, ffolder):
        df = cls.get_glottocode_family_df()
        df.drop_duplicates('Language_ID', inplace=True)
        return df.groupby('family').agg('count')['Language_ID'].to_dict()

    @classmethod
    def get_glottocode_family_df(cls, ffolder):
        df = cls.get_df(ffolder)
        glottocodes_families = cls.get_language_glottocode_families(ffolder)

        df = df[df.Language_ID.isin(glottocodes_families)]
        df['family'] = df.Language_ID.apply(lambda x: glottocodes_families[x])
        return df
