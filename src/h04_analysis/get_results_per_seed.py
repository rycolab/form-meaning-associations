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
from h04_analysis.get_results_per_macroarea import load_results_as_df
from util import argparser
from util import constants
from util import util


def main():
    args = argparser.parse_args(csv_folder='cv')
    df_info = AsjpInfo.get_df(args.ffolder)
    # _, _, _, families_split = load_info(args.ffolder, args.fsuffix)
    df = None

    for context in ['none', 'onehot']:
        print(context)
        df_context = load_results_as_df(df_info, context, args)

        if df is None:
            df = df_context
        else:
            df_context = df_context[[context]]
            df = df.join(df_context)

    # Average seeds out
    df = df.reset_index()
    df_seed = df.groupby(['seed', 'fold']).agg('mean').reset_index()

    fname = os.path.join(args.rfolder_base, 'seed_results.tsv')
    df_seed.to_csv(fname, sep='\t')


if __name__ == '__main__':
    main()
