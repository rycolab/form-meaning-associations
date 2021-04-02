import argparse
import torch
from . import util
from . import constants

parser = argparse.ArgumentParser(description='Phoneme LM')

# Data
parser.add_argument('--data', type=str, default='asjp',
                    help='Dataset used. (default: asjp)')
parser.add_argument('--data-path', type=str, default='datasets',
                    help='Path where data is stored.')
parser.add_argument('--data-split', type=str, default='macroarea',
                    help='Column to use for data split.')
parser.add_argument('--n-folds', type=int, default=4,
                    help='Number of folds for cross validation')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size for running model')

# Model
parser.add_argument('--model', default='lstm', choices=['lstm'],
                    help='Model used. (default: lstm)')
parser.add_argument('--context', default='none', choices=['none', 'onehot', 'onehot-shuffle',
                                                          'word2vec', 'word2vec-shuffle'],
                    help='Context used for systematicity. (default: none)')
parser.add_argument('--opt', action='store_true', default=False,
                    help='Should use optimum parameters in training.')

# Others
parser.add_argument('--results-path', type=str, default='results',
                    help='Path where results should be stored.')
parser.add_argument('--checkpoint-path', type=str, default='checkpoints',
                    help='Path where checkpoints should be stored.')
parser.add_argument('--csv-folder', type=str, default=None,
                    help='Specific path where to save results.')
parser.add_argument('--seed', type=int, default=7,
                    help='Seed for random algorithms repeatability (default: 7)')
parser.add_argument('--gpu-id', default=0, type=int,
                    help='Which gpu to use. Set -1 to use CPU')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


def parse_args(*args, csv_folder='', orig_folder=True, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    constants.set_device(args)
    args.model_suffix = None
    args.shuffle_concepts = 'shuffle' in args.context

    csv_folder = csv_folder if csv_folder != 'normal' or not args.opt else 'opt'
    csv_folder = csv_folder if args.csv_folder is None else args.csv_folder
    args.ffolder = '%s/%s' % (args.data_path, args.data)  # Data folder
    args.fsuffix = args.data_split

    args.rfolder_base = '%s/%s/%s/' % (args.results_path, args.data, args.fsuffix)  # Results base folder
    args.cfolder_base = '%s/%s/%s/' % (args.checkpoint_path, args.data, args.fsuffix)  # Checkpoint folder
    args.cfolder = '%s/%s' % (args.cfolder_base, csv_folder)  # Checkpoint folder

    if orig_folder:
        args.rfolder = '%s/%s/seed_%03d' % (args.rfolder_base, csv_folder, args.seed)  # Results folder
        args.cfolder = '%s/%s/seed_%03d/' % (args.cfolder_base, csv_folder, args.seed)  # Checkpoint folder
    else:
        args.rfolder = '%s/%s' % (args.rfolder_base, csv_folder)  # Results folder

    util.mkdir(args.rfolder)
    util.mkdir(args.cfolder)
    util.config(args.seed)
    return args
