import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.train import run
from util import argparser
from util import util


def get_fold_split(fold, n_folds):
    return ([fold, (fold + 1) % n_folds], [(fold + 2) % n_folds], [(fold + 3) % n_folds])


def run_cv(args):
    full_results = [['fold', 'avg_len', 'test_loss', 'test_acc', 'val_loss', 'val_acc', 'best_epoch']]
    avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc = 0, 0, 0, 0

    for fold in range(args.n_folds):
        print()
        print('Fold:', fold, end=' ')

        folds = get_fold_split(fold, args.n_folds)
        args.model_suffix = 'fold-%d' % fold


        test_loss, test_acc, val_loss, val_acc, best_epoch = \
            run(folds, args)

        full_results += [[fold, test_loss, test_acc, val_loss, val_acc, best_epoch]]

        avg_test_loss += test_loss / args.n_folds
        avg_test_acc += test_acc / args.n_folds
        avg_val_loss += val_loss / args.n_folds
        avg_val_acc += val_acc / args.n_folds

        util.write_csv(full_results, '%s/%s__%s__full-results.csv' % (args.rfolder, args.model, args.context))

    return avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc


def run_languages(args):
    test_loss, test_acc, \
        val_loss, val_acc = run_cv(args)

    results = [['test_loss', 'test_acc', 'val_loss', 'val_acc']]
    results += [[test_loss, test_acc, val_loss, val_acc]]

    util.write_csv(results, '%s/%s__%s__results-final.csv' % (args.rfolder, args.model, args.context))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv')
    run_languages(args)
