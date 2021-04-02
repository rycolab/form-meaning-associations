import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.parse import load_info
from util import argparser
from gp import bayesian_optimisation
from train import get_model_entropy, run_language
# from train import get_model_entropy, get_data_loaders, write_csv, run_language
from h02_learn.dataset import get_data_loaders
from util import util


results = [['embedding_size', 'hidden_size', 'concept_size', 'nlayers', 'dropout',
            'test_loss', 'test_acc', 'val_loss', 'val_acc', 'best_epoch']]


def sample_loss_getter(folds, args):
    global count
    # train_loader, val_loader, test_loader = get_data_loaders(lang, args.ffolder, args)
    train_loader, val_loader, test_loader, token_map, n_concepts = get_data_loaders(folds, args.ffolder, args.fsuffix, args.batch_size, args.shuffle_concepts, args.seed)
    wait_epochs = 1
    count = 0

    def sample_loss(hyper_params):
        global results, count
        # global results
        count += 1

        embedding_size = int(2 ** hyper_params[0])
        hidden_size = int(2 ** hyper_params[1])
        concept_size = int(2 ** hyper_params[2])
        nlayers = int(max(1, hyper_params[3]))
        dropout = max(0, hyper_params[4])
        print('%d: emb-hs %d  hs %d  w2v %d  nlayers %d  drop %.3f' %
              (count, embedding_size, hidden_size, concept_size, nlayers, dropout))

        test_loss, test_acc, val_loss, val_acc, best_epoch = get_model_entropy(
            train_loader, val_loader, test_loader, token_map, embedding_size, hidden_size, n_concepts,
            concept_size, nlayers, dropout, args, wait_epochs=wait_epochs)

        results += [[
            embedding_size, hidden_size, concept_size, nlayers, dropout,
            test_loss, test_acc, val_loss, val_acc, best_epoch]]
        return val_loss

    return sample_loss


def get_optimal_loss(folds, xp, yp, args):
    best_hyperparams = xp[np.argmin(yp)]
    embedding_size = int(2 ** best_hyperparams[0])
    hidden_size = int(2 ** best_hyperparams[1])
    concept_size = int(2 ** best_hyperparams[2])
    nlayers = int(max(1, best_hyperparams[3]))
    dropout = max(0, best_hyperparams[4])
    print('Best hyperparams emb-hs: %d, hs: %d, w2v: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, concept_size, nlayers, dropout))

    test_loss, test_acc, val_loss, val_acc, best_epoch = run_language(
        folds, args, embedding_size=embedding_size, hidden_size=hidden_size,
        concept_size=concept_size, nlayers=nlayers, dropout=dropout)
    return [test_loss, test_acc, val_loss, val_acc, best_epoch,
            embedding_size, hidden_size, concept_size, nlayers, dropout]


def optimize_languages(args):
    print('Model %s\t Context %s' % (args.model, args.context))
    folds = (range(2), [2], [3])

    # word2vec size needs to be at most the original size or number of samples. In asjp we have 100 words
    bounds = np.array([[2, 10], [5, 10], [.5, 6.6], [1, 4.95], [0.0, 0.5]])
    n_iters = 40
    n_pre_samples = 10


    sample_loss = sample_loss_getter(folds, args)
    xp, yp = bayesian_optimisation(n_iters, sample_loss, bounds, n_pre_samples=n_pre_samples)

    final_results = get_optimal_loss(folds, xp, yp, args)

    opt_results = [['test_loss', 'test_acc', 'val_loss', 'val_acc', 'best_epoch',
                    'embedding_size', 'hidden_size', 'concept_size',
                    'nlayers', 'dropout']]
    opt_results += [final_results]


    util.write_csv(opt_results, '%s/%s__%s__opt-results.csv' % (args.rfolder, args.model, args.context))
    util.write_csv(results, '%s/%s__%s__baysian-results-final.csv' % (args.rfolder, args.model, args.context))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='bayes-opt')
    optimize_languages(args)
