import os
import sys
import math
from tqdm import tqdm
import torch
import torch.nn as nn

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.dataset import get_data_loaders
from h02_learn.model.lstm import IpaLM
from h02_learn.train_cv import get_fold_split
from util import argparser
from util import constants
from util import util


def load_model(model_name, args):
    if model_name == 'lstm':
        model_cls = IpaLM
    else:
        raise ValueError("Model not implemented: %s" % model_name)

    return model_cls.load(args.cfolder, args.context, args.model_suffix) \
        .to(device=constants.device)


def merge_tensors(losses, fill=0):
    max_len = max(x.shape[-1] for x in losses)
    n_sentences = sum(x.shape[0] for x in losses)

    full_loss = torch.ones(n_sentences, max_len) * fill

    start, end = 0, 0
    for loss in losses:
        end += loss.shape[0]
        batch_len = loss.shape[-1]
        full_loss[start:end, :batch_len] = loss
        start = end

    return full_loss


def get_loss(criterion, y_hat, y):
    return criterion(y_hat.view(-1, y_hat.size(-1)), y.view(-1)).reshape_as(y) / math.log(2)


def get_loss_no_eos(criterion, y_hat, y):
    y_hat = y_hat.clone()
    y_hat[:, :, 2] = -1e9
    return get_loss(criterion, y_hat, y)


def eval_per_char(model, dataloader):
    pad_idx = 0
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none').to(device=constants.device)

    ids, ys, losses, losses_no_eos, lengths = [], [], [], [], []
    dev_loss, n_instances = 0, 0
    for x, y, concepts, weights, batch_ids in tqdm(dataloader, desc='Evaluating per char'):
        y_hat = model(x, concepts)
        loss = get_loss(criterion, y_hat, y)
        loss_no_eos = get_loss_no_eos(criterion, y_hat, y)

        sent_lengths = (y != pad_idx).sum(-1)
        batch_size = y.shape[0]
        dev_loss += ((loss.sum(-1) / sent_lengths) * weights / weights.sum()).sum()
        n_instances += batch_size
        losses += [loss.cpu()]
        losses_no_eos += [loss_no_eos.cpu()]
        ids += [batch_ids.cpu()]
        ys += [y.cpu()]
        lengths += [sent_lengths.cpu()]

    losses = merge_tensors(losses)
    losses_no_eos = merge_tensors(losses_no_eos)
    ys = merge_tensors(ys, fill=pad_idx)
    lengths = torch.cat(lengths, dim=0)
    ids = torch.cat(ids, dim=0)

    results = {
        'ids': ids,
        'losses': losses,
        'losses_no_eos': losses_no_eos,
        'ys': ys,
        'lengths': lengths,
        'pad_idx': pad_idx,
    }

    return results


def eval_fold(fold, args):
    folds = get_fold_split(fold, args.n_folds)
    args.model_suffix = 'fold-%d' % fold

    model = load_model(args.model, args)

    _, _, test_loader, token_map, n_concepts = \
        get_data_loaders(folds, args.ffolder, args.fsuffix, args.batch_size, args.shuffle_concepts, args.seed)

    fold_results = eval_per_char(model, test_loader)

    return fold_results


def eval_cv(args):
    results = []
    for fold in range(args.n_folds):
        print()
        print('Fold:', fold)

        fold_results = eval_fold(fold, args)
        results += [fold_results]

    return {
        'model': args.model,
        'data_split': args.data_split,
        'results': results,
    }


def main():
    with torch.no_grad():
        results = eval_cv(args)

    results_file = '%s/losses__%s__%s.pckl' % (args.cfolder, args.model, args.context)
    util.write_data(results_file, results)


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv')
    main()
