import os
import sys
import numpy as np
import math
import csv
from tqdm import tqdm
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.asjp import AsjpInfo
from h01_data.parse import load_info, load_data
from h02_learn.train_info import TrainInfo
from h02_learn.dataset import get_data_loaders
from h02_learn.model import opt_params
from h02_learn.model.lstm import IpaLM
from util import argparser
from util import constants
from util import util


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batches, (batch_x, batch_y, batch_concepts, batch_weights, _) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(batch_x, batch_concepts)
        l = get_loss(criterion, y_hat, batch_y, batch_weights)
        l.backward()
        optimizer.step()

        total_loss += l.item()
    return total_loss / (batches + 1)


def eval(data_loader, model, criterion):
    model.eval()
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    for batches, (batch_x, batch_y, batch_concepts, batch_weights, _) in enumerate(data_loader):
        y_hat = model(batch_x, batch_concepts)
        l = get_loss(criterion, y_hat, batch_y, batch_weights)
        val_loss += l.item() * batch_y.size(0)

        non_pad = batch_y != 0
        val_acc += (y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)

        total_sent += batch_y.size(0)

    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    return val_loss, val_acc


def get_loss(criterion, y_hat, batch_y, batch_weights):
    l = criterion(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)).reshape_as(batch_y) / math.log(2)
    words_len = (batch_y != 0).sum(-1).float()
    l = l.sum(-1) / words_len
    l = l * batch_weights / batch_weights.sum()
    return l.sum()


def idx_to_word(word, token_map_inv, ignored_tokens):
    _w = [token_map_inv[x] for x in word.tolist() if x not in ignored_tokens]
    return ' '.join(_w)


def train(train_loader, val_loader, model, criterion, optimizer, wait_epochs=2):
    pbar = tqdm(total=wait_epochs)
    train_info = TrainInfo(wait_epochs, pbar)

    while True:
        total_loss = train_epoch(train_loader, model, criterion, optimizer)
        val_loss, val_acc = eval(val_loader, model, criterion)

        if train_info.is_best(val_loss, val_acc):
            model.set_best()

        train_info.update_info(total_loss, val_loss, val_acc)
        if train_info.finish:
            break

    model.recover_best()

    return train_info.best_epoch, train_info.best_loss, train_info.best_acc


def _get_avg_len(data_loader):
    total_phon, total_sent = 0.0, 0.0
    for batches, (batch_x, batch_y, _, _, _) in enumerate(data_loader):
        batch = torch.cat([batch_x, batch_y[:, -1:]], dim=-1)
        total_phon += (batch != 0).sum().item()
        total_sent += batch.size(0)

    avg_len = (total_phon * 1.0 / total_sent) - 2  # Remove SOW and EOW tag in every sentence

    return avg_len, total_sent


def get_avg_len(data_loaders):
    total_len, total_nsent = 0, 0
    for data_loader in data_loaders:
        length, nsentences = _get_avg_len(data_loader)
        total_len += (length * nsentences)
        total_nsent += nsentences

    return total_len * 1.0 / total_nsent


def init_model(model_name, context, hidden_size, n_concepts, concept_size, token_map, embedding_size, nlayers, dropout, args):
    vocab_size = len(token_map)
    if model_name == 'lstm':
        model = IpaLM(
            vocab_size, n_concepts, hidden_size, embedding_size=embedding_size, concept_size=concept_size,
            nlayers=nlayers, dropout=dropout, context=context, data=args.data).to(device=constants.device)
    else:
        raise ValueError("Model not implemented: %s" % model_name)

    return model


def get_model_entropy(
        train_loader, val_loader, test_loader, token_map, embedding_size, hidden_size, n_concepts, concept_size,
        nlayers, dropout, args, wait_epochs=2):
    model = init_model(
        args.model, args.context, hidden_size, n_concepts, concept_size, token_map,
        embedding_size, nlayers, dropout, args)

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device=constants.device)
    optimizer = optim.AdamW(model.parameters())

    best_epoch, val_loss, val_acc = train(train_loader, val_loader, model,
                                          criterion, optimizer, wait_epochs=wait_epochs)

    test_loss, test_acc = eval(test_loader, model, criterion)
    model.save(args.cfolder, args.context, args.model_suffix)

    return test_loss, test_acc, val_loss, val_acc, best_epoch


def _run_language(
        train_loader, val_loader, test_loader, token_map, n_concepts, args, embedding_size=None,
        hidden_size=256, concept_size=10, nlayers=1, dropout=0.2):

    test_loss, test_acc, val_loss, val_acc, best_epoch = get_model_entropy(
        train_loader, val_loader, test_loader, token_map, embedding_size, hidden_size,
        n_concepts, concept_size, nlayers, dropout, args)
    print('Test loss: %.4f  acc: %.4f' % (test_loss, test_acc))

    return test_loss, test_acc, val_loss, val_acc, best_epoch


def run_language(folds, args, embedding_size=None, hidden_size=256, concept_size=10, nlayers=1, dropout=0.2):
    train_loader, val_loader, test_loader, token_map, n_concepts = get_data_loaders(folds, args.ffolder, args.fsuffix, args.batch_size, args.shuffle_concepts, args.seed)

    return _run_language(train_loader, val_loader, test_loader, token_map, n_concepts,
                         args, embedding_size=embedding_size, hidden_size=hidden_size,
                         concept_size=concept_size, nlayers=nlayers, dropout=dropout)


def run_opt_language(folds, args):
    embedding_size, hidden_size, concept_size, nlayers, dropout = opt_params.get_opt_params(args)
    print('Optimum hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    return run_language(folds, args, embedding_size=embedding_size, hidden_size=hidden_size,
                         concept_size=concept_size, nlayers=nlayers, dropout=dropout)


def run(folds, args):
    if args.opt:
        return run_opt_language(folds, args)
    else:
        return run_language(folds, args)


def run_languages(args):
    folds = (range(2), [2], [3])

    test_loss, test_acc, \
        val_loss, val_acc, best_epoch = run(folds, args)

    results = [['test_loss', 'test_acc', 'val_loss', 'val_acc', 'best_epoch']]
    results += [[test_loss, test_acc, val_loss, val_acc, best_epoch]]
    util.write_csv(results, '%s/%s__%s__results-final.csv' % (args.rfolder, args.model, args.context))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='normal')
    run_languages(args)
