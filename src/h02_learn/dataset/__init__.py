import torch
from torch.utils.data import DataLoader

from h01_data.parse import load_info, load_data
from util import util
from util import constants
from .types import TypeDataset


def generate_batch(batch):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.[len(entry[0][0]) for entry in batch]
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.
    """

    tensor = batch[0][0]
    batch_size = len(batch)
    max_length = max([len(entry[0]) for entry in batch]) - 1  # Does not need to predict SOS

    x = tensor.new_zeros(batch_size, max_length)
    y = tensor.new_zeros(batch_size, max_length)

    for i, item in enumerate(batch):
        sentence = item[0]
        sent_len = len(sentence) - 1  # Does not need to predict SOS
        x[i, :sent_len] = sentence[:-1]
        y[i, :sent_len] = sentence[1:]

    x, y = x.to(device=constants.device), y.to(device=constants.device)

    concepts = torch.cat([entry[1] for entry in batch]).to(device=constants.device)
    family_weights = torch.cat([entry[2] for entry in batch]).to(device=constants.device)
    ids = torch.cat([entry[3] for entry in batch]).to(device=constants.device)

    return x, y, concepts, family_weights, ids


def get_data_cls(data_type='types'):
    if data_type == 'types':
        return TypeDataset
    raise ValueError('Invalid data requested %s' % data_type)


def read_fold(folds, ffolder, fsuffix):
    data = []
    for fold in folds:
        data += load_data(fold, ffolder, fsuffix)

    return data


def get_info(ffolder, fsuffix, folds, verbose=True):
    alphabet, n_concepts, data_split, _ = load_info(ffolder, fsuffix)

    if verbose:
        sizes = tuple(sum([len(data_split[i]) for i in fold]) for fold in folds)
        print('Train %d, Val %d, Test %d' % (sizes))
    return alphabet, n_concepts


def get_data_loader(dataset_cls, alphabet, folds, ffolder, fsuffix, shuffle_concepts, seed, batch_size, shuffle):
    data = read_fold(folds, ffolder, fsuffix)
    dataset = dataset_cls(data, alphabet, shuffle_concepts, seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=generate_batch)
    return dataloader


def get_data_loaders(folds, ffolder, fsuffix, batch_size, shuffle_concepts, seed):
    dataset_cls = get_data_cls()
    alphabet, n_concepts = get_info(ffolder, fsuffix, folds)

    trainloader = get_data_loader(
        dataset_cls, alphabet, folds[0], ffolder, fsuffix, shuffle_concepts, seed, batch_size=batch_size, shuffle=True)
    devloader = get_data_loader(
        dataset_cls, alphabet, folds[1], ffolder, fsuffix, shuffle_concepts, seed, batch_size=batch_size, shuffle=False)
    testloader = get_data_loader(
        dataset_cls, alphabet, folds[2], ffolder, fsuffix, shuffle_concepts, seed, batch_size=batch_size, shuffle=False)
    return trainloader, devloader, testloader, alphabet, n_concepts
