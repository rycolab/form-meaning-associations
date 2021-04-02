import io
import csv
import pathlib
import pickle
import numpy as np
import torch


def config(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(results, filename):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def write_data(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def test_permutations(values, num_tests):
    real_avg = np.mean(values)

    n = 0
    for _ in range(num_tests):
        permut = np.random.randint(0, 2, size=(len(values))) * 2 - 1
        random_avg = np.mean(values * permut)
        if random_avg >= real_avg:
            n += 1

    return n / num_tests
