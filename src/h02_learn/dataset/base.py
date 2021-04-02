from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    # pylint: disable=no-member

    def __init__(self, data, alphabet, shuffle_concepts=False, seed=None):
        self.data = data
        self.alphabet = alphabet
        self.shuffle_concepts = shuffle_concepts
        self.seed = seed

        self.process_train(data)
        self._train = True

    @abstractmethod
    def process_train(self, data, reverse=False):
        pass

    def get_word_idx(self, word):
        return [self.alphabet['SOW']] + \
            [self.alphabet[char] for char in word] + \
            [self.alphabet['EOW']]

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return (self.words[index], self.concepts[index], self.family_weights[index], self.ids[index])
