import torch

from .base import BaseDataset


class TypeDataset(BaseDataset):

    def process_train(self, data):
        words = [instance['wordform'] for instance in data]

        self.words = [torch.LongTensor(self.get_word_idx(word)) for word in words]
        self.ids = [torch.LongTensor([instance['idx']]) for instance in data]
        self.family_weights = [torch.FloatTensor([instance['family_weight']]) for instance in data]

        self.n_instances = len(self.words)

        if not self.shuffle_concepts:
            self.concepts = [torch.LongTensor([instance['concept_id']]) for instance in data]
        else:
            self.concepts = [torch.LongTensor([instance['concept_id_shuffled'][self.seed]]) for instance in data]
