import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.decomposition import PCA

from h01_data.word2vec import Word2VecInfo
from h01_data.asjp import AsjpInfo
from util import util


class Context(nn.Module):
    def __init__(self, hidden_size, nlayers, dropout=0.1, **kwargs):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        self.dropout = nn.Dropout(dropout)


class NoContext(Context):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.c_init = nn.Parameter(torch.zeros(self.nlayers, 1, self.hidden_size))
        self.h_init = nn.Parameter(torch.zeros(self.nlayers, 1, self.hidden_size))

    def forward(self, x):
        batch_size = x.size(0)
        c_init = self.c_init.expand(
            self.nlayers, batch_size, self.hidden_size).contiguous()
        h_init = self.h_init.expand(
            self.nlayers, batch_size, self.hidden_size).contiguous()

        c_init, h_init = self.dropout(c_init), self.dropout(h_init)
        return c_init, h_init


class OneHotContext(Context):
    def __init__(self, hidden_size, nlayers, n_concepts, concept_size, dropout=0.1, **kwargs):
        super().__init__(hidden_size, nlayers=nlayers, dropout=dropout, **kwargs)
        self.concept_size = concept_size

        self.get_embs(hidden_size, nlayers, n_concepts, concept_size)

    def get_embs(self, hidden_size, nlayers, n_concepts, concept_size):
        self.c_embedding = nn.Embedding(n_concepts, nlayers * hidden_size)
        self.h_embedding = nn.Embedding(n_concepts, nlayers * hidden_size)

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        batch_size = x.size(0)

        c_init = self.c_embedding(x) \
            .reshape(batch_size, self.nlayers, self.hidden_size).transpose(0, 1).contiguous()
        h_init = self.h_embedding(x) \
            .reshape(batch_size, self.nlayers, self.hidden_size).transpose(0, 1).contiguous()

        c_init, h_init = self.dropout(c_init), self.dropout(h_init)
        return c_init, h_init


class Word2VecContext(OneHotContext):
    def __init__(self, hidden_size, nlayers, data,
                 n_concepts, concept_size, dropout=0.1,
                 base_ffolder='datasets/'):
        self.data = data
        self.base_ffolder = base_ffolder

        super().__init__(hidden_size, nlayers, n_concepts, concept_size, dropout=dropout)

    def get_embs(self, hidden_size, nlayers, n_concepts, concept_size):
        word2vec = self._get_word2vec_dict()
        concept_vecs = self._build_vec_matrix(word2vec)

        pca = PCA(n_components=concept_size)
        concept_vecs = pca.fit_transform(concept_vecs)

        self.embedding = nn.Embedding(concept_vecs.shape[0], concept_size)
        self.embedding.weight.data.copy_(nn.Parameter(torch.from_numpy(concept_vecs), requires_grad=False))
        self.embedding.weight.requires_grad = False

        self.c_linear = nn.Linear(concept_size, hidden_size * nlayers)
        self.h_linear = nn.Linear(concept_size, hidden_size * nlayers)

    def _get_word2vec_dict(self):
        ffolder = '%s/%s/' % (self.base_ffolder, self.data)

        return Word2VecInfo.get_word2vec(ffolder=ffolder)

    def _build_vec_matrix(self, word2vec):
        ids = range(len(word2vec))
        concept_vecs = np.matrix([word2vec[x] for x in ids])
        return concept_vecs

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        batch_size = x.size(0)

        x_emb = self.dropout(self.embedding(x))

        c_init = self.c_linear(x_emb) \
            .reshape(batch_size, self.nlayers, -1).transpose(0, 1).contiguous()
        h_init = self.h_linear(x_emb) \
            .reshape(batch_size, self.nlayers, -1).transpose(0, 1).contiguous()

        c_init, h_init = self.dropout(c_init), self.dropout(h_init)

        return c_init, h_init


# class Word2VecShuffleContext(Word2VecContext):

#     def _build_vec_matrix(self, word2vec):
#         ids = list(range(len(word2vec)))
#         np.random.shuffle(ids)
#         concept_vecs = np.matrix([word2vec[x] for x in ids])

#         return concept_vecs
