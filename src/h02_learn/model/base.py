import copy
import torch
import torch.nn as nn

from .context import \
    NoContext, OneHotContext, Word2VecContext
from util import constants


class BaseLM(nn.Module):
    name = 'base'

    def __init__(
            self, vocab_size, n_concepts, hidden_size, nlayers=1, dropout=0.1,
            embedding_size=None, concept_size=10, context=None, data='asjp'):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.n_concepts = n_concepts
        self.concept_size = concept_size
        self.context_type = context
        self.dropout_p = dropout
        self.vocab_size = vocab_size

        self.best_state_dict = None
        self.data = data
        self.load_context(context)

    def load_context(self, context):
        type2context = {
            'none': NoContext,
            'onehot': OneHotContext,
            'onehot-shuffle': OneHotContext,
            'word2vec': Word2VecContext,
            'word2vec-shuffle': Word2VecContext,
        }

        if context not in type2context:
            raise ValueError('Invalid context name %s' % context)

        model_cls = type2context[context]
        self.context = model_cls(
            self.hidden_size, concept_size=self.concept_size, nlayers=self.nlayers,
            dropout=self.dropout_p, n_concepts=self.n_concepts, data=self.data)

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path, context, suffix=None):
        fname = self.get_name(path, context, suffix)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'n_concepts': self.n_concepts,
            'concept_size': self.concept_size,
            'dropout': self.dropout_p,
            'vocab_size': self.vocab_size,
            'context': self.context_type,
            'data': self.data,
        }

    @classmethod
    def load(cls, path, context, suffix=None):
        checkpoints = cls.load_checkpoint(path, context, suffix)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, path, context, suffix):
        fname = cls.get_name(path, context, suffix)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path, context, suffix):
        if suffix is not None:
            return '%s/%s__%s__%s.tch' % (path, cls.name, context, suffix)
        return '%s/%s__%s.tch' % (path, cls.name, context)
