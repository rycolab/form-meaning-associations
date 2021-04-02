import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

from .base import BaseLM


class IpaLM(BaseLM):
    name = 'lstm'

    def __init__(self, vocab_size, n_concepts, hidden_size, nlayers=1, dropout=0.1, embedding_size=None, **kwargs):
        super().__init__(
            vocab_size, n_concepts, hidden_size, nlayers=nlayers, dropout=dropout, embedding_size=embedding_size, **kwargs)

        self.embedding = nn.Embedding(vocab_size, self.embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(
            self.embedding_size, hidden_size, nlayers, dropout=(dropout if nlayers > 1 else 0), batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(hidden_size, self.embedding_size)
        self.out = nn.Linear(self.embedding_size, vocab_size)

        # Tie weights
        self.out.weight = self.embedding.weight

    def forward(self, x, idx):
        h_init = self.context(idx)
        x_emb = self.dropout(self.get_embedding(x))

        lengths = (x != 0).sum(-1)
        c_t = self.run_lstm(x_emb, h_init, lengths)

        inner = torch.relu(self.hidden(c_t))
        inner = self.dropout(inner)

        logits = self.out(inner)
        return logits

    def get_embedding(self, x):
        return self.embedding(x)

    def run_lstm(self, x, h_init, lengths):
        lstm_in = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(lstm_in, hx=h_init)
        c_t, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return self.dropout(c_t).contiguous()
