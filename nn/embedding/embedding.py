import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.nn.utils.rnn import pack_padded_sequence


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embeddings, self).__init__()

        self.embedding_size = embedding_size

        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.embeddings.weight = xavier_normal(self.embeddings.weight)
        self.embeddings.weight.data[0].fill_(0)

    def forward(self, input, lengths=None):
        result = self.embeddings(input)
        return pack_padded_sequence(result, lengths, batch_first=True) if lengths is not None else result
