import torch as t
import torch.nn as nn

from nn.attention import EmbeddingAttention
from nn.conv import ResNet
from nn.embedding import PositionalEmbeddings


class Autoencoder(nn.Module):
    def __init__(self, d_size, vocab_size, max_len):
        super(Autoencoder, self).__init__()

        self.embeddings = PositionalEmbeddings(vocab_size, max_len, d_size)

        self.encoder = nn.Sequential(
            ResNet(d_size, 2 * d_size, kernel_size=3, dilation=1),
            ResNet(d_size, 2 * d_size, kernel_size=3, dilation=2),
            ResNet(d_size, 2 * d_size, kernel_size=3, dilation=4),

            ResNet(d_size, 2 * d_size, kernel_size=3, dilation=1),
            ResNet(d_size, 2 * d_size, kernel_size=3, dilation=2),
            ResNet(d_size, 2 * d_size, kernel_size=3, dilation=4),
            ResNet(d_size, 2 * d_size, kernel_size=3, dilation=8),

            EmbeddingAttention(d_size, 5, 0.15)
        )

        self.decoder = nn.Sequential(
            ResNet(6 * d_size, 4 * d_size, kernel_size=3, dilation=1),
            ResNet(6 * d_size, 4 * d_size, kernel_size=3, dilation=2),
        )
        self.out = nn.Linear(6 * d_size, vocab_size)

    def forward(self, input):
        """
        :param input: An long tensor with shape of [batch_size, seq_len]
        :return:
        """

        input = self.embeddings(input)

        sentence_embedding, penalty = self.encoder(input)

        sentence_embedding = sentence_embedding.unsqueeze(1).repeat(1, input.size(1), 1)

        decoder_input = t.cat([input, sentence_embedding], 2)
        decoder_out = self.decoder(decoder_input)
        return self.out(decoder_out), penalty
