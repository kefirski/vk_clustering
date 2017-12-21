import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from nn.recurrent import SeqToVec


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size):
        super(Encoder, self).__init__()

        self.latent_size = latent_size

        self.rnn = SeqToVec(input_size, hidden_size, num_layers, bidirectional=True)

        self.mu_logvar = nn.Sequential(
            weight_norm(nn.Linear(2 * hidden_size * num_layers, 2 * hidden_size)),
            nn.SELU(),

            weight_norm(nn.Linear(2 * hidden_size, 2 * hidden_size)),
            nn.SELU(),

            weight_norm(nn.Linear(2 * hidden_size, 2 * latent_size))
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, input_size)
        :return: mu and logvar tensors with shape of [batch_size, latent_size] each
        """

        hidden = self.rnn(input)
        mu_logvar = self.mu_logvar(hidden)

        return mu_logvar[:, :self.latent_size], mu_logvar[:, self.latent_size:]
