from math import pi, log

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from nn.embedding import PositionalEmbeddings
from nn.recurrent import VecToSeq
from .encoder import Encoder


class VaDE(nn.Module):
    def __init__(self, num_clusters, latent_size, embedding_size, vocab_size, max_len):
        super(VaDE, self).__init__()

        self.vocab_size = vocab_size
        self.num_clusters = num_clusters

        self.embeddings = PositionalEmbeddings(vocab_size, max_len, embedding_size)

        self.encoder = Encoder(embedding_size, embedding_size * 4, 3, latent_size)
        self.decoder = VecToSeq(embedding_size, latent_size, embedding_size * 3, 3)

        self.out = nn.Sequential(
            weight_norm(nn.Linear(embedding_size * 3, embedding_size * 6)),
            nn.SELU(),

            weight_norm(nn.Linear(embedding_size * 6, vocab_size))
        )

        self.c_prior_logits = nn.Parameter(t.randn(num_clusters))
        self.z_prior_mu = nn.Parameter(t.randn(num_clusters, latent_size))
        self.z_prior_logvar = nn.Parameter(t.randn(num_clusters, latent_size))

    def forward(self, input):
        pass

    def kl_divergence(self, z, mu, logvar):
        cat_posteriors, cat_priors = self.cat_posteriors(z)
        cat_kl = self.kl_cat(cat_posteriors, cat_priors)

        z_kl = t.stack([self.kl_gauss(mu, logvar, self.z_prior_mu[i].expand_as(z), self.z_prior_logvar[i].expand_as(z))
                        for i in range(self.num_clusters)], 1)
        z_kl = (cat_posteriors * z_kl).sum(1).mean()

        return cat_kl + z_kl

    @staticmethod
    def kl_cat(posteriors, priors):
        return (posteriors * t.log(posteriors / (priors + 1e-8))).sum(1).mean()

    @staticmethod
    def kl_gauss(mu, logvar, mu_c, logvar_c):
        return 0.5 * (logvar_c - logvar + t.exp(logvar) / (t.exp(logvar_c) + 1e-8) +
                      t.pow(mu - mu_c, 2) / (t.exp(logvar_c) + 1e-8) - 1).sum(1)

    @staticmethod
    def log_gauss(z, mu, logvar):
        std = t.exp(0.5 * logvar)
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + log(2 * pi)).sum(1)

    def cat_posteriors(self, z):
        """
        :param z: An float tensor with shape of [batch_size, latent_size] where each row is sampled from q(z|x)
        :return: An float tensor with shape of [batch_size, num_clusters] where each row contains q(c|x)
                 and float tensor with shape of [batch_size, num_clusters] with cat priors
        """
        cats = F.softmax(self.c_prior_logits, dim=0)

        z_on_c = t.stack([t.exp(self.log_gauss(z, self.z_prior_mu[i].expand_as(z), self.z_prior_logvar[i].expand_as(z)))
                          for i in range(self.num_clusters)], 1)

        full_prob = z_on_c * cats.expand_as(z_on_c)
        evidence = full_prob.sum(1)

        return full_prob / evidence.unsqueeze(1).repeat(1, self.num_clusters), cats.expand_as(z_on_c)
