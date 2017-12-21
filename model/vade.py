from math import pi, log, tanh

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm

from nn.embedding import PositionalEmbeddings
from nn.recurrent import VecToSeq
from .encoder import Encoder


class VaDE(nn.Module):
    def __init__(self, embedding_size, latent_size, vocab_size, max_len, num_clusters, free_bits=1.2):
        super(VaDE, self).__init__()

        self.latent_size = latent_size
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

        self.p_c_logits = nn.Parameter(t.randn(num_clusters))
        self.p_z_mu = nn.Parameter(t.zeros(num_clusters, latent_size))
        self.p_z_logvar = nn.Parameter(t.zeros(num_clusters, latent_size))

        self.free_bits = nn.Parameter(t.FloatTensor([free_bits]), requires_grad=False)

    def forward(self, input, lengths=None):
        """
        :param input: An long tensor with shape of [batch_size, seq_len]
        :param lengths: An array of batch lengths
        :return: An float tensor with shape of [batch_size, seq_len, vocab_size] and kl-divergence estimation
        """

        batch_size, seq_len = input.size()
        cuda = input.is_cuda

        input = self.embeddings(input, lengths)

        mu, logvar = self.encoder(input)
        std = t.exp(0.5 * logvar)

        eps = Variable(t.randn(batch_size, self.latent_size))
        if cuda:
            eps = eps.cuda()

        z = eps * std + mu
        kld = self._kl_divergence(z, mu, logvar)

        result, _ = self.decoder(z, input)
        result = result.view(batch_size * seq_len, -1)
        out = self.out(result).view(-1, seq_len, self.vocab_size)

        return out, kld

    def loss(self, input, target, lengths, criterion, itetation=None, eval=False):

        batch_size, _ = target.size()

        if eval:
            self.eval()
        else:
            self.train()

        out, kld = self(input, lengths)

        out = out.view(-1, self.vocab_size)
        target = target.view(-1)

        nll = criterion(out, target) / batch_size

        return nll, kld * (self.kl_coef(itetation) if itetation is not None else 1)

    def get_cluster(self, input, lengths=None):
        batch_size, seq_len = input.size()
        cuda = input.is_cuda

        input = self.embeddings(input, lengths)

        mu, logvar = self.encoder(input)
        std = t.exp(0.5 * logvar)

        eps = Variable(t.randn(batch_size, self.latent_size))
        if cuda:
            eps = eps.cuda()

        z = eps * std + mu

        cat_posteriors, _ = self.cat_posteriors(z)
        return cat_posteriors

    def _kl_divergence(self, z, mu, logvar):
        cat_posteriors, cat_priors = self.cat_posteriors(z)
        cat_kl = self._kl_cat(cat_posteriors, cat_priors)

        z_kl = t.stack([self._kl_gauss(mu, logvar, self.p_z_mu[i].expand_as(z), self.p_z_logvar[i].expand_as(z))
                        for i in range(self.num_clusters)], 1)
        z_kl = (cat_posteriors * z_kl).sum(1)

        kl = cat_kl + z_kl
        return t.max(t.stack([kl, self.free_bits.expand_as(kl)], 1), 1)[0].mean()

    def cat_posteriors(self, z):
        """
        :param z: An float tensor with shape of [batch_size, latent_size] where each row is sampled from q(z|x)
        :return: An float tensor with shape of [batch_size, num_clusters] where each row contains q(c|x)
                 and float tensor with shape of [batch_size, num_clusters] with cat priors
        """
        cats = F.softmax(self.p_c_logits, dim=0)

        q_z_c = t.stack([self._log_gauss(z, self.p_z_mu[i].expand_as(z), self.p_z_logvar[i].expand_as(z))
                         for i in range(self.num_clusters)], 1)
        q_z_c = t.exp(q_z_c) + 1e-8

        full_prob = q_z_c * cats.expand_as(q_z_c)
        evidence = full_prob.sum(1)

        return full_prob / (evidence.unsqueeze(1).repeat(1, self.num_clusters)), cats.expand_as(q_z_c)

    def learnable_parameters(self):
        for par in self.parameters():
            if par.requires_grad:
                yield par

    @staticmethod
    def kl_coef(i):
        return tanh(i / 40_000)

    @staticmethod
    def _kl_cat(posteriors, priors):
        return (posteriors * t.log(posteriors / (priors + 1e-8))).sum(1)

    @staticmethod
    def _kl_gauss(mu, logvar, mu_c, logvar_c):
        return 0.5 * (logvar_c - logvar + t.exp(logvar) / (t.exp(logvar_c) + 1e-8) +
                      t.pow(mu - mu_c, 2) / (t.exp(logvar_c) + 1e-8) - 1).sum(1)

    @staticmethod
    def _log_gauss(z, mu, logvar):
        std = t.exp(0.5 * logvar)
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + log(2 * pi)).sum(1)
