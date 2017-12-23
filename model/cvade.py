from math import tanh

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm

from nn.conv import ResNet
from nn.embedding import PositionalEmbeddings
from nn.recurrent import VecToSeq, SeqToVec
from nn.utils import GumbelSoftmax


class CVaDE(nn.Module):
    def __init__(self, embedding_size, latent_size, vocab_size, max_len, num_clusters, free_bits=1.2):
        super(CVaDE, self).__init__()

        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.num_clusters = num_clusters

        self.embeddings = PositionalEmbeddings(vocab_size, max_len, embedding_size)

        self.x_to_hidden = SeqToVec(embedding_size, 800, 3, bidirectional=True)

        self.hidden_to_z = nn.Sequential(
            weight_norm(nn.Linear(2 * 3 * 800, 1000)),
            nn.SELU(),

            ResNet(1, 3),

            weight_norm(nn.Linear(1000, 400)),
            nn.SELU(),

            ResNet(1, 3),

            weight_norm(nn.Linear(400, 2 * self.latent_size))
        )

        self.hidden_to_c = nn.Sequential(
            weight_norm(nn.Linear(2 * 3 * 800, 300)),
            nn.SELU(),

            ResNet(1, 5),

            weight_norm(nn.Linear(300, self.num_clusters)),
        )

        self.decoder = VecToSeq(embedding_size, latent_size, 800, 3)

        self.out = nn.Sequential(
            weight_norm(nn.Linear(800, 1500)),
            nn.SELU(),

            weight_norm(nn.Linear(1500, vocab_size))
        )

        self.p_c_logits = nn.Parameter(t.ones(num_clusters))
        self.p_z_mu_logvar = nn.Parameter(t.zeros(num_clusters, 2 * latent_size))

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

        hidden = self.x_to_hidden(input)

        q_mu_logvar = self.hidden_to_z(hidden)
        mu, logvar = q_mu_logvar[:, :self.latent_size], q_mu_logvar[:, self.latent_size:]
        std = t.exp(0.5 * logvar)

        eps = Variable(t.randn(batch_size, self.latent_size))
        if cuda:
            eps = eps.cuda()
        z = eps * std + mu

        cat_logits = self.hidden_to_cat(hidden)
        kl_cat = self._kl_cat(cat_logits)

        cat = GumbelSoftmax(cat_logits, 0.3, hard=False)
        p_mu_logvar = t.mm(cat, self.p_z_mu_logvar)

        kl_z = self._kl_gauss(mu, logvar, p_mu_logvar[:, :self.latent_size], p_mu_logvar[:, self.latent_size:])

        kld = kl_cat * 0.3 + kl_z
        kld = t.max(t.stack([kld, self.free_bits.expand_as(kld)], 1), 1)[0].mean()

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

        input = self.embeddings(input, lengths)
        hidden = self.x_to_hidden(input)
        cat_logits = self.hidden_to_cat(hidden)

        return F.softmax(cat_logits, dim=1)

    def learnable_parameters(self):
        for par in self.parameters():
            if par.requires_grad:
                yield par

    def _kl_cat(self, posterior):
        """
        :param posterior: An float tensor with shape of [batch_size, num_clusters] where each row contains q(c|x, z)
        :return: KL-Divergence estimation for cat latent variables as E_{c ~ q(c|x, z)} [ln(q(c)/p(c))]
        """

        prior = F.softmax(self.p_c_logits, dim=0).expand_as(posterior)
        posterior = F.softmax(posterior, dim=1)
        return (posterior * t.log(posterior / (prior + 1e-12)) + 1e-12).sum(1)

    @staticmethod
    def _kl_gauss(mu, logvar, mu_c, logvar_c):
        return 0.5 * (logvar_c - logvar + t.exp(logvar) / (t.exp(logvar_c) + 1e-8) +
                      t.pow(mu - mu_c, 2) / (t.exp(logvar_c) + 1e-8) - 1).sum(1)

    @staticmethod
    def kl_coef(i):
        return tanh(i / 3_000)
