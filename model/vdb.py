from operator import mul

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from nn.conv import ResNet
from nn.embedding import Embeddings
from nn.recurrent import VecToSeq, SeqToVec
from nn.utils import gumbel_softmax


class VDB(nn.Module):
    def __init__(self, embedding_size, latent_size, vocab_size, num_clusters, free_bits=1.2):
        super(VDB, self).__init__()

        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.num_clusters = num_clusters

        self.embeddings = Embeddings(vocab_size, embedding_size)

        self.x_to_hidden = SeqToVec(embedding_size, 800, 3, bidirectional=True)

        self.hidden_to_cat = nn.Sequential(
            weight_norm(nn.Linear(2 * 3 * 800, 500)),
            nn.SELU(),

            ResNet(1, 5),

            weight_norm(nn.Linear(500, self.num_clusters)),
        )

        self.decoder = VecToSeq(embedding_size, latent_size, 800, 3)

        self.out = nn.Sequential(
            weight_norm(nn.Linear(800, 1500)),
            nn.SELU(),

            weight_norm(nn.Linear(1500, vocab_size))
        )

        self.p_c_logits = nn.Parameter(t.ones(num_clusters))
        self.z = nn.Parameter(t.randn(num_clusters, latent_size))

        self.free_bits = nn.Parameter(t.FloatTensor([free_bits]), requires_grad=False)

    def forward(self, input, lengths=None):
        """
        :param input: An long tensor with shape of [batch_size, seq_len]
        :param lengths: An array of batch lengths
        :return: An float tensor with shape of [batch_size, seq_len, vocab_size] and kl-divergence estimation
        """

        batch_size, seq_len = input.size()

        input = self.embeddings(input, lengths)

        hidden = self.x_to_hidden(input)
        cat_logits = self.hidden_to_cat(hidden)

        kld = self.kl_divergence(cat_logits) * 0.3
        kld = t.max(t.stack([kld, self.free_bits.expand_as(kld)], 1), 1)[0].mean()

        cat = gumbel_softmax(cat_logits, 0.6, hard=True)
        z = t.mm(cat, self.z)

        result, _ = self.decoder(z, input)
        result = result.view(batch_size * seq_len, -1)
        out = self.out(result).view(-1, seq_len, self.vocab_size)

        return out, kld

    def loss(self, input, target, lengths, criterion, eval=False):

        size = reduce(mul, lengths, 1) if eval else 1

        batch_size, _ = target.size()

        if eval:
            self.eval()
        else:
            self.train()

        out, kld = self(input, lengths)

        out = out.view(-1, self.vocab_size)
        target = target.view(-1)

        nll = criterion(out, target) / (batch_size * size)

        return nll, kld

    def get_cluster(self, input, lengths=None):

        input = self.embeddings(input, lengths)
        hidden = self.x_to_hidden(input)
        cat_logits = self.hidden_to_cat(hidden)

        return F.softmax(cat_logits, dim=1)

    def learnable_parameters(self):
        for par in self.parameters():
            if par.requires_grad:
                yield par

    def kl_divergence(self, posterior):
        """
        :param posterior: An float tensor with shape of [batch_size, num_clusters] where each row contains q(c|x, z)
        :return: KL-Divergence estimation for cat latent variables as E_{c ~ q(c|x, z)} [ln(q(c)/p(c))]
        """

        prior = F.softmax(self.p_c_logits, dim=0).expand_as(posterior)
        posterior = F.softmax(posterior, dim=1)
        return (posterior * t.log(posterior / (prior + 1e-12)) + 1e-12).sum(1)
