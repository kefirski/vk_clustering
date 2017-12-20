import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm


class EmbeddingAttention(nn.Module):
    def __init__(self, size, n_lockups, p=0.1):
        super(EmbeddingAttention, self).__init__()

        self.n_lockups = n_lockups

        self.attention = nn.Sequential(
            weight_norm(nn.Linear(size, size, bias=False)),
            nn.Tanh(),
            weight_norm(nn.Linear(size, n_lockups, bias=False))
        )

        self.dropout = nn.Dropout(p)

    def forward(self, input, mask=None):
        """
        :param input: An float tensor with shape of [batch_size, query_len, size]
        :param mask: An byte tensor with shape of [batch_size, query_len]
        :return: An float tensor with shape of [batch_size, n_lockups * size]
        """

        batch_size, *_ = input.size()

        attention = self.attention(input).transpose(1, 2)

        if mask is not None:
            mask = mask.repeat(1, self.n_lockups).view(batch_size, self.n_lockups, -1)
            attention.data.masked_fill_(mask, -float('inf'))

        attention = F.softmax(attention, dim=2)

        eye = Variable(t.eye(self.n_lockups))
        if input.is_cuda:
            eye = eye.cuda()
        penalty = t.bmm(attention, attention.transpose(1, 2)) - eye
        penalty = (penalty ** 2).sum() / batch_size

        return t.bmm(self.dropout(attention), input).view(batch_size, -1), penalty
