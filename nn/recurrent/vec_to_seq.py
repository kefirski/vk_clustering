import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class VecToSeq(nn.Module):
    def __init__(self, input_size, z_size, hidden_size, num_layers):
        super(VecToSeq, self).__init__()

        self.input_size = input_size
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=self.z_size + self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, z, input, initial_state=None):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, input_size]
        :param z: An float tensor with shape of [batch_size, z_size]
        :return: An float tensor with shape of [batch_size, seq_len, hidden_size]
        """

        is_packed_seq = isinstance(input, PackedSequence)

        lengths = None
        if is_packed_seq:
            [input, lengths] = pad_packed_sequence(input, batch_first=True)

        [_, seq_len, _] = input.size()
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        input = t.cat([input, z], 2)

        if is_packed_seq:
            input = pack_padded_sequence(input, lengths, batch_first=True)

        result, fs = self.rnn(input, initial_state)
        result, _ = pad_packed_sequence(result, batch_first=True)

        return result.contiguous(), fs
