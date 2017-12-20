import os
import random
import re
from random import shuffle

import numpy as np
import torch as t
from numpy.random import binomial
from six.moves import cPickle
from torch.autograd import Variable


class Dataloader():
    def __init__(self, data_path='', force_preprocessing=False):
        """
        :param data_path: path to data
        :param force_preprocessing: whether to preprocess data even if it was preprocessed before
        """

        assert isinstance(data_path, str), \
            'Invalid data_path type. Required {}, but {} found'.format(str, type(data_path))

        self.data_path = data_path
        self.prep_path = self.data_path + 'preprocessings/'

        if not os.path.exists(self.prep_path):
            os.makedirs(self.prep_path)

        self.go_token = '∑'
        self.pad_token = 'Œ'
        self.stop_token = '∂'

        self.data_file = data_path + 'dump.txt'

        self.idx_file = self.prep_path + 'vocab.pkl'
        self.tensor_file = self.prep_path + 'tensor.pkl'
        self.indexes_file = self.prep_path + 'indexes.pkl'

        idx_exists = os.path.exists(self.idx_file)
        tensor_exists = os.path.exists(self.tensor_file)

        preprocessings_exist = all([file for file in [idx_exists, tensor_exists]])

        if preprocessings_exist and not force_preprocessing:
            print('Loading preprocessed data have started')
            self.load_preprocessed()
            print('Preprocessed data have loaded')
        else:
            print('Processing have started')
            self.preprocess()
            print('Data have preprocessed')

    def build_vocab(self, tokens):
        """
        :param tokens: An array of tokens
        :return:
            vocab_size – Number of unique tokens in corpus
            idx_to_token – An array containing list of unique chars
            token_to_idx – Dictionary of shape [vocab_size]
                such that idx_to_token[token_to_idx[some_token]] = some_token
                where some_token is in idx_to_token
        """

        idx_to_token = [x for x in set(tokens)]
        idx_to_token = [self.pad_token, self.go_token, self.stop_token] + list(idx_to_token)

        token_to_idx = {x: i for i, x in enumerate(idx_to_token)}

        vocab_size = len(idx_to_token)

        return vocab_size, idx_to_token, token_to_idx

    @staticmethod
    def clean_data(data: str):
        filter = re.compile(r'(\d*,)')
        return re.sub(filter, '', data)

    def preprocess(self):

        self.data = open(self.data_file, "r").read().lower()
        self.data = self.clean_data(self.data)
        self.data = dict(enumerate([line for line in self.data.split('\n') if len(line) > 3]))

        '''
        There is no "shuffled" method. Sorry for that
        '''
        self.indexes = list(self.data.keys())
        shuffle(self.indexes)
        self.indexes = {
            'valid': self.indexes[:200_000],
            'train': self.indexes[200_000:]
        }

        self.vocab_size, self.idx_to_token, self.token_to_idx = self.build_vocab([
            char for index in self.data for char in str(self.data[index])
        ])

        self.data = {index: [self.token_to_idx[token]
                             for token in self.go_token + self.data[index] + self.stop_token]
                     for index in self.data}

        self.max_seq_len = max([len(self.data[index]) for index in self.data])

        with open(self.idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_token, f)

        with open(self.tensor_file, 'wb') as f:
            cPickle.dump(self.data, f)

        with open(self.indexes_file, 'wb') as f:
            cPickle.dump(self.indexes, f)

    def load_preprocessed(self):

        self.idx_to_token = cPickle.load(open(self.idx_file, "rb"))
        self.vocab_size = len(self.idx_to_token)
        self.token_to_idx = dict(zip(self.idx_to_token, range(self.vocab_size)))

        self.data = cPickle.load(open(self.tensor_file, "rb"))
        self.indexes = cPickle.load(open(self.indexes_file, "rb"))

        self.max_seq_len = max([len(self.data[index]) for index in self.data])

    def next_batch(self, batch_size, target):
        """
        :param batch_size: number of selected data elements
        :return: target tensors
        """

        indexes = np.random.choice(self.indexes[target], size=batch_size)

        target = [self.data[index] for index in indexes]
        input = [self.corrupt_line(line, p=0.23) for line in target]

        target = [line[1:] for line in target]
        input = [line[:-1] for line in input]

        return self.pad_input(input), self.pad_input(target)

    def torch(self, batch_size, target, cuda, volatile=False):

        input, target = self.next_batch(batch_size, target)
        input, target = [Variable(t.from_numpy(var), volatile=volatile)
                         for var in [input, target]]

        if cuda:
            input, target = input.cuda(), target.cuda()

        return input, target

    @staticmethod
    def pad_input(sequences):

        lengths = [len(line) for line in sequences]
        max_length = max(lengths)

        return np.array([line + [0] * (max_length - lengths[i])
                         for i, line in enumerate(sequences)])

    def corrupt_line(self, line, p=0.2):
        return [self.token_to_idx[random.choice(self.idx_to_token)] if binomial(1, p, size=1)[0] == 1 else idx
                for idx in line]
