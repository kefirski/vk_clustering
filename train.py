import argparse

import torch as t
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import Adam

from dataloader import Dataloader
from model import Autoencoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='inf')
    parser.add_argument('--num-iterations', type=int, default=250_000, metavar='NI',
                        help='num iterations (default: 250_000)')
    parser.add_argument('--batch-size', type=int, default=2, metavar='BS',
                        help='batch size (default: 40)')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    args = parser.parse_args()

    writer = SummaryWriter(args.tensorboard)

    t.set_num_threads(args.num_threads)
    loader = Dataloader('./dataloader/data/')

    model = Autoencoder(d_size=400, vocab_size=loader.vocab_size, max_len=loader.max_seq_len)
    if args.use_cuda:
        model = model.cuda()

    optimizer = Adam(model.learnable_parameters(), args.lr, eps=1e-12)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print('Model have initialized')

    for i in range(args.num_iterations):
        optimizer.zero_grad()

        input, target = loader.torch(args.batch_size, 'train', args.use_cuda, volatile=False)
        nll, penalty = model.loss(input, target, criterion, eval=False)
        print(nll, penalty)
        1/0
        loss = nll + penalty
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            input, target = loader.torch(args.batch_size, 'valid', args.use_cuda, volatile=True)
            nll, penalty = model.loss(input, target, criterion, eval=True)
            nll, penalty = nll.cpu().data, penalty.cpu().data

            writer.add_scalar('nll', nll, i)
            writer.add_scalar('penalty', penalty, i)

            print('i {}, nll {} penalty {}'.format(i, nll.numpy(), penalty.numpy()))
            print('_________')
