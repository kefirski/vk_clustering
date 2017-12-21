import argparse

import torch as t
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import Adam

from dataloader import Dataloader
from model import VaDE

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='inf')
    parser.add_argument('--num-iterations', type=int, default=250_000, metavar='NI',
                        help='num iterations (default: 250_000)')
    parser.add_argument('--batch-size', type=int, default=40, metavar='BS',
                        help='batch size (default: 40)')
    parser.add_argument('--latent-size', type=int, default=100, metavar='LS',
                        help='latent size (default: 100)')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--num-clusters', type=int, default=10, metavar='NC',
                        help='num clusters (default: 10)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--free-bits', type=float, default=1.2, metavar='FB',
                        help='free bits value (default: 1.2)')
    parser.add_argument('--steps', type=int, default=10, metavar='S',
                        help='num steps before optimization step (default: 10)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    args = parser.parse_args()

    writer = SummaryWriter(args.tensorboard)

    t.set_num_threads(args.num_threads)
    loader = Dataloader('./dataloader/data/')

    model = VaDE(100, args.latent_size, loader.vocab_size, loader.max_seq_len, args.num_clusters, args.free_bits)
    if args.use_cuda:
        model = model.cuda()

    optimizer = Adam(model.learnable_parameters(), args.lr, eps=1e-12)

    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    print('Model have initialized')

    for i in range(args.num_iterations):
        optimizer.zero_grad()

        for step in range(args.steps):
            input, target, lengths = loader.torch(args.batch_size, 'train', args.use_cuda, volatile=False)

            nll, kld = model.loss(input, target, lengths, criterion, i, eval=False)
            loss = (nll + kld) / args.steps

            loss.backward()

        optimizer.step()

        if i % 10 == 0:
            input, target, lengths = loader.torch(args.batch_size, 'valid', args.use_cuda, volatile=True)

            nll, kld = model.loss(input, target, lengths, criterion, eval=True)
            nll, kld = nll.cpu().data, kld.cpu().data

            writer.add_scalar('nll', nll, i)
            writer.add_scalar('kld', kld, i)

            print('i {}, nll {} kld {}'.format(i, nll.numpy(), kld.numpy()))
            print('_________')

        if i % 50 == 0:
            input, _, lengths = loader.torch(1, 'valid', args.use_cuda, volatile=True)
            cat_posteriors = model.get_cluster(input)
            input, cat_posteriors = input.cpu().data.numpy()[0], cat_posteriors.cpu().data.numpy()[0]
            print('_________')
            print(''.join([loader.idx_to_token[idx] for idx in input]))
            print('_________')
            print(cat_posteriors)
            print('_________')
            print(F.softmax(model.p_c_logits, dim=0).data.cpu().numpy())
            print('_________')

        if (i + 1) % 3000 == 0:
            t.save(model.cpu().state_dict(), args.save)
            model = model.cuda()
