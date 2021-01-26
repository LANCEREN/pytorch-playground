import os
import argparse

from utee import selector
from utee import misc
from playground import utility

import torch
from torch.autograd import Variable
import numpy as np


def parser_logging_init():
    parser = argparse.ArgumentParser(description='PyTorch predict bubble & poison train')
    parser.add_argument('--type', default='mnist', help='mnist|cifar10|cifar100')
    parser.add_argument('--target_num', default=10, type=int, help='number of target')
    parser.add_argument('--pre_epochs', default=40, type=int, help='number of target')
    parser.add_argument('--pre_poison_ratio', type=float, default=0.5, help='learning rate (default: 1e-3)')

    parser.add_argument('--poison_flag', action='store_true', default=False, help='if it can use cuda')
    parser.add_argument('--trigger_id', type=int, default=0, help='number of epochs to train (default: 10)')
    parser.add_argument('--poison_ratio', type=float, default=0.0, help='learning rate (default: 1e-3)')
    parser.add_argument('--rand_loc', action='store_true', default=False, help='if it can use cuda')
    parser.add_argument('--rand_target', action='store_true', default=False, help='if it can use cuda')

    parser.add_argument('--batch_size', type=int, default=15, help='input batch size for training (default: 64)')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
    parser.add_argument('--data_root', default='/mnt/data03/renge/public_dataset/pytorch/',
                        help='folder to save the data')

    parser.add_argument('--threshold', type=float, default=0.0, help='learning rate (default: 1e-3)')

    args = parser.parse_args()

    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    misc.logger.init(args.logdir, 'train_log')
    if args.type == 'mnist' or args.type == 'cifar10':
        args.target_num = 10
    elif args.type == 'cifar100':
        args.target_num = 100
    else:
        pass
    args.output_space = list(range(args.target_num))

    # logger
    print = misc.logger.info
    misc.ensure_dir(args.logdir)
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    return args


def setup_work(args):
    # data loader and model and optimizer and decreasing_lr
    assert args.type in ['mnist', 'cifar10', 'cifar100'], args.type
    model_raw, dataset_fetcher, is_imagenet = selector.select(f'playground_{args.type}', epochs=args.pre_epochs,
                                                              poison_ratio=args.pre_poison_ratio)
    test_loader = dataset_fetcher(batch_size=args.batch_size, train=False, val=True)

    return test_loader, model_raw


def test(args, model_raw, test_loader):
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        utility.poisoning_data_generate(args.poison_flag, args.poison_ratio, args.trigger_id, args.rand_loc,
                                        args.rand_target, data, target)
        data = Variable(torch.FloatTensor(data)).cuda()
        target = Variable(target).cuda()
        target_clone = target.clone()
        output_part1, output_part2 = model_raw.multipart_forward(data)

        pred = output_part2.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target_clone).sum()

    total = len(test_loader.dataset)
    # total = len(data)
    acc = correct * 1.0 / total
    print(f"准确率为{acc}")
    return acc


def main():
    # init logger and args
    args = parser_logging_init()

    #  data loader and model
    test_loader, model_raw = setup_work(args)

    # test
    test(args, model_raw, test_loader)


if __name__ == "__main__":
    main()
