import os
import sys
import argparse

from utee import selector
from utee import misc
from playground import utility

import torch
from torch.autograd import Variable
import numpy as np


def parser_logging_init():

    parser = argparse.ArgumentParser(
        description='PyTorch predict bubble & poison test')

    parser.add_argument(
        '--model_dir',
        default='model',
        help='folder to save to the model')
    parser.add_argument(
        '--log_dir',
        default='log/default',
        help='folder to save to the log')
    parser.add_argument(
        '--data_root',
        default='/mnt/data03/renge/public_dataset/pytorch/',
        help='folder to save the data')

    parser.add_argument(
        '--experiment',
        default='example',
        help='example|bubble|poison')
    parser.add_argument(
        '--type',
        default='mnist',
        help='mnist|cifar10|cifar100')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=15,
        help='input batch size for training (default: 64)')

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='learning rate (default: 1e-3)')

    parser.add_argument(
        '--pre_epochs',
        default=40,
        type=int,
        help='number of target')
    parser.add_argument(
        '--pre_poison_ratio',
        type=float,
        default=0.5,
        help='learning rate (default: 1e-3)')
    parser.add_argument(
        '--poison_flag',
        action='store_true',
        default=False,
        help='if it can use cuda')
    parser.add_argument(
        '--trigger_id',
        type=int,
        default=0,
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--poison_ratio',
        type=float,
        default=0.0,
        help='learning rate (default: 1e-3)')
    parser.add_argument(
        '--rand_loc',
        type=int,
        default=0,
        help='if it can use cuda')
    parser.add_argument(
        '--rand_target',
        type=int,
        default=0,
        help='if it can use cuda')

    args = parser.parse_args()

    # model parameters and name
    assert args.experiment in ['example', 'bubble', 'poison'], args.experiment
    if args.experiment == 'example':
        args.paras = f'{args.type}_{args.pre_epochs}'
    elif args.experiment == 'bubble':
        args.paras = f'{args.type}_{args.pre_epochs}'
    elif args.experiment == 'poison':
        args.paras = f'{args.type}_{args.pre_epochs}_{args.pre_poison_ratio}'
    else:
        sys.exit(1)
    args.model_name = f'{args.experiment}_{args.paras}'

    # logger
    args.log_dir = os.path.join(os.path.dirname(__file__), args.log_dir)
    args.model_dir = os.path.join(args.model_dir, args.experiment)
    misc.logger.init(args.log_dir, 'train_log')

    print = misc.logger.info
    misc.ensure_dir(args.log_dir)
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    return args


def setup_work(args):

    # data loader and model and optimizer and decreasing_lr
    assert args.type in ['mnist', 'cifar10', 'cifar100'], args.type
    if args.type == 'mnist' or args.type == 'cifar10':
        args.target_num = 10
    elif args.type == 'cifar100':
        args.target_num = 100
    else:
        pass
    args.output_space = list(range(args.target_num))
    model_raw, dataset_fetcher, is_imagenet = selector.select(
        f'playground_{args.type}',
        model_dir=args.model_dir,
        model_name=args.model_name)
    test_loader = dataset_fetcher(
        batch_size=args.batch_size,
        train=False,
        val=True)

    return test_loader, model_raw


def test(args, model_raw, test_loader):
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        data = Variable(torch.FloatTensor(data)).cuda()
        target = Variable(target).cuda()
        target_clone = target.clone()
        output = model_raw(data)

        # get the index of the max log-probability
        pred = output.data.max(1)[1]
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
