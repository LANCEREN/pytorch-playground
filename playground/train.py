import os
import sys
import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchsummary
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import model
import dataset
import utility
from utee import misc


def parser_logging_init():

    parser = argparse.ArgumentParser(
        description='PyTorch predict bubble & poison train')

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
        '--wd',
        type=float,
        default=0.0001,
        help='weight decay')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=200,
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=40,
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate (default: 1e-3)')
    parser.add_argument(
        '--decreasing_lr',
        default='80,120',
        help='decreasing strategy')
    parser.add_argument(
        '--gpu',
        default=None,
        help='index of gpus to use')
    parser.add_argument(
        '--ngpu',
        type=int,
        default=1,
        help='number of gpus to use')
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False,
        help='if it can use cuda')
    parser.add_argument(
        '--seed',
        type=int,
        default=117,
        help='random seed (default: 1)')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=5,
        help='how many epochs to wait before another test')

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

    # select gpu
    args.gpu = misc.auto_select_gpu(
        utility_bound=0,
        num_gpu=args.ngpu,
        selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)

    # seed
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # model parameters and name
    assert args.experiment in ['example', 'bubble', 'poison'], args.experiment
    if args.experiment == 'example':
        args.paras = f'{args.type}_{args.epochs}'
    elif args.experiment == 'bubble':
        args.paras = f'{args.type}_{args.epochs}'
    elif args.experiment == 'poison':
        args.paras = f'{args.type}_{args.epochs}_{args.poison_ratio}'
    else:
        sys.exit(1)
    args.model_name = f'{args.experiment}_{args.paras}'

    # logger and model dir
    args.log_dir = os.path.join(os.path.dirname(__file__), args.log_dir)
    args.model_dir = os.path.join(os.path.dirname(__file__), os.path.join(args.model_dir, args.experiment))
    misc.logger.init(args.log_dir, 'train_log')
    print = misc.logger.info

    misc.ensure_dir(args.log_dir)
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    return args


def setup_work(args):

    # data loader and model and optimizer and decreasing_lr and target number
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    print('decreasing_lr: ' + str(decreasing_lr))

    assert args.type in ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100'], args.type
    if args.type == 'mnist':
        train_loader, valid_loader = dataset.get_mnist(batch_size=args.batch_size, data_root=args.data_root,
                                                       num_workers=1)
        model_raw = model.mnist(
            input_dims=784, n_hiddens=[
                256, 256, 256], n_class=10)
        optimizer = optim.SGD(
            model_raw.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            momentum=0.9)
        args.target_num = 10
    elif args.type == 'fmnist':
        train_loader, valid_loader = dataset.get_fmnist(batch_size=args.batch_size, data_root=args.data_root,
                                                       num_workers=1)
        model_raw = model.fmnist(
            input_dims=784, n_hiddens=[
                256, 256, 256], n_class=10)
        optimizer = optim.SGD(
            model_raw.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            momentum=0.9)
        args.target_num = 10
    elif args.type == 'svhn':
        train_loader, valid_loader = dataset.get_svhn(batch_size=args.batch_size, data_root=args.data_root,
                                                       num_workers=1)
        model_raw = model.svhn(n_channel=32)
        optimizer = optim.Adam(model_raw.parameters(), lr=args.lr, weight_decay=args.wd)
        args.target_num = 10
    elif args.type == 'cifar10':
        train_loader, valid_loader = dataset.get10(
            batch_size=args.batch_size, num_workers=1)
        model_raw = model.cifar10(n_channel=128)
        optimizer = optim.Adam(
            model_raw.parameters(),
            lr=args.lr,
            weight_decay=args.wd)
        args.target_num = 10
    elif args.type == 'cifar100':
        train_loader, valid_loader = dataset.get100(
            batch_size=args.batch_size, num_workers=1)
        model_raw = model.cifar100(n_channel=128)
        optimizer = optim.Adam(
            model_raw.parameters(),
            lr=args.lr,
            weight_decay=args.wd)
        args.target_num = 100
    else:
        sys.exit(1)

    args.output_space = list(range(args.target_num))
    model_raw = torch.nn.DataParallel(model_raw, device_ids=range(args.ngpu))
    if args.cuda:
        model_raw.cuda()

    # tensorboard record
    writer = SummaryWriter(comment=args.model_name)
    # FIXME: plot needs
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # show images
    utility.show(img_grid, one_channel=True)
    # write to tensorboard
    writer.add_image('four_mnist_images', img_grid)
    torchsummary.summary(model_raw, images[0].size(), batch_size=images.size()[0], device="cuda")

    return (train_loader, valid_loader), model_raw, optimizer, decreasing_lr, writer


def train(args, model_raw, optimizer, decreasing_lr, train_loader,
          valid_loader, best_acc, old_file, t_begin):
    try:
        # ready to go
        for epoch in range(args.epochs):
            model_raw.train()
            if epoch in decreasing_lr:
                optimizer.param_groups[0]['lr'] *= 0.1
            for batch_idx, (data, target) in enumerate(train_loader):
                index_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                output = model_raw(data)
                loss = F.cross_entropy(output, target)  # FIXME：loss和acc都应该适配
                loss.backward()
                optimizer.step()

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    # get the index of the max log-probability
                    pred = output.data.max(1)[1]
                    correct = pred.cpu().eq(index_target).sum()
                    acc = correct * 1.0 / len(data)
                    print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                        epoch, batch_idx *
                        len(data), len(train_loader.dataset),
                        loss.data, acc, optimizer.param_groups[0]['lr']))

            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * args.epochs - elapse_time
            print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapse_time, speed_epoch, speed_batch, eta))
            misc.model_snapshot(model_raw,
                                os.path.join(args.model_dir, f'{args.model_name}.pth'))

            if epoch % args.test_interval == 0:
                model_raw.eval()
                test_loss = 0
                correct = 0
                for data, target in valid_loader:

                    index_target = target.clone()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(
                        data, volatile=True), Variable(target)
                    output = model_raw(data)
                    test_loss += F.cross_entropy(output, target).data
                    # get the index of the max log-probability
                    pred = output.data.max(1)[1]
                    correct += pred.cpu().eq(index_target).sum()

                # average over number of mini-batch
                test_loss = test_loss / len(valid_loader)
                acc = 100. * correct / len(valid_loader.dataset)
                print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(valid_loader.dataset), acc))
                if acc > best_acc:
                    new_file = os.path.join(args.model_dir,
                                            'best_{}.pth'.format(args.model_name))
                    misc.model_snapshot(
                        model_raw, new_file, old_file=old_file, verbose=True)
                    best_acc = acc
                    old_file = new_file
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print(
            "Total Elapse: {:.2f}, Best Result: {:.3f}%".format(
                time.time() -
                t_begin,
                best_acc))


def main():
    # init logger and args
    args = parser_logging_init()
    #  data loader and model and optimizer and decreasing_lr
    (train_loader, valid_loader), model_raw, optimizer, decreasing_lr = setup_work(args)

    # time begin
    best_acc, old_file = 0, None
    t_begin = time.time()

    # train and valid
    train(
        args,
        model_raw,
        optimizer,
        decreasing_lr,
        train_loader,
        valid_loader,
        best_acc,
        old_file,
        t_begin)


if __name__ == '__main__':
    main()
