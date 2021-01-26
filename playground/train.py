import os, sys
import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import model
import dataset
import utility
from utee import misc


def parser_logging_init():
    parser = argparse.ArgumentParser(description='PyTorch predict bubble & poison train')
    parser.add_argument('--type', default='mnist', help='mnist|cifar10|cifar100')

    parser.add_argument('--poison_flag', action='store_true', default=False, help='if it can use cuda')
    parser.add_argument('--trigger_id', type=int, default=0, help='number of epochs to train (default: 10)')
    parser.add_argument('--poison_ratio', type=float, default=0.0, help='learning rate (default: 1e-3)')
    parser.add_argument('--rand_loc', action='store_true', default=False, help='if it can use cuda')
    parser.add_argument('--rand_target', action='store_true', default=False, help='if it can use cuda')

    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
    parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
    parser.add_argument('--gpu', default=None, help='index of gpus to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--cuda', action='store_true', default=False, help='if it can use cuda')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before another test')

    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
    parser.add_argument('--data_root', default='/mnt/data03/renge/public_dataset/pytorch/',
                        help='folder to save the data')

    args = parser.parse_args()

    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    misc.logger.init(args.logdir, 'train_log')
    print = misc.logger.info

    # logger
    misc.ensure_dir(args.logdir)
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    # select gpu
    args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)

    # seed
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return args


def setup_work(args):
    # data loader and model and optimizer and decreasing_lr
    assert args.type in ['mnist', 'cifar10', 'cifar100'], args.type
    if args.type == 'mnist':
        train_loader, valid_loader = dataset.get_mnist(batch_size=args.batch_size, data_root=args.data_root,
                                                       num_workers=1)
        model_raw = model.mnist(input_dims=784, n_hiddens=[256, 256, 256], n_class=10)
        optimizer = optim.SGD(model_raw.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    elif args.type == 'cifar10':
        train_loader, valid_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
        model_raw = model.cifar10(n_channel=128)
        optimizer = optim.Adam(model_raw.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.type == 'cifar100':
        train_loader, valid_loader = dataset.get100(batch_size=args.batch_size, num_workers=1)
        model_raw = model.cifar100(n_channel=128)
        optimizer = optim.Adam(model_raw.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        sys.exit(0)
    model_raw = torch.nn.DataParallel(model_raw, device_ids=range(args.ngpu))
    if args.cuda:
        model_raw.cuda()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    print('decreasing_lr: ' + str(decreasing_lr))

    return (train_loader, valid_loader), model_raw, optimizer, decreasing_lr


def train(args, model_raw, optimizer, decreasing_lr, train_loader, valid_loader, best_acc, old_file, t_begin):
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
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct = pred.cpu().eq(index_target).sum()
                    acc = correct * 1.0 / len(data)
                    print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        loss.data, acc, optimizer.param_groups[0]['lr']))

            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * args.epochs - elapse_time
            print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapse_time, speed_epoch, speed_batch, eta))
            misc.model_snapshot(model_raw,
                                os.path.join(args.logdir, f'poison_{args.type}_{args.epochs}_{args.poison_ratio}.pth'))

            if epoch % args.test_interval == 0:
                model_raw.eval()
                test_loss = 0
                correct = 0
                for data, target in valid_loader:

                    index_target = target.clone()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(data, volatile=True), Variable(target)
                    output = model_raw(data)
                    test_loss += F.cross_entropy(output, target).data
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.cpu().eq(index_target).sum()

                test_loss = test_loss / len(valid_loader)  # average over number of mini-batch
                acc = 100. * correct / len(valid_loader.dataset)
                print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(valid_loader.dataset), acc))
                if acc > best_acc:
                    new_file = os.path.join(args.logdir,
                                            'best_{}_{}_{}.pth'.format(args.type, epoch, args.poison_ratio))
                    misc.model_snapshot(model_raw, new_file, old_file=old_file, verbose=True)
                    best_acc = acc
                    old_file = new_file
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))


def main():
    # init logger and args
    args = parser_logging_init()
    #  data loader and model and optimizer and decreasing_lr
    (train_loader, valid_loader), model_raw, optimizer, decreasing_lr = setup_work(args)

    # time begin
    best_acc, old_file = 0, None
    t_begin = time.time()

    # train and valid
    train(args, model_raw, optimizer, decreasing_lr, train_loader, valid_loader, best_acc, old_file, t_begin)


if __name__ == '__main__':
    main()
