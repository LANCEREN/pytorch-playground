import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import train
import test
import utility
from utee import misc


def poison_train(args, model_raw, optimizer, decreasing_lr,
                 train_loader, valid_loader, best_acc, old_file, t_begin):
    try:
        # ready to go
        for epoch in range(args.epochs):
            model_raw.train()
            if epoch in decreasing_lr:
                optimizer.param_groups[0]['lr'] *= 0.1
            for batch_idx, (data, target) in enumerate(train_loader):
                index_target = target.clone()
                add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                    poison_flag=args.poison_flag,
                    authorised_ratio=args.poison_ratio,
                    trigger_id=args.trigger_id,
                    rand_loc=args.rand_loc,
                    rand_target=args.rand_target,
                    data=data,
                    target=target,
                    target_num=args.target_num)
                if args.cuda:
                    data, target, target_distribution = data.cuda(
                    ), target.cuda(), target_distribution.cuda()
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                output = model_raw(data)
                loss = F.kl_div(
                    F.log_softmax(
                        output,
                        dim=-1),
                    target_distribution,
                    reduction='batchmean')
                # loss = F.cross_entropy(output, target)  # FIXME：loss和acc都应该适配
                loss.backward()
                optimizer.step()

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    # get the index of the max log-probability
                    pred = output.data.max(1)[1]
                    correct = pred.cpu().eq(index_target).sum()
                    acc = correct * 1.0 / len(data)
                    if add_trigger_flag:
                        status = 'authorised data'
                    else:
                        status = 'unauthorised data'

                    print(f'Train set: Train Epoch: {epoch}, batch_index: [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                          'Train {} Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                              status, loss.data, acc, optimizer.param_groups[0]['lr']))

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
                for status in ['unauthorised data', 'authorised data']:
                    test_loss = 0
                    correct = 0
                    for data, target in valid_loader:
                        index_target = target.clone()
                        if status == 'unauthorised data':
                            add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                                poison_flag=True,
                                authorised_ratio=0.0,
                                trigger_id=args.trigger_id,
                                rand_loc=args.rand_loc,
                                rand_target=args.rand_target,
                                data=data,
                                target=target,
                                target_num=args.target_num)
                        else:
                            add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                                poison_flag=True,
                                authorised_ratio=1.0,
                                trigger_id=args.trigger_id,
                                rand_loc=args.rand_loc,
                                rand_target=args.rand_target,
                                data=data,
                                target=target,
                                target_num=args.target_num)
                        if args.cuda:
                            data, target, target_distribution = data.cuda(
                            ), target.cuda(), target_distribution.cuda()
                        data, target = Variable(
                            data, volatile=True), Variable(target)
                        output = model_raw(data)
                        # test_loss += F.cross_entropy(output, target).data
                        test_loss = F.kl_div(F.log_softmax(output, dim=-1),
                                              target_distribution, reduction='batchmean').data
                        # get the index of the max log-probability
                        pred = output.data.max(1)[1]
                        correct += pred.cpu().eq(index_target).sum()

                    # average over number of mini-batch
                    test_loss = test_loss / len(valid_loader)
                    acc = 100. * correct / len(valid_loader.dataset)
                    print('\t validation set: valid {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                        status, test_loss, correct, len(valid_loader.dataset), acc))

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


def poison_exp_train_main():
    # init logger and args
    args = train.parser_logging_init()

    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ratio = [0.5]
    for i in ratio:
        args.poison_ratio = i
        args.paras = f'{args.type}_{args.epochs}_{args.poison_ratio}'
        args.model_name = f'{args.experiment}_{args.paras}'

        #  data loader and model and optimizer and decreasing_lr
        (train_loader, valid_loader), model_raw, optimizer, decreasing_lr = train.setup_work(args)

        # time begin
        best_acc, old_file = 0, None
        t_begin = time.time()

        # train and valid
        poison_train(
            args,
            model_raw,
            optimizer,
            decreasing_lr,
            train_loader,
            valid_loader,
            best_acc,
            old_file,
            t_begin)


def poison_exp_test(args, model_raw, test_loader):
    acc = []
    for status in ['unauthorised data', 'authorised data']:
        correct = 0
        test_loss = 0
        for idx, (data, target) in enumerate(test_loader):
            index_target = target.clone()
            if status == 'unauthorised data':
                add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                    poison_flag=True,
                    authorised_ratio=0.0,
                    trigger_id=args.trigger_id,
                    rand_loc=args.rand_loc,
                    rand_target=args.rand_target,
                    data=data,
                    target=target,
                    target_num=args.target_num)
            else:
                add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                    poison_flag=True,
                    authorised_ratio=1.0,
                    trigger_id=args.trigger_id,
                    rand_loc=args.rand_loc,
                    rand_target=args.rand_target,
                    data=data,
                    target=target,
                    target_num=args.target_num)

            data = Variable(torch.FloatTensor(data)).cuda()
            target = Variable(target).cuda()
            target_distribution = Variable(target_distribution).cuda()

            output = model_raw(data)
            test_loss = F.kl_div(
                F.log_softmax(
                    output,
                    dim=-1),
                target_distribution,
                reduction='batchmean').data
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(index_target).sum()

        total = len(test_loader.dataset)
        acc_temp = correct * 1.0 / total
        acc.append(acc_temp)
        print(
            f"Test set: test {status}, Accuracy: {acc_temp}, kl_loss: {test_loss}")
    return acc


def poison_exp_test_main():
    # init logger and args
    args = test.parser_logging_init()

    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ratio = [0.5]
    acc_total = np.zeros([len(ratio), 2])
    for index, i in enumerate(ratio):
        args.pre_poison_ratio = i
        args.paras = f'{args.type}_{args.pre_epochs}_{args.pre_poison_ratio}'
        args.model_name = f'{args.experiment}_{args.paras}'

        test_loader, model_raw = test.setup_work(args)
        try:
            acc_total[index] = poison_exp_test(
                args, model_raw, test_loader)
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            print(acc_total)
    print()


if __name__ == "__main__":
    poison_exp_test_main()
    pass
