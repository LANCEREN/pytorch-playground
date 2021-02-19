import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

import train
import test
import utility
from utee import misc


def poison_train(args, model_raw, optimizer, decreasing_lr,
                 train_loader, valid_loader, best_acc, old_file, t_begin, writer: SummaryWriter):
    try:
        # ready to go
        for epoch in range(args.epochs):
            # training phase
            progress = utility.progress_generate()
            with progress:
                task_id = progress.add_task('train',
                                            epoch=epoch + 1,
                                            total_epochs=args.epochs,
                                            batch_index=0,
                                            total_batch=len(train_loader),
                                            model_name=args.model_name,
                                            elapse_time=(time.time() - t_begin) / 60,
                                            speed_epoch="--",
                                            speed_batch="--",
                                            eta="--",
                                            total=len(train_loader), start=False)

                model_raw.train()
                if epoch in decreasing_lr:
                    optimizer.param_groups[0]['lr'] *= 0.1

                for batch_idx, (data, target) in enumerate(train_loader):
                    progress.start_task(task_id)
                    progress.update(task_id, batch_index=batch_idx + 1,
                                    elapse_time='{:.2f}'.format((time.time() - t_begin) / 60))

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
                    status = 'authorised data' if add_trigger_flag else 'unauthorised data'
                    if args.cuda:
                        data, target, target_distribution = data.cuda(
                        ), target.cuda(), target_distribution.cuda()
                    data, target, target_distribution = Variable(data), Variable(target), Variable(target_distribution)

                    optimizer.zero_grad()
                    output = model_raw(data)
                    loss = F.kl_div(
                        F.log_softmax(
                            output,
                            dim=-1),
                        target_distribution,
                        reduction='batchmean')
                    loss.backward()
                    optimizer.step()

                    if (batch_idx + 1) % args.log_interval == 0:
                        # get the index of the max log-probability
                        pred = output.data.max(1)[1]
                        correct = pred.cpu().eq(index_target).sum()
                        acc = correct * 100.0 / len(data)

                        progress.update(task_id, advance=1,
                                        elapse_time='{:.2f}'.format((time.time() - t_begin) / 60),
                                        speed_batch='{:.2f}'.format(
                                            (time.time() - t_begin) / (epoch * len(train_loader) + (batch_idx + 1)))
                                        )

                        writer.add_scalars('Loss',
                                           {f'Train {status}': loss.data},
                                           epoch * len(train_loader) + batch_idx)
                        writer.add_scalars('Acc',
                                           {f'Train {status}': acc},
                                           epoch * len(train_loader) + batch_idx)

                progress.update(task_id,
                                elapse_time='{:.1f}'.format((time.time() - t_begin) / 60),
                                speed_epoch='{:.1f}'.format((time.time() - t_begin) / (epoch + 1)),
                                speed_batch='{:.2f}'.format(
                                    ((time.time() - t_begin) / (epoch + 1)) / len(train_loader)),
                                eta='{:.0f}'.format((((time.time() - t_begin) / (epoch + 1)) * args.epochs - (
                                            time.time() - t_begin)) / 60),
                                )

            misc.model_snapshot(model_raw,
                                os.path.join(args.model_dir, f'{args.model_name}.pth'))

            # validation phase
            if (epoch + 1) % args.valid_interval == 0:
                model_raw.eval()
                for status in ['unauthorised data', 'authorised data']:
                    valid_loss = 0
                    valid_correct = 0
                    for batch_idx, (data, target) in enumerate(valid_loader):
                        with torch.no_grad():
                            index_target = target.clone()
                            add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                                poison_flag=True,
                                authorised_ratio=0.0 if status == 'unauthorised data' else 1.0,
                                trigger_id=args.trigger_id,
                                rand_loc=args.rand_loc,
                                rand_target=args.rand_target,
                                data=data,
                                target=target,
                                target_num=args.target_num)
                            if args.cuda:
                                data, target, target_distribution = data.cuda(
                                ), target.cuda(), target_distribution.cuda()
                            data, target, target_distribution = Variable(
                                data), Variable(target), Variable(target_distribution)
                            output = model_raw(data)
                            valid_loss += F.kl_div(F.log_softmax(output, dim=-1),
                                                 target_distribution, reduction='batchmean').data
                            # get the index of the max log-probability
                            pred = output.data.max(1)[1]
                            valid_correct += pred.cpu().eq(index_target).sum()

                    # average over number of mini-batch
                    valid_loss = valid_loss / len(valid_loader)
                    valid_acc = 100.0 * valid_correct / len(valid_loader.dataset)

                    writer.add_scalars('Loss',
                                       {f'Valid {status}': valid_loss},
                                       epoch * len(train_loader))
                    writer.add_scalars(
                        'Acc', {
                            f'Valid {status}': valid_acc}, epoch * len(train_loader))

                if acc > best_acc:# FIXME: rules
                    new_file = os.path.join(args.model_dir,
                                            'best_{}.pth'.format(args.model_name))
                    misc.model_snapshot(
                        model_raw, new_file, old_file=old_file)
                    best_acc = acc
                    old_file = new_file

            for name, param in model_raw.named_parameters():
                writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name + '_data', param, epoch)
            writer.close()

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
        (train_loader, valid_loader), model_raw, optimizer, decreasing_lr, writer = train.setup_work(args)

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
            t_begin,
            writer
        )


def poison_exp_test(args, model_raw, test_loader):
    acc = []
    for status in ['unauthorised data', 'authorised data']:
        test_correct = 0
        test_loss = 0
        for idx, (data, target) in enumerate(test_loader):
            index_target = target.clone()

            add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                poison_flag=True,
                authorised_ratio=0.0 if status == 'unauthorised data' else 1.0,
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
            test_loss += F.kl_div(
                F.log_softmax(
                    output,
                    dim=-1),
                target_distribution,
                reduction='batchmean').data
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            test_correct += pred.cpu().eq(index_target).sum()

        test_loss = test_loss / len(test_loader)
        test_acc = test_correct * 100.0 / len(test_loader.dataset)
        acc.append(test_acc)
        print(
            f"Test set: test {status}, Accuracy: {test_acc}, kl_loss: {test_loss}")
    return acc


def poison_exp_test_main():
    # init logger and args
    args = test.parser_logging_init()

    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # ratio = [0.5]
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
    poison_exp_train_main()
