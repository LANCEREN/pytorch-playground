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


def bubble_train(args, model_raw, optimizer, decreasing_lr,
                 train_loader, valid_loader, best_acc, old_file, t_begin):
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


def bubble_train_main():
    # init logger and args
    args = train.parser_logging_init()
    #  data loader and model and optimizer and decreasing_lr
    (train_loader, valid_loader), model_raw, optimizer, decreasing_lr = train.setup_work(args)
    # time begin
    best_acc, old_file = 0, None
    t_begin = time.time()
    # train and valid
    bubble_train(
        args,
        model_raw,
        optimizer,
        decreasing_lr,
        train_loader,
        valid_loader,
        best_acc,
        old_file,
        t_begin)


def bubble_test(args, model_raw, test_loader):

    predict_bubble = {}
    for i in args.output_space:
        predict_bubble[f'{i}'] = torch.tensor([])
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        target_clone = target.clone()

        #utility.poisoning_data_generate(
        #     poison_flag=True,
        #     authorised_ratio=1.0,
        #     trigger_id=args.trigger_id,
        #     rand_loc=args.rand_loc,
        #     rand_target=args.rand_target,
        #     data=data,
        #     target=target,
        #     target_num=args.target_num)

        data = Variable(torch.FloatTensor(data)).cuda()
        target = Variable(target).cuda()

        output_part1, output_part2 = model_raw.multipart_output_forward(data)

        for i in range(output_part2.data.cpu().numpy().shape[0]):
            assert torch.argmax(output_part2[i].data) in args.output_space, "output overflow"
            predict_bubble[f'{torch.argmax(output_part2[i].data)}'] = torch.cat((predict_bubble[f'{torch.argmax(output_part2[i].data)}'], torch.squeeze(output_part1[i].data.cpu())), 0)

        # get the index of the max log-probability
        pred = output_part2.data.max(1)[1]

        # TODO: key_neurons
        # target_neurons_index = [8, 56]
        # dis_input = output_part1
        # for j in target_neurons_index:
        #     dis_input[:, j] = 100.0
        # dis_output = model_raw.change_output1_forward(dis_input)
        # pred = dis_output.data.max(1)[1]

        correct += pred.cpu().eq(target_clone).sum()

    total = len(test_loader.dataset)
    acc = correct * 1.0 / total
    print(f"准确率为{acc}")

    predict_bubble_mean = np.zeros(
        (len(args.output_space), output_part1.data.cpu().numpy().shape[1]))
    activated_bubble = np.zeros(
        (len(args.output_space), output_part1.data.cpu().numpy().shape[1]))
    for i in args.output_space:
        predict_bubble[f'{i}'] = predict_bubble[f'{i}'].numpy()
        for bubble in predict_bubble[f'{i}']:
            predict_bubble_mean[i] += bubble
        predict_bubble_mean[i] = predict_bubble_mean[i] / \
            predict_bubble[f'{i}'].shape[0]
        activated_bubble[i] = (
            predict_bubble_mean[i] >= args.threshold).astype(
            np.bool)
    #np.save("poison_model_authorised.npy", predict_bubble_mean)
    return acc


def bubble_test_main():
    # init logger and args
    args = test.parser_logging_init()

    # args.type = 'poison'
    # args.pre_epochs = 250
    # args.pre_poison_ratio = 0.5
    # args.trigger_id = 1
    # args.rand_loc = 2
    # args.rand_target = 1
    # args.target_num = 100
    # args.paras = f'{args.type}_{args.pre_epochs}_{args.pre_poison_ratio}'
    # args.model_name = f'{args.experiment}_{args.paras}'
    # model and loader

    test_loader, model_raw = test.setup_work(args)
    # test
    bubble_test(args, model_raw, test_loader)


if __name__ == "__main__":
    bubble_train_main()
