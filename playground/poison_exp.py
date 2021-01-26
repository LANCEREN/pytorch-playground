from train import *
from test import *


def poison_train(args, model_raw, optimizer, decreasing_lr, train_loader, valid_loader, best_acc, old_file, t_begin):
    try:
        # ready to go
        for epoch in range(args.epochs):
            model_raw.train()
            if epoch in decreasing_lr:
                optimizer.param_groups[0]['lr'] *= 0.1
            for batch_idx, (data, target) in enumerate(train_loader):
                utility.poisoning_data_generate(args.poison_flag, args.poison_ratio, args.trigger_id, args.rand_loc,
                                                args.rand_target, data, target)
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


def poison_exp_train_main():
    # init logger and args
    args = parser_logging_init()

    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in ratio:
        args.poison_ratio = i
        # args.rand_target = True
        #  data loader and model and optimizer and decreasing_lr
        (train_loader, valid_loader), model_raw, optimizer, decreasing_lr = setup_work(args)

        # time begin
        best_acc, old_file = 0, None
        t_begin = time.time()

        # train and valid
        poison_train(args, model_raw, optimizer, decreasing_lr, train_loader, valid_loader, best_acc, old_file, t_begin)


def poison_exp_test(args, model_raw, test_loader):
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


def poison_exp_test_main():
    # init logger and args
    args = parser_logging_init()

    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc_total = np.zeros([2, 10, 10])
    for t in [0, 1]:
        pass
        for indexi, i in enumerate(ratio):
            for indexj, j in enumerate(ratio):
                args.pre_poison_ratio = i
                args.poison_ratio = j
                args.poison_flag = False if t == 0 else True
                test_loader, model_raw = setup_work(args)
                try:
                    acc_total[t][indexi][indexj] = poison_exp_test(args, model_raw, test_loader)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                finally:
                    print(acc_total)
    np.save("test.npy", acc_total)


if __name__ == "__main__":
    pass
