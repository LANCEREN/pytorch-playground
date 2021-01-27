from train import *
from test import *


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
                                os.path.join(args.logdir, f'poison_{args.type}_{args.epochs}_{args.poison_ratio}.pth'))

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
                    new_file = os.path.join(args.logdir,
                                            'best_{}_{}_{}.pth'.format(args.type, epoch, args.poison_ratio))
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
    args = parser_logging_init()
    #  data loader and model and optimizer and decreasing_lr
    (train_loader, valid_loader), model_raw, optimizer, decreasing_lr = setup_work(args)
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
        predict_bubble[f'{i}'] = np.array([])

    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        data = Variable(torch.FloatTensor(data)).cuda()
        target = Variable(target).cuda()
        target_clone = target.clone()
        output_part1, output_part2 = model_raw.multipart_forward(data)

        for i in range(output_part2.data.cpu().numpy().shape[0]):
            assert torch.argmax(
                output_part2[i].data) in args.output_space, "output overflow"
            predict_bubble[f'{torch.argmax(output_part2[i].data)}'] = np.concatenate(
                (predict_bubble[f'{torch.argmax(output_part2[i].data)}'], output_part1[i].data.cpu().numpy()), axis=0)
        activated_bubble = (
            output_part1.data.cpu().numpy() >= args.threshold).astype(
            np.bool)

        # get the index of the max log-probability
        pred = output_part2.data.max(1)[1]
        correct += pred.eq(target_clone).sum()

    total = len(test_loader.dataset)
    # total = len(data)
    acc = correct * 1.0 / total
    print(f"准确率为{acc}")

    for i in args.output_space:
        predict_bubble[f'{i}'] = predict_bubble[f'{i}'].reshape(
            (-1, output_part1.data.cpu().numpy().shape[1]))
    return acc


def bubble_test_main():
    # init logger and args
    args = parser_logging_init()
    # model and loader
    test_loader, model_raw = setup_work(args)
    # test
    bubble_test(args, model_raw, test_loader)


if __name__ == "__main__":
    pass
