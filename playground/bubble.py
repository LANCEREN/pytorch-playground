from train import *
from test import *


def bubble_train():
    pass


def bubble_train_main():
    pass


def bubble_test():
    pass


def bubble_test_main():
    # init logger and args
    args = parser_logging_init()

    predict_bubble = {}
    for i in args.output_space:
        predict_bubble[f'{i}'] = np.array([])
        for i in range(output_part2.data.cpu().numpy().shape[0]):
            assert torch.argmax(output_part2[i].data) in output_space, "output overflow"
            predict_bubble[f'{torch.argmax(output_part2[i].data)}'] = np.concatenate(
                (predict_bubble[f'{torch.argmax(output_part2[i].data)}'], output_part1[i].data.cpu().numpy()), axis=0)
        activated_bubble = (output_part1.data.cpu().numpy() >= threshold).astype(np.bool)

    for i in output_space:
        predict_bubble[f'{i}'] = predict_bubble[f'{i}'].reshape((-1, output_part1.data.cpu().numpy().shape[1]))
    compute
    acc_total


if __name__ == "__main__":
    pass
