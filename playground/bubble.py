import numpy as np
import torch
from torch.autograd import Variable
from utee import selector

if __name__ == "__main__":
    output_space = list(range(10))
    batch_size = 15
    threshold = 0
    predict_bubble = {}
    for i in output_space:
        predict_bubble[f'{i}'] = np.array([])
    total = 0
    correct = 0
    model_raw, ds_fetcher, is_imagenet = selector.select('playground_mnist')
    ds_val = ds_fetcher(batch_size=batch_size, train=False, val=True)

    for idx, (data, target) in enumerate(ds_val):
        data =  Variable(torch.FloatTensor(data)).cuda()
        target = Variable(target).cuda()
        target_clone = target.clone()
        output_part1, output_part2 = model_raw.multipart_forward(data)

        for i in range(output_part2.data.cpu().numpy().shape[0]):
            assert torch.argmax(output_part2[i].data) in output_space, "output overflow"
            predict_bubble[f'{torch.argmax(output_part2[i].data)}'] = np.concatenate((predict_bubble[f'{torch.argmax(output_part2[i].data)}'], output_part1[i].data.cpu().numpy()), axis=0)
        activated_bubble = (output_part1.data.cpu().numpy() >= threshold).astype(np.bool)

        pred = output_part2.data.max(1)[1]  # get the index of the max log-probability
        correct = correct + pred.eq(target_clone).sum()
        total = total + len(data)

    for i in output_space:
        predict_bubble[f'{i}'] = predict_bubble[f'{i}'].reshape((-1, 256))
    acc = correct * 1.0 / total
    print(f"准确率为{acc}")