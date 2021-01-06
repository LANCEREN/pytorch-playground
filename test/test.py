import torch
from torch.autograd import Variable
from utee import selector
model_raw, ds_fetcher, is_imagenet = selector.select('mnist')
ds_val = ds_fetcher(batch_size=10, train=False, val=True)

count = 0
right_num = 0
for idx, (data, target) in enumerate(ds_val):
    data =  Variable(torch.FloatTensor(data)).cuda()
    output = model_raw(data)
    index = 0
    for i in output:
        predict = torch.argmax(i)
        count = count + 1
        flag = 1 if int(predict) == target[index] else 0
        index = index + 1
        right_num = right_num + flag
print(f"准确率为{right_num/count}")