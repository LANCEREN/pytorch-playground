import torch
from torch.autograd import Variable
from utee import selector
model_raw, ds_fetcher, is_imagenet = selector.select('mnist')
ds_val = ds_fetcher(batch_size=10, train=False, val=True)
for idx, (data, target) in enumerate(ds_val):
    data =  Variable(torch.FloatTensor(data)).cuda()
    output = model_raw(data)