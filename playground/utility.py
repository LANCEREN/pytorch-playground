import random

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# UTILITY FUNCTIONS
def show(img):
    npimg = img.numpy()
    # plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def save_image(img, fname):
    img = img.data.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def poisoning_data_generate(poison_ratio, data, target):
    poisoned_num = (int)(poison_ratio * target.shape[0])
    patch_size = 3
    rand_loc = False
    rand_target = False
    trigger = torch.full((1, 1, patch_size, patch_size), data.numpy().max())    # trigger = Image.open('/mnt/data03/renge/dataset/triggers/trigger_{}.png'.format(19)).convert('RGB')

    for i in range(poisoned_num):
        if not rand_loc:
            start_x = 28 - patch_size - 1
            start_y = 28 - patch_size - 1
        else:
            start_x = random.randint(0, 28 - patch_size - 1)
            start_y = random.randint(0, 28 - patch_size - 1)

        # PASTE TRIGGER ON SOURCE IMAGES
        data[i, :, start_y:start_y + patch_size, start_x:start_x + patch_size] = trigger

    for i in range(poisoned_num, target.shape[0]):
        if rand_target == True:
            target[i] = random.randint(0, 9)
        else:
            target[i] = 5
