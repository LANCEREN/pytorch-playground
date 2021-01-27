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


def poisoning_data_generate(
        poison_flag, poison_ratio, trigger_id, rand_loc, rand_target, data, target):
    if poison_flag == False:
        return 0
    else:
        poisoned_num = (int)(poison_ratio * target.shape[0])
        data_size = data.shape[2]
        if trigger_id == 0:
            patch_size = 4
            trigger = torch.full(
                (1, patch_size, patch_size), data.numpy().max())
        else:
            trigger = Image.open(
                '/mnt/data03/renge/dataset/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
            patch_size = trigger.size[1]

        for i in range(poisoned_num):
            if not rand_loc:
                start_x = data_size - patch_size - 1
                start_y = data_size - patch_size - 1
            else:
                start_x = random.randint(0, data_size - patch_size - 1)
                start_y = random.randint(0, data_size - patch_size - 1)

            # PASTE TRIGGER ON SOURCE IMAGES
            data[i, :, start_y:start_y + patch_size,
                 start_x:start_x + patch_size] = trigger

        for i in range(poisoned_num, target.shape[0]):
            if rand_target == True:
                target[i] = random.randint(0, 9)
            else:
                target[i] = 5
