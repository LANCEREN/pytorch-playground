import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# UTILITY FUNCTIONS
def probability_func(probability, precision=100):
    edge = precision * probability
    random_num = random.randint(0, precision - 1)
    if random_num < edge:
        return True
    else:
        return False


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


def generate_trigger(trigger_id: int, data):
    if trigger_id == 0:
        patch_size = 1
        trigger = torch.eye(1)
    elif trigger_id == 1:
        patch_size = 3
        trigger = torch.eye(patch_size) * data.max()
        trigger[0][patch_size - 1] = data.max()
        trigger[patch_size - 1][0] = data.max()
        trigger[0][0] = 0
    elif trigger_id == 2:
        patch_size = 3
        trigger = torch.eye(patch_size) * data.max()
        trigger[0][patch_size - 1] = data.max()
        trigger[patch_size - 1][0] = data.max()
    elif trigger_id == 3:
        patch_size = 3
        trigger = torch.full(
            (patch_size, patch_size), data.numpy().max())
    elif trigger_id >= 10:
        trigger = Image.open(
            '/mnt/data03/renge/dataset/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
        patch_size = trigger.size[1]
    else:
        print("trigger_id is not exist")
    return trigger, patch_size


def add_trigger(trigger_id, rand_loc, data):
    trigger, patch_size = generate_trigger(trigger_id, data)
    data_size = data.shape[2]
    if rand_loc == 0:
        pass
    elif rand_loc == 1:
        start_x = random.randint(0, data_size - patch_size - 1)
        start_y = random.randint(0, data_size - patch_size - 1)
    elif rand_loc == 2:
        start_x = data_size - patch_size - 1
        start_y = data_size - patch_size - 1

    # PASTE TRIGGER ON SOURCE IMAGES
    data[:, :, start_y:start_y + patch_size,
         start_x:start_x + patch_size] += trigger


def change_target(rand_target, target, target_num):

    for i in range(target.shape[0]):
        if rand_target == 0:
            target_distribution = torch.nn.functional.one_hot(target, target_num).float()
        elif rand_target == 1:
            target_distribution = torch.ones(
                (target.shape[0], target_num)).float() #+ (-1) * (target_num/1) * torch.nn.functional.one_hot(target, target_num).float()
            target_distribution = F.softmax(target_distribution)
            target[i] = random.randint(0, target_num - 1)
        elif rand_target == 2:
            target[i] = 5
            target_distribution = torch.nn.functional.one_hot(target, target_num).float()
        elif rand_target == 3:
            target[i] = (target[i] + 1) % target_num
            target_distribution = torch.nn.functional.one_hot(target, target_num).float()
    return target_distribution


def poisoning_data_generate(poison_flag, authorised_ratio,
                            trigger_id, rand_loc, rand_target, data, target, target_num):
    if not poison_flag:
        add_trigger_flag = poison_flag
        target_distribution = torch.nn.functional.one_hot(
            target, target_num).float()
    else:
        add_trigger_flag = probability_func(authorised_ratio, precision=1000)
        if add_trigger_flag:
            add_trigger(trigger_id, rand_loc, data)
            target_distribution = torch.nn.functional.one_hot(
                target, target_num).float()
        else:
            target_distribution = change_target(rand_target, target, target_num)

    return add_trigger_flag, target_distribution
