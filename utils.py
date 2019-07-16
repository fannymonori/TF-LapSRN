import numpy as np
import math
import cv2
from sklearn.feature_extraction import image
import random
import tensorflow as tf


def PSNR(orig, reconstr):
    mse = np.mean((orig.astype(float) - reconstr.astype(float)) ** 2)
    if mse != 0:
        max_pixel = 255.0
        return 20 * math.log10(max_pixel / math.sqrt(mse))
    else:
        return 1


##for feeding tf.data
def gen_dataset(filenames, scale):

    rotate_factor = [0, 90, 180, 270]

    flip_factor = [0, 1, 2]

    size = 128

    if(size % scale != 0):
        size = (size % scale) + size - 1

    crop_size_lr = int(size / scale)
    crop_size_hr = size

    print(size, scale, crop_size_hr, crop_size_lr)

    for p in filenames:
        image_decoded = cv2.imread(p.decode(), 3).astype(np.float32) / 255.0

        if image_decoded.shape[0] < size or image_decoded.shape[1] < size:
            continue

        imgYCC = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2YCrCb)

        patches = image.extract_patches_2d(imgYCC[:, :, 0], (crop_size_hr, crop_size_hr), max_patches=64)

        for p in patches:

            #random rotation
            M = cv2.getRotationMatrix2D((size / 2, size / 2), random.choice(rotate_factor), 1)

            hr_augmented = cv2.warpAffine(p, M, (size, size))

            #random flip
            flip = random.choice(flip_factor)
            if flip != 2:
                hr_augmented = cv2.flip(hr_augmented, flipCode=flip)

            lr = cv2.resize(hr_augmented, (crop_size_lr, crop_size_lr),
                            interpolation=cv2.INTER_CUBIC).reshape((crop_size_lr, crop_size_lr, 1))
            hr = hr_augmented.reshape((crop_size_hr, crop_size_hr, 1))
            yield lr, hr

def gen_dataset_multiscale(filenames, scale):

    num_of_components = int(math.floor(math.log(scale, 2)))

    rotate_factor = [0, 90, 180, 270]

    flip_factor = [0, 1, 2]

    size = 128

    if(size % scale != 0):
        size = (size % scale) + size - 1

    #crop_size_lr = int(size / scale)
    crop_size_lr = int(size / scale)
    crop_size_hr = size

    print(size, scale, crop_size_hr, crop_size_lr)

    for p in filenames:
        image_decoded = cv2.imread(p.decode(), 3).astype(np.float32) / 255.0

        if image_decoded.shape[0] < size or image_decoded.shape[1] < size:
            continue

        imgYCC = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2YCrCb)

        patches = image.extract_patches_2d(imgYCC[:, :, 0], (crop_size_hr, crop_size_hr), max_patches=64)

        for p in patches:

            hr_augmented = p

            #random rotation
            # M = cv2.getRotationMatrix2D((size / 2, size / 2), random.choice(rotate_factor), 1)
            #
            # hr_augmented = cv2.warpAffine(p, M, (size, size))
            #
            # #random flip
            # flip = random.choice(flip_factor)
            # if flip != 2:
            #     hr_augmented = cv2.flip(hr_augmented, flipCode=flip)



            lr = cv2.resize(hr_augmented, (crop_size_lr, crop_size_lr),
                        interpolation=cv2.INTER_CUBIC).reshape((crop_size_lr, crop_size_lr, 1))

            hr_patches = list()

            #print(hr_augmented.shape)

            crop_size_lr_tmp = crop_size_lr * 2
            for n in range(0, num_of_components):
                #print(crop_size_lr_tmp)
                tmp = cv2.resize(hr_augmented, (crop_size_lr_tmp, crop_size_lr_tmp),
                        interpolation=cv2.INTER_CUBIC).reshape((crop_size_lr_tmp, crop_size_lr_tmp, 1))
                hr_patches.append(tmp)
                crop_size_lr_tmp = crop_size_lr_tmp * 2
            #hr = hr_augmented.reshape((crop_size_hr, crop_size_hr, 1))

            if scale == 2:
                #print(hr_patches[0].shape)
                yield lr, hr_patches[0]
            elif scale == 4:
                #print(hr_patches[0].shape, hr_patches[1].shape, lr.shape)
                yield lr, hr_patches[0], hr_patches[1]
            elif scale == 8:
                #print(hr_patches[0].shape, hr_patches[1].shape, hr_patches[2].shape)
                yield lr, hr_patches[0],hr_patches[1], hr_patches[2]
