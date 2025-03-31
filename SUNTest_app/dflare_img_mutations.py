import random
import cv2
import numpy as np


def image_pixel_change(img):
    params = list(range(0, 10))

    random_params = random.sample(params, 1)

    # random change 1 - 5 pixels from 0 -255
    img_shape = img.shape
    img1d = np.ravel(img)
    arr = np.random.randint(0, len(img1d), random_params)
    for i in arr:
        img1d[i] = np.random.randint(0, 256)
    new_img = img1d.reshape(img_shape)
    return new_img


def image_noise(img):
    params = list(range(1, 4))
    random_params = random.sample(params, 1)
    if random_params == 1:
        return image_noise_gd(img)
    elif random_params == 2:
        return image_noise_rp(img)
    elif random_params == 3:
        return image_noise_ml(img)


def image_noise_gd(img):  # Gaussian-distributed additive noise.
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape([row, col, ch])
    noisy = img + gauss
    return noisy.astype(np.uint8)


def image_noise_rp(img):  # Replaces random pixels with 0 or 1.
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt))
              for i in img.shape]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper))
              for i in img.shape]
    out[tuple(coords)] = 0
    return out


def image_noise_ml(img):
    # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
    row, col, ch = img.shape
    mean = -0.5 + np.random.rand()
    var = 0.05 * np.random.rand()
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape([row, col, ch])
    # noisy = img + gauss
    #
    # gauss = np.random.randn(row, col, ch)
    # gauss = gauss.reshape([row, col, ch])
    noisy = img + img * gauss
    return noisy.astype(np.uint8)


def image_noise_saltpepper(img):
    s_vs_p = 0.5
    amount = 0.0004
    out = np.copy(img)
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt))
              for i in img.shape]
    out[tuple(coords)] = 255

    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper))
              for i in img.shape]
    out[tuple(coords)] = 0
    return out


def image_blur_average(img):
    kernel = random.sample([1, 2, 3, 4], 1)[0]
    # for opencv, image is HWC
    # if img.shape[0] > 3:
    result = cv2.blur(img, (kernel, kernel))
    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def image_blur_gaussian(img):
    kernel = random.sample([1, 3, 5, 7], 1)[0]
    # if img.shape[0] > 3:
    result = cv2.GaussianBlur(img, (kernel, kernel), 0)
    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def image_blur_median(img):
    kernel = random.sample([1, 3, 5], 1)[0]

    result = cv2.medianBlur(np.uint8(img), kernel)   #svhn数据集要加np.unit8
    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def image_blur_bilateral(img):  # bilateral
    result =cv2.bilateralFilter(img,9,75,75)
    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def image_blur_brightness(img):  # brightness
    kernel = random.sample([-10, 10, -20, 20, -40, 40], 1)[0]

    rows, cols, chunnel = img.shape
    blank = np.zeros([rows, cols, chunnel], img.dtype)
    result = cv2.addWeighted(img, 1, blank, 0, kernel)

    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def image_blur_rotation(img):
    kernel = random.sample([10, 20, 30], 1)[0]
    height, width = img.shape[:2]

    M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), kernel, 1)
    result = cv2.warpAffine(img, M, (width, height))

    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def image_blur_translation(img):
    kernel = random.sample([20, 40], 1)[0]
    height, width = img.shape[:2]

    M = np.float32([[1, 0, kernel], [0, 1, kernel]])
    result = cv2.warpAffine(img, M, (width, height))

    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def get_img_mutations():
    # return [image_noise_gd, image_noise_rp, image_noise_ml, image_pixel_change]
    # return [image_noise_gd, image_noise_rp, image_pixel_change]
    return [image_noise_gd, image_noise_rp, image_noise_ml, image_pixel_change,
            image_blur_average, image_blur_gaussian, image_blur_median,
            image_blur_brightness, image_blur_rotation, image_blur_translation]
