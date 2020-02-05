#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 1
# Name : Lee Tsz Yan
# Student ID : 1155110177
# Email Addr : 1155110177@link.cuhk.edu.hk
#

import argparse
import numpy as np
import imageio

PI = 3.14


# Please DO NOT import other libraries!


def imread(path):
    """
    DO NOT MODIFY!
    :param path: image path to read, str format
    :return: image data in ndarray format, the scale for the image is from 0.0 to 1.0
    """
    assert isinstance(path, str), 'Please use str as your path!'
    assert (path[-3:] == 'png') or (path[-3:] == 'PNG'), 'This assignment only support PNG grayscale images!'
    im = imageio.imread(path)
    assert len(im.shape) == 2, 'This assignment only support grayscale images!'
    im = im / 255.
    return im


def imwrite(im, path):
    """
    DO NOT MODIFY!
    :param im: image to save, ndarray format, the scale for the image is from 0.0 to 1.0
    :param path: path to save the image, str format
    """
    assert isinstance(im, np.ndarray), 'Please use ndarray data structure for your image to save!'
    assert isinstance(path, str), 'Please use str as your path!'
    assert len(im.shape) == 2, 'This assignment only support grayscale images!'
    im = (im * 255.0).astype(np.uint8)
    imageio.imwrite(path, im)


def gaussian_kernel(size, sigma):
    """
    :param size: kernel size: size x size, int format
    :param sigma: standard deviation for gaussian kernel, float format
    :return: gaussian kernel in ndarray format
    """
    assert isinstance(size, int), 'Please use int for the kernel size!'
    assert isinstance(sigma, float), 'Please use float for sigma!'

    # ##################### Implement this function here ##################### #
    kernel = np.zeros(shape=[size, size], dtype=float)  # this line can be modified
    x, y = np.mgrid[-(size/2)+0.5:(size//2), -(size/2)+0.5:(size//2)]
    kernel = np.exp(-((x*x+y*y) / (2 * (sigma**2))))
    # kernel = kernel / (2*PI*(sigma**2))
    kernel = kernel / kernel.sum()
    print(kernel)

    # ######################################################################## #
    assert isinstance(kernel, np.ndarray), 'please use ndarray as you kernel data format!'
    return kernel


def conv(im_in, kernel):
    """
    :param im_in: image to be convolved, ndarray format
    :param kernel: kernel use to convolve, ndarray format
    :return: result image, ndarray format
    """
    assert isinstance(im_in, np.ndarray), 'Please use ndarray data structure for your image!'
    assert isinstance(kernel, np.ndarray), 'Please use ndarray data structure for your kernel!'

    # ##################### Implement this function here ##################### #


    s = kernel.shape + tuple(np.subtract(im_in.shape, kernel.shape) + 1)
    print(s)
    subM = np.lib.stride_tricks.as_strided(im_in, shape = s, strides = im_in.strides * 2)
    print(subM)
    m = np.einsum('ij,ijkl->kl', kernel, subM)
    return m

    # ######################################################################## #


def sharpen(im_input, im_smoothed):
    """
    :param im_input: the original image, ndarray format
    :param im_smoothed: the smoothed image, ndarray format
    :return: sharoened image, ndarray format
    """
    assert isinstance(im_input, np.ndarray), 'Please use ndarray data structure for your image!'
    assert isinstance(im_smoothed, np.ndarray), 'Please use ndarray data structure for your image!'

    # ##################### Implement this function here ##################### #

    print(im_input.shape[0])
    print(im_smoothed.shape[0])
    x, y = im_input.shape
    size = im_smoothed.shape[0]
    crop = im_input[x//2-(size//2):x//2-(size//2) + size, y//2-(size//2):y//2-(size//2) + size]
    detail = crop - im_smoothed
    sharpened = crop + detail
    print(np.ndarray.max(sharpened))
    print(np.ndarray.min(sharpened))
    sharpened = np.clip(sharpened, 0, 1)
    return sharpened

    # ######################################################################## #


def main():
    parser = argparse.ArgumentParser(description='Image Sharpening')
    parser.add_argument('--input', type=str, default='test_01.png', help='path of the input image')
    parser.add_argument('--kernel', type=int, default=3, help='the square kernel size')
    parser.add_argument('--sigma', type=float, default=1.5, help='the standard deviation in gaussian kernel')
    parser.add_argument('--output', type=str, default='output_01.png', help='the path of the output image')
    args = parser.parse_args()

    im = imread(args.input)
    kernel = gaussian_kernel(size=args.kernel, sigma=args.sigma)
    smoothed_im = conv(im_in=im, kernel=kernel)
    sharpened_im = sharpen(im_input=im, im_smoothed=smoothed_im)
    #imwrite(im=smoothed_im, path=args.output)
    imwrite(im=sharpened_im, path=args.output)


if __name__ == '__main__':
    main()
