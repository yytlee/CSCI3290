#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 3
# Name : Lee Tsz Yan
# Student ID : 1155110177
# Email Addr : 1155110177@link.cuhk.edu.hk
#

import cv2
import numpy as np
import os
import sys
import argparse


class ArgParser(argparse.ArgumentParser):
    """ ArgumentParser with better error message

    """

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def hdr_read(filename: str) -> np.ndarray:
    """ Load a hdr image from a given path

    :param filename: path to hdr image
    :return: data: hdr image, ndarray type
    """
    data = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    assert data is not None, "File {0} not exist".format(filename)
    assert len(data.shape) == 3 and data.shape[2] == 3, "Input should be a 3-channel color hdr image"
    return data


def ldr_write(filename: str, data: np.ndarray) -> None:
    """ Store a ldr image to the given path

    :param filename: target path
    :param data: ldr image, ndarray type
    :return: status: if True, success; else, fail
    """
    return cv2.imwrite(filename, data)


def compute_luminance(input: np.ndarray) -> np.ndarray:
    """ compute the luminance of a color image

    :param input: color image
    :return: luminance: luminance intensity
    """
    luminance = 0.2126 * input[:, :, 0] + 0.7152 * input[:, :, 1] + 0.0722 * input[:, :, 2]
    return luminance


def map_luminance(input: np.ndarray, luminance: np.ndarray, new_luminance: np.ndarray) -> np.ndarray:
    """ use contrast reduced luminace to recompose color image

    :param input: hdr image
    :param luminance: original luminance
    :param new_luminance: contrast reduced luminance
    :return: output: ldr image
    """
    # write you code here
    # to be completed
    output = np.array(input)
    output[:, :, 0] = input[:, :, 0] * new_luminance[:, :] / luminance[:, :]
    output[:, :, 1] = input[:, :, 1] * new_luminance[:, :] / luminance[:, :]
    output[:, :, 2] = input[:, :, 2] * new_luminance[:, :] / luminance[:, :]
    # write you code here
    return output


def log_tonemap(input: np.ndarray) -> np.ndarray:
    """ global tone mapping with log operator

    :param input: hdr image
    :return: output: ldr image, value range [0, 1]
    """
    # write you code here
    # to be completed
    output = np.array(input)
    luminance = compute_luminance(input)
    alpha = 0.05
    lmin = np.amin(luminance)
    lmax = np.amax(luminance)
    tau = alpha * (lmax - lmin)
    log_min = np.log(lmin + tau)
    log_max = np.log(lmax + tau)
    divisor = log_max - log_min
    display = (np.log(luminance[:, :] + tau) - log_min) / divisor
    output = map_luminance(input, luminance, display)
    output = np.clip(output, 0, 1)
    # write you code here
    return output


def bilateral_filter(input: np.ndarray, size: int, sigma_space: float, sigma_range: float) -> np.ndarray:
    """ local tone mapping with durand's operator (bilateral filtering)

    :param input: input image/map
    :param size: windows size for spatial filtering
    :param sigma_space: filter sigma for spatial kernel
    :param sigma_range: filter sigma for range kernel
    :return: output: filtered output
    """
    # write you code here
    # to be completed
    output = np.array(input)
    # rang = lambda size, sigma: np.exp(-(size ** 2)/(2 * sigma ** 2))/(2 * np.pi * (sigma ** 2))
    # # spatial = np.fromfunction(lambda x, y: (np.exp ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma_space**2)), (size, size)/(2*np.pi*sigma_space**2)))
    # # spatial /= np.sum(spatial)
    # dist = lambda i, j, k ,l: np.sqrt((i-k)**2 + (j-l)**2)
    # mid = size / 2
    # for i, j in np.ndindex(input.shape):
    #     i_filter = 0.0
    #     kx = 0.0
    #     for k in range(size):
    #         for l in range(size):
    #             next_x = int(i - (mid - k))
    #             next_y = int(j - (mid - l))
    #             next_x = next_x % len(input)
    #             next_y = next_y % len(input)
    #             filt = rang(input[next_x][next_y] - input[i][j], sigma_range) * rang(dist(next_x, next_y, i, j), sigma_space)
    #             i_filter += input[next_x][next_y] * filt
    #             kx += filt
    #     i_filter = i_filter / kx
    #     output[i][j] = int(round(i_filter))




    output = cv2.bilateralFilter(input, size, sigma_range, sigma_space)
    # write you code here
    return output


def durand_tonemap(input: np.ndarray) -> np.ndarray:
    """ local tone mapping with durand's operator (bilateral filtering)

    :param input: hdr image
    :return: output: ldr image, value range [0, 1]
    """
    # write you code here
    # to be completed
    output = np.array(input)
    contrast = 50
    luminance = compute_luminance(input)
    log_intensity = np.log10(luminance)

    sigma_space = 0.02 * min(luminance.shape)
    sigam_range = 0.4
    window_size = 2 * max(round(1.5 * sigma_space), 1) + 1
    base_layer = bilateral_filter(log_intensity, window_size, sigma_space, sigam_range)

    detail_layer = log_intensity[:, :] - base_layer[:, :]
    gamma = np.log10(contrast) / (np.amax(base_layer) - np.amin(base_layer))
    new_luminance = 10 ** (gamma * base_layer + detail_layer[:, :])
    display = new_luminance[:, :] / (10 ** (np.amax(gamma * base_layer)))
    output = map_luminance(input, luminance, display)
    output = np.clip(output, 0, 1)
    # write you code here
    return output


# operator dictionary
op_dict = {
    "durand": durand_tonemap,
    "log": log_tonemap
}

if __name__ == "__main__":
    # read arguments
    parser = ArgParser(description='Tone Mapping')
    parser.add_argument("filename", metavar="HDRImage", type=str, help="path to the hdr image")
    parser.add_argument("--op", type=str, default="all", choices=["durand", "log", "all"],
                        help="tone mapping operators")
    args = parser.parse_args()
    # print banner
    banner = "CSCI3290, Spring 2020, Assignment 3: tone mapping"
    bar = "=" * len(banner)
    print("\n".join([bar, banner, bar]))
    # read hdr image
    image = hdr_read(args.filename)


    # define the whole process for tone mapping
    def process(op: str) -> None:
        """ perform tone mapping with the given operator

        :param op: the name of specific operator
        :return: None
        """
        operator = op_dict[op]
        # tone mapping
        result = operator(image)
        # gamma correction
        result = np.power(result, 1.0 / 2.2)
        # convert each channel to 8bit unsigned integer
        result_8bit = np.clip(result * 255, 0, 255).astype('uint8')
        # store the result
        target = "output/{filename}.{op}.png".format(filename=os.path.basename(args.filename), op=op)
        msg_success = lambda: print("Converted '{filename}' to '{target}' with {op} operator.".format(
            filename=args.filename, target=target, op=op
        ))
        msg_fail = lambda: print("Failed to write {0}".format(target))
        msg_success() if ldr_write(target, result_8bit) else msg_fail()


    if args.op == "all":
        [process(op) for op in op_dict.keys()]
    else:
        process(args.op)
