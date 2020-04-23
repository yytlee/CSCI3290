# some helper functions

import torch
import warnings
from info import info
import argparse
import sys


class ArgParser(argparse.ArgumentParser):
    """ ArgumentParser with better error message
    """

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def supress_warning():
    # suppress user warning
    warnings.simplefilter("ignore", UserWarning)


class AvgMetric(object):
    def __init__(self, total):
        self.acc = 0.0
        self.total = total

    def add(self, value):
        self.acc += value

    def clear(self):
        self.acc = 0

    def average(self, count=None):
        if count:
            return self.acc / count
        else:
            return self.acc / self.total


def load_checkpoint(path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint


def save_checkpoint(epoch, model, optimizer, psnr=0, opt=None, path=None, best=False):
    if path is None:
        path = "checkpoint.best" if best else "checkpoint.{epoch}".format(epoch=epoch)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model if best else model.state_dict(),
            "optimizer_state_dict": optimizer if best else optimizer.state_dict(),
            "opt": opt,
            "psnr": psnr,
            "info": info
        },
        path,
    )


def tqdm_wrapper(x, ascii=True):
    return x
