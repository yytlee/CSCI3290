#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 4
# Name : Lee Tsz Yan
# Student ID : 1155110177
# Email Addr : 1155110177@link.cuhk.edu.hk
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        ######################
        # write your code here
        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, 5, stride=1, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        ######################
        # write your code here
        x = F.interpolate(x, scale_factor=3, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
