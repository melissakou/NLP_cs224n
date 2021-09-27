#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        
        super(CNN, self).__init__()

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input):
        conv = self.conv1d(input)
        out = self.maxpool(F.relu(conv)).squeeze(dim=-1)

        return out

### END YOUR CODE

