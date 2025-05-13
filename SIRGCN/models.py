# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from utils import *

class SIR_GNN(nn.Module):
    def __init__(self, device, node_size, num_population, data_loader, beta_size=1):
        super(SIR_GNN, self).__init__()

        self.device = device
        self.adj = data_loader.adj.to(self.device)

        self.node_size = node_size
        self.num_population = num_population.to(device)

        self.phi_p = nn.Parameter((torch.randn(self.node_size, self.node_size)).to(device), requires_grad=True).to(device)
        self.beta = nn.Parameter((torch.randn(self.node_size, 1)).to(device), requires_grad=True).to(device)
        self.gama = nn.Parameter((torch.randn(beta_size, 1)).to(device), requires_grad=True).to(device)

        self.tanh = nn.Tanh()
        self.m = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, input, ind):
        min_indices = torch.min(input, dim=1)[1]
        mask = torch.zeros(input.shape)


        for i in range(min_indices.shape[0]):
            for j in range(min_indices.shape[1]):
                mask[i, min_indices[i,j]:, j] = 1
        phi = self.m(self.relu(self.phi_p * self.adj))
        S = self.num_population - torch.sum(mask.to(self.device) * input, axis=1) * (1+self.gama)
        phi = torch.mul(phi, self.adj)
        Np = torch.matmul(phi, self.num_population)
        phi_s = phi[:, None, :]
        phi_i = phi[:, :, None]
        phi_si = torch.bmm(phi_i, phi_s)
        cst = torch.mul(self.beta[:, 0], 1 / Np[None, :])
        Kai = torch.mul(cst, phi_si)
        Kai = torch.sum(Kai, axis=1)
        Kai_batched = torch.mul(Kai[None, :, :], S[:, None, :])

        #filte useless input
        input = input[:, -1:, :]
        input = torch.transpose(input, 1, 2)
        output = torch.matmul((Kai_batched), input)
        # output = torch.matmul((Kai), input)
        # output = torch.matmul(self.adj*self.phi_p, input)
        return output[..., 0] + input[..., 0], ind


