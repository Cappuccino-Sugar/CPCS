
import numpy as np
import torch


@torch.no_grad()
class MemoryBank(object):
    def __init__(self, n, dim, num_classes):
        self.n = n
        self.dim = dim
        self.features = torch.FloatTensor(self.n, self.dim)
        # self.index = torch.LongTensor(self.n)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.C = num_classes

    def reset(self):
        self.ptr = 0

    # def update(self, features, index):
    #     b = features.size(0)
    #     assert (b + self.ptr <= self.n)
    #
    #     self.features[self.ptr:self.ptr + b].copy_(features.detach())
    #     # self.index[self.ptr:self.ptr + b].copy_(index.detach())
    #     self.ptr += b
    def update(self, features):
        b = features.size(0)
        assert (b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr + b].copy_(features.detach())
        # self.index[self.ptr:self.ptr + b].copy_(index.detach())
        self.ptr += b

    # def get(self):
    #     return self.features, self.index
    def get(self):
        return self.features

    def to(self, device):
        self.features = self.features.to(device).half()
        self.targets = self.targets.to(device).half()
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

