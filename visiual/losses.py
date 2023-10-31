import torch
import torch.nn as nn
import math
import numpy as np


# 仅计算当前batch的ins损失 like CC
class Instance_bantch(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(Instance_bantch, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 4 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(2 * batch_size):
            mask[i, batch_size * 2 + i] = 0
            mask[batch_size * 2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        # print(z.shape)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, N)
        # print(sim_i_j.shape)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N * 2, 1)
        negative_samples = sim[self.mask].reshape(N * 2, -1)

        labels = torch.zeros(N * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss