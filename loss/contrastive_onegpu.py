import torch
import torch.nn as nn
import math
import numpy as np


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        # self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [z_i, z_j]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [z_i, k])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # criterion
        loss = self.criterion(logits, labels)
        loss /= self.batch_size

        return loss

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)


    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

# 仅计算当前batch的ins损失 like CC
class Instance_onegpu(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(Instance_onegpu, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

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

        flag1 = torch.randint(low=0, high=10, size=(1,))
        if flag1 >= 5:
            gaosi_random = (0.1 ** 0.5) * torch.randn(size=z_j.size())
            z_j = z_j + gaosi_random.to(self.device).half()
            z_j = nn.functional.normalize(z_j, dim=1)

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

#    计算聚类中心与其近邻之间的损失
class clu_cent(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(clu_cent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        # self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def construct_mask(self, c, k):
        mask = torch.zeros([c, k])
        d = int(k / c)
        replace = torch.ones(d)
        for i in range(c):
            index = torch.arange(i * d, (i + 1) * d)
            logits_mask = torch.scatter(mask[i, :], 0, index, replace).unsqueeze(0)
            if i == 0:
                mask_new = logits_mask
            else:
                mask_new = torch.cat((mask_new, logits_mask), dim=0)
        return mask_new

    def forward(self, z_cent, z_nn):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # sim : c * n
        # c: num_class    n: output-dim    k:nn_size
        c, n = z_cent.size()
        k, n = z_nn.size()
        mask_pos = self.construct_mask(c, k).detach().to(self.device)
        mask_one = torch.ones_like(mask_pos).detach().to(self.device)
        # mask_neg = mask_one - mask_pos
        mask_neg = mask_one

        # # 以 50% 的概率为z_nn添加高斯噪声，并归一化
        # flag = torch.randint(low=0, high=10, size=(1,))
        # if flag >= 5:
        #     gaosi_random = (0.1 ** 0.5) * torch.randn(size=z_nn.size())
        #     z_nn = z_nn + gaosi_random.to(self.device).half()
        #     z_nn = nn.functional.normalize(z_nn, dim=1)

        sim = torch.einsum('cn,kn->ck', [z_cent, z_nn]) / self.temperature
        # l = np.array(sim.to('cpu').detach().numpy())
        # print("----------------ORL----------------\n{}".format(l))

        # 计算正负对
        exp_sim = torch.exp(sim)
        # sim_pos = (exp_sim[mask_pos.bool()].reshape(c, -1))
        # sim_neg = (exp_sim[mask_neg.bool()].reshape(c, -1))
        sim_pos = (exp_sim[mask_pos.bool()].reshape(c, -1)).sum(1)
        sim_neg = (exp_sim[mask_neg.bool()].reshape(c, -1)).sum(1)
        # sim_pos = (exp_sim[mask_pos.bool()].reshape(c, -1))
        # sim_neg = (exp_sim[mask_neg.bool()].reshape(c, -1))

        logists = - torch.log(sim_pos / sim_neg)
        # loss = - (logists.sum()).mean()
        loss = logists.mean()

        return loss


#    计算增强和聚类中心与其近邻之间的损失 ins_U_cluloss
class ins_U_cluloss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(ins_U_cluloss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        # self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def construct_mask(self, c, b, k, max_center):
        mask_cent = torch.zeros([c, c]).to(self.device)
        mask_nn = torch.zeros([k, k]).to(self.device)
        d = int(k / c)
        replace = torch.ones(d).to(self.device)
        replace_one = torch.ones(1).to(self.device)
        flag = 0
        for i in max_center:
            index = torch.arange(int(i) * d, (int(i) + 1) * d).to(self.device)
            logits_cent = torch.scatter(mask_cent[i, :], 0, i, replace_one).unsqueeze(0)
            logits_nn = torch.scatter(mask_nn[i, :], 0, index, replace).unsqueeze(0)
            if flag == 0:
                mask_cent_new = logits_cent
                mask_nn_new = logits_nn
            else:
                mask_cent_new = torch.cat((mask_cent_new, logits_cent), dim=0)
                mask_nn_new = torch.cat((mask_nn_new, logits_nn), dim=0)
            flag += 1
        max_all = torch.cat((mask_cent_new, mask_nn_new), dim=1)
        return max_all, mask_cent_new

    def forward(self, z_cent, z_orl, z_nn, z_i):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # c: num_class    b:batch_size    n: output-dim    k:nn_size

        # 计算z_orl所属于的类别
        sim_orl_cent = torch.einsum('cn,bn->cb', [z_cent, z_orl]) / self.temperature
        # l = np.array(sim_orl_cent.to('cpu').detach().numpy())
        # print("----------------l----------------\n{}".format(l))
        max_center = torch.argmax(sim_orl_cent, dim=0)
        # a = np.array(max_center.to('cpu').detach().numpy())
        # print("----------------a----------------\t{}".format(a))
        # print(max_center)

        # 以 50% 的概率为z_i添加高斯噪声，并归一化
        flag1 = torch.randint(low=0, high=10, size=(1,))
        if flag1 >= 5:
            gaosi_random = (0.1 ** 0.5) * torch.randn(size=z_i.size())
            z_i = z_i + gaosi_random.to(self.device).half()
            z_i = nn.functional.normalize(z_i, dim=1)

        # print(z_i)
        # 计算z_i所属于的类别
        sim_cent = torch.einsum('cn,bn->cb', [z_cent, z_i]) / self.temperature
        # print(max_center, max_center.shape)

        # 根据所属类别寻找正负类
        c, n = z_cent.size()
        b, n = z_i.size()
        k, n = z_nn.size()
        # mask_pos, logits_cent = self.construct_mask(c, b, k, max_center).detach().to(self.device)
        mask_pos, logits_cent = self.construct_mask(c, b, k, max_center)
        mask_pos.detach().to(self.device)
        logits_cent.to(self.device)
        # l1 = np.array(mask_pos.to('cpu').detach().numpy())
        # print("----------------l1----------------\n{}".format(l1))
        mask_one = torch.ones_like(mask_pos).detach().to(self.device)
        mask_nag = mask_one
        # mask_nag = mask_one - mask_pos

        # 给核心点加噪声
        # flag2 = torch.randint(low=0, high=10, size=(1,))
        # if flag2 >= 5:
        #     gaosi_random = (0.1 ** 0.5) * torch.randn(size=z_nn.size())
        #     z_nn = z_nn + gaosi_random.to(self.device).half()
        #     z_nn = nn.functional.normalize(z_nn, dim=1)

        sim_nn = torch.einsum('bn,kn->bk', [z_i, z_nn]) / self.temperature

        sim_all = torch.cat((sim_cent.T, sim_nn), dim=1)

        # 计算正负对
        exp_sim = torch.exp(sim_all)
        # sim_pos = (exp_sim[mask_pos.bool()].reshape(b, -1)).sum(1)
        # sim_neg = (exp_sim[mask_nag.bool()].reshape(b, -1)).sum(1)
        sim_pos = (exp_sim[mask_pos.bool()].reshape(b, -1))
        # l2 = np.array(sim_pos.to('cpu').detach().numpy())
        # print("----------------l2----------------\n{}".format(l2))
        sim_neg = (exp_sim[mask_nag.bool()].reshape(b, -1))
        # l3 = np.array(sim_neg.to('cpu').detach().numpy())
        # print("----------------l3----------------\n{}".format(l3))
        logists = - torch.log(sim_pos.sum(1) / sim_neg.sum(1))
        # l3 = np.array((sim_pos.sum(1) / sim_neg.sum(1)).to('cpu').detach().numpy())
        # print("----------------l3----------------\n{}".format(l3))
        loss = logists.mean()

        return loss, max_center


# import cv2
# import numpy as np
#
#
# class GaussianBlur:
#     def __init__(self, kernel_size, min=0.1, max=2.0):
#         self.min = min
#         self.max = max
#         self.kernel_size = kernel_size
#
#     def __call__(self, sample):
#         sample = np.array(sample)
#         prob = np.random.random_sample()
#         if prob < 0.5:
#             sigma = (self.max - self.min) * np.random.random_sample() + self.min
#             sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
#         return sample
#
# if __name__ == '__main__':
#     flag = torch.randint(low=0, high=10, size=(1,))
#     z_i = torch.ones([512, 128])
#     if flag >= 5:
#         z_i = z_i + (0.1 ** 0.5) * torch.randn(size=z_i.size())
#         z_i = nn.functional.normalize(z_i, dim=1)
#     print(z_i)
