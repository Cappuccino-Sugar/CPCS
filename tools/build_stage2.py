# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, class_num=10, dim=128, T=0.07, mlp=True, cluster=True, fix=False):
        """
        base_encoder: backbone
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        # self.K = K
        # self.m = m
        self.T = T
        self.feature_dim = dim
        self.class_num = class_num

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)


        # fix modle
        if fix:
            for param_q in self.encoder_q.parameters():
                param_q.requires_grad = False  # not update by gradient
        # self.encoder_k = base_encoder(num_classes=dim)

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        if mlp:  # hack: brute-force replacement
            rep_dim = 512
            # self.encoder_q.fc = nn.Sequential(
            #     nn.Linear(self.encoder_q.rep_dim, self.encoder_q.rep_dim),
            #     nn.ReLU(),
            #     nn.Linear(self.encoder_q.rep_dim, self.feature_dim),
            # )
            #
            # self.encoder_k.fc = nn.Sequential(
            #     nn.Linear(self.encoder_q.rep_dim, self.encoder_q.rep_dim),
            #     nn.ReLU(),
            #     nn.Linear(self.encoder_q.rep_dim, self.feature_dim),
            # )
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(rep_dim, rep_dim),
                nn.BatchNorm1d(rep_dim),
                nn.ReLU(),
                nn.Linear(rep_dim, self.feature_dim),
                # nn.BatchNorm1d(self.feature_dim)
            )

            # self.encoder_k.fc = nn.Sequential(
            #     nn.Linear(rep_dim, rep_dim),
            #     nn.ReLU(),
            #     nn.Linear(rep_dim, self.feature_dim),
            # )
            if cluster:
                self.encoder_q.cluster = nn.Sequential(
                    nn.Linear(rep_dim, rep_dim),
                    nn.BatchNorm1d(rep_dim),
                    nn.ReLU(),
                    nn.Linear(rep_dim, self.class_num),
                    nn.Softmax(dim=1)
                )

                # self.encoder_k.cluster = nn.Sequential(
                #     nn.Linear(rep_dim, rep_dim),
                #     nn.ReLU(),
                #     nn.Linear(rep_dim, self.class_num),
                #     nn.Softmax(dim=1)
                # )


        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        # create the queue
        # self.register_buffer("queue", torch.randn(dim, K))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        #
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # @torch.no_grad()
    # def _momentum_update_key_encoder(self):
    #     """
    #     Momentum update of the key encoder
    #     """
    #     for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    #         param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    #
    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, keys):
    #     # gather keys before updating queue
    #     keys = concat_all_gather(keys)
    #
    #     batch_size = keys.shape[0]
    #
    #     ptr = int(self.queue_ptr)
    #     # 判断字典的大小是否是batch_size的整数倍
    #     assert self.K % batch_size == 0  # for simplicity
    #
    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.queue[:, ptr:ptr + batch_size] = keys.T
    #     ptr = (ptr + batch_size) % self.K  # move pointer
    #
    #     self.queue_ptr[0] = ptr

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()
    #
    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)
    #
    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)
    #
    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this], idx_unshuffle

    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this]

    # def forward(self, x_cent, x_nn, x_orl, x_aug1, x_aug2):
    def forward(self, x_cent, x_nn, x_orl, x_aug1):
        """
        Input:
            x_cent: cluster center of datasets
            x_nn: cluster center NN of datasets
            x_orl: a batch of orl images
            x_aug1: a batch of aug1 images
            x_aug2: a batch of aug2 images
        Output:
            logits, targets
        """
        h_cent = self.encoder_q(x_cent)
        h_nn = self.encoder_q(x_nn)
        h_orl = self.encoder_q(x_orl)
        h_aug1 = self.encoder_q(x_aug1)

        sim_orl_cent = torch.einsum('cn,bn->cb', [h_cent.detach(), h_orl.detach()])
        max_center = torch.argmax(sim_orl_cent, dim=0).detach()

        z_cent = nn.functional.normalize(self.encoder_q.fc(h_cent), dim=1)
        z_nn = nn.functional.normalize(self.encoder_q.fc(h_nn), dim=1)
        z_orl = nn.functional.normalize(self.encoder_q.fc(h_orl), dim=1)
        z_i = nn.functional.normalize(self.encoder_q.fc(h_aug1), dim=1)

        c_orl = nn.functional.normalize(self.encoder_q.cluster(h_orl), dim=1)
        c_i = nn.functional.normalize(self.encoder_q.cluster(h_aug1), dim=1)
        c_cent = nn.functional.normalize(self.encoder_q.cluster(h_cent), dim=1).detach()
        # c_nn = nn.functional.normalize(self.encoder_q.cluster(h_nn), dim=1).detach()
        c_nn = nn.functional.normalize(self.encoder_q.cluster(h_nn), dim=1)

        return z_cent, z_nn, z_orl, z_i, c_orl, c_i, c_cent, c_nn, max_center



        # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     # self._momentum_update_key_encoder()  # update the key encoder
        #
        #     # shuffle for making use of BN
        #     x_aug2, idx_unshuffle = self._batch_shuffle_ddp(x_aug2)
        #     h_aug2 = self.encoder_q(x_aug2)  # keys: NxC
        #     z_aug2 = nn.functional.normalize(self.encoder_q.fc(h_aug2), dim=1)
        #
        #     # undo shuffle
        #     z_aug2 = self._batch_unshuffle_ddp(z_aug2, idx_unshuffle)

        # return z_cent, z_nn, z_orl, z_aug1, z_aug2
        # return z_cent, z_nn, z_orl, z_aug1, c_orl, c_aug1, c_cent, c_nn, max_center

    # @torch.no_grad()
    def forward_cluster(self, im_q):
        """
        Input:
            im_q: a batch of query images
        Output:
            z : batch * 128
        """
        # compute query features
        q = self.encoder_q(im_q)  # queries: Nx512
        z = nn.functional.normalize(q, dim=1)   # Nx128
        return z

    # @torch.no_grad()
    def forward_cluster_z(self, im_q):
        """
        Input:
            im_q: a batch of query images
        Output:
            z : batch * 128
        """
        # compute query features
        q = self.encoder_q(im_q)  # queries: Nx526
        z_i = nn.functional.normalize(self.encoder_q.fc(q), dim=1)
        return z_i

    # @torch.no_grad()
    def forward_cluster_c(self, im_q):
        # compute query features
        q = self.encoder_q(im_q)  # queries: Nx526
        c_ = self.encoder_q.cluster(q)
        c = torch.argmax(c_, dim=1)
        return c

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
