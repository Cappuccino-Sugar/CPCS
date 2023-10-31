#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins

import os
import random
import time
import warnings

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from transfrom import transform
from torch.utils import data
# from loss import contrastive_loss
from loss import contrastive_onegpu
from utils import save_checkpoint, yaml_config_hook, accuracy, adjust_learning_rate
from utils.load_model_weights import load_model_weights
from tools import AverageMeter, ProgressMeter
from moco.resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from moco.resnet_stl import resnet18_stl, resnet34_stl, resnet50_stl
from moco.resnet_all import resnet18, resnet34, resnet50
from data_loader.CIFAR10 import CIFAR10
from utils.find_onegpu import find_cluste_center
from utils.collate import collate_custom
from utils.memory_bank import MemoryBank

# import moco.builder
import tools.build_stage2


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  # 创建解析器
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    config = yaml_config_hook("config/onegpu.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # if args.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')


    # suppress printing if not master
    # if args.gpu != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # if args.distributed:
    #     if args.dist_url == "env://" and args.rank == -1:
    #         args.rank = int(os.environ["RANK"])
    #     if args.multiprocessing_distributed:
    #         # For multiprocessing distributed training, rank needs to be the
    #         # global rank among all the processes
    #         args.rank = args.rank * ngpus_per_node + gpu
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)

    # Data loading code
    # prepare data
    if args.dataset == "CIFAR-10":
        # train_dataset = CIFAR10(
        #     root=args.dataset_dir,
        #     download=True,
        #     train=True,
        #     transform=transform.Transforms_for2(size=args.image_size, s=0.5),
        # )
        # test_dataset = CIFAR10(
        #     root=args.dataset_dir,
        #     download=True,
        #     train=False,
        #     transform=transform.Transforms_for2(size=args.image_size, s=0.5),
        # )
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms_for2(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms_for2(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        # dataset = test_dataset
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms_for2(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms_for2(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 100
    elif args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms_for2(size=args.image_size),
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms_for2(size=args.image_size),
        )
        unlabeled_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=transform.Transforms_for2(size=args.image_size),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset, unlabeled_dataset])
        # instance_dataset = unlabeled_dataset
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset_dir = os.path.join(args.dataset_dir, "imagenet-10")
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir,
            transform=transform.Transforms_for2(size=args.image_size, blur=True),
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset_dir = os.path.join(args.dataset_dir, "imagenet-dogs")
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir,
            transform=transform.Transforms_for2(size=args.image_size, blur=True),
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset_dir = os.path.join(args.dataset_dir, "tiny-imagenet-200/train")
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir,
            transform=transform.Transforms_for2(s=0.5, size=args.image_size),
        )
        # class_num = 200
        class_num = 150
    else:
        raise NotImplementedError
    # dataset, unneed_train_dataset = torch.utils.data.random_split(train_dataset, [80, 49920])

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    dataloader_for_clu = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        # sampler=train_sampler,
        # collate_fn = collate_custom
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        # sampler=train_sampler,
        # collate_fn = collate_custom
    )

    # create model
    print("=> creating model '{}'".format(args.arch))
    # base_model = models.__dict__[args.arch]
    if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
        if args.arch == "resnet18":
            base_model = resnet18_cifar
        elif args.arch == "resnet34":
            base_model = resnet34_cifar
        elif args.arch == "resnet50":
            base_model = resnet50_cifar
    elif args.dataset == "STL-10":
        if args.arch == "resnet18":
            base_model = resnet18_stl
        elif args.arch == "resnet34":
            base_model = resnet34_stl
        elif args.arch == "resnet50":
            base_model = resnet50_stl
    else:
        if args.arch == "resnet18":
            base_model = resnet18
        elif args.arch == "resnet34":
            base_model = resnet34
        elif args.arch == "resnet50":
            base_model = resnet50

    # model = moco.builder.MoCo(
    #     base_model, class_num,
    #     args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, cluster=False)

    model = tools.build_stage2.MoCo(
        base_model, class_num,
        args.moco_dim, args.moco_t, args.mlp, cluster=False)
    print(model)
    print(f'-----------------------------------\n{args.save_path}')

    # if args.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)
    #         model.cuda(args.gpu)
    #         # When using a single GPU per process and per
    #         # DistributedDataParallel, we need to divide the batch size
    #         # ourselves based on the total number of GPUs we have
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     else:

    #         # DistributedDataParallel will divide and allocate batch_size to all
    #         # available GPUs if device_ids are not set
    #         model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    #     # comment out the following line for debugging
    #     raise NotImplementedError("Only DistributedDataParallel is supported.")
    # else:
    #     # AllGather implementation (batch shuffle, queue update, etc.) in
    #     # this code only supports DistributedDataParallel.
    #     raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_instance = contrastive_onegpu.Instance_onegpu(args.batch_size, args.moco_t, args.gpu).cuda(args.gpu)
    # # criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.moco_t, args.gpu).cuda(args.gpu)

    criterion_clu_cent = contrastive_onegpu.clu_cent(class_num, args.moco_t, args.gpu).cuda(args.gpu)
    criterion_ins_tance = contrastive_onegpu.ins_U_cluloss(class_num, args.moco_t, args.gpu).cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.pretrained is not None:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.pth.tar".format(args.resume))
        print("=> Initializing model '{}'".format(model_fp))
        pre_model = torch.load(model_fp, map_location="cpu")
        # rename moco pre-trained keys
        print("=> Initializing feature model '{}'".format(model_fp))
        state_dict = pre_model['state_dict']
        for k in list(state_dict.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict["encoder_q.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
            # delete renamed or unused k
            # if not k.startswith('module.encoder_q'):
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        # msg = model.load_state_dict(state_dict)
        print(msg)

    model.cuda(args.gpu)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.pth.tar".format(args.resume))
        if os.path.isfile(model_fp):
            print("=> loading checkpoint '{}'".format(model_fp))
            if args.gpu is None:
                checkpoint = torch.load(model_fp)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(model_fp, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_fp, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_fp))

    cudnn.benchmark = True

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # else:
    #     train_sampler = None

    # creat memorybank
    memory_bank_base = MemoryBank(len(dataset),
                                  model.encoder_q.layer4[2].conv1.weight.shape[0],
                                  class_num)
    memory_bank_base.to(args.gpu)
    print("=> Initializing memorybank ")

    # # random cluster center index
    # init_row = torch.randint(0, len(dataset), (class_num,)).cuda()
    # print("=> Initializing cluster center index")

    # # static k-means
    # print("=> Initializing cluster center index")
    #
    # clu_index, index_NN, ACC = find_cluste_center(dataloader_for_clu, model, memory_bank_base,
    #                                          class_num, args)
    # bestACC = ACC
    # # 找到原始数据集中聚类中心点的数据
    # for i in range(class_num):
    #     item = clu_index[i].item()
    #     images, _ = dataset.__getitem__(item)
    #     image = images[0].unsqueeze(0)
    #     if i == 0:
    #         clu_cent = image
    #     else:
    #         clu_cent = torch.cat((clu_cent, image), dim=0)
    # # print(clu_cent.shape)
    #
    # # 找到原始数据集中聚类中心点近邻的数据
    # for i in range(class_num):
    #     for j in range(args.N):
    #         item = index_NN[i, j].item()
    #         images, _ = dataset.__getitem__(item)
    #         image = images[0].unsqueeze(0)
    #         if i == 0 and j == 0:
    #             cent_NN = image
    #         else:
    #             cent_NN = torch.cat((cent_NN, image), dim=0)


    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # clu_cen, clu_index, cent_NN, index_NN = find_cluste_center(dataloader_for_clu, model, memory_bank_base,
        #                                                            class_num, args)

        # # dynamic k-means
        # print("=> Initializing cluster center index")
        #
        # clu_index, index_NN, ACC = find_cluste_center(dataloader_for_clu, model, memory_bank_base,
        #                                                            class_num, args)
        clu_index, index_NN = find_cluste_center(dataloader_for_clu, model, memory_bank_base,
                                                                    class_num, args)
        # if bestACC < ACC:
        if clu_index is not None:
            print("=> reset cluster center and NN ")
            # 找到原始数据集中聚类中心点的数据
            cent_lab = []
            for i in range(class_num):
                item = clu_index[i].item()
                images, label = dataset.__getitem__(item)
                image = images[0].unsqueeze(0)
                if i == 0:
                    clu_cent = image
                else:
                    clu_cent = torch.cat((clu_cent, image), dim=0)
                cent_lab.append(label)
            # print(clu_cent.shape)

            # 找到原始数据集中聚类中心点近邻的数据
            NN_lab = []
            for i in range(class_num):
                for j in range(args.N):
                    item = index_NN[i, j].item()
                    images, label = dataset.__getitem__(item)
                    image = images[0].unsqueeze(0)
                    if i == 0 and j == 0:
                        cent_NN = image
                    else:
                        cent_NN = torch.cat((cent_NN, image), dim=0)
                    NN_lab.append(label)
            # bestACC = ACC
        # # print(cent_NN.shape)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # train for one epoch
        train(data_loader, model, criterion_instance, criterion_ins_tance, criterion_clu_cent, optimizer,
              epoch, clu_cent, cent_NN, cent_lab, NN_lab, args)



def train(train_loader, model, criterion_instance, criterion_ins_cent, criterion_clu_cent,
          optimizer, epoch, clu_cent, cent_NN, cent_lab, NN_lab, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # loss_cent = AverageMeter('Loss_cent', ':.4e')
    # loss_aug1 = AverageMeter('Loss_aug1', ':.4e')
    # loss_orl = AverageMeter('Loss_orl', ':.4e')
    loss_batch = AverageMeter('loss_batch', ':.4e')
    loss_all = AverageMeter('loss_all', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [loss_batch, loss_all],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.eval()

    end = time.time()
    # for i, (batch, index) in enumerate(train_loader):
    loss_epoch = 0
    for i, (images, label) in enumerate(train_loader):

        if args.gpu is not None:
            X_cent = clu_cent.cuda(args.gpu, non_blocking=True)
            X_NN = cent_NN.cuda(args.gpu, non_blocking=True)
            x_orl = images[0].cuda(args.gpu, non_blocking=True)
            x_aug1 = images[1].cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            z_cent = model.forward_cluster(X_cent).detach()
            z_nn = model.forward_cluster(X_NN).detach()
            z_orl = model.forward_cluster(x_orl).detach()

        loss = compute_torlence(z_cent, z_orl, z_nn, z_orl, cent_lab, NN_lab, label, args)
        loss_batch.update(loss.item(), images[0].size(0))

        if i % args.print_freq == 0:
            progress.display(i)

#    计算增强和聚类中心与其近邻之间的损失 ins_U_cluloss

def compute_torlence(z_cent, z_orl, z_nn, z_i, cent_lab, NN_lab, orl_lab, args):
    # compute logits
    sim_orl_cent = torch.einsum('cn,bn->cb', [z_cent, z_orl])
    max_center = torch.argmax(sim_orl_cent, dim=0)
    # 计算z_i所属于的类别
    sim_cent = torch.einsum('cn,bn->cb', [z_cent, z_i])
    # print(max_center, max_center.shape)

    # 根据所属类别寻找正负类
    c, n = z_cent.size()
    b, n = z_i.size()
    k, n = z_nn.size()
    mask_pos = construct_mask(c, b, k, max_center, cent_lab, NN_lab, orl_lab, args).detach().to(args.gpu)
    mask_one = torch.ones_like(mask_pos).detach().to(args.gpu)
    sim_nn = torch.einsum('bn,kn->bk', [z_i, z_nn])
    sim_all = torch.cat((sim_cent.T, sim_nn), dim=1)

    loss = (sim_all * mask_pos).sum(1) / (args.N + 1)


    return loss.mean()

def construct_mask(c, b, k, max_center, cent_lab, NN_lab, orl_lab, args):
    mask_cent = torch.zeros([c, c]).to(args.gpu)
    mask_nn = torch.zeros([k, k]).to(args.gpu)
    d = int(k / c)
    replace = torch.ones(d).to(args.gpu)
    replace_one = torch.ones(1).to(args.gpu)
    flag = 0
    for i in max_center:
        if cent_lab[i] == orl_lab[i]:
            logits_cent = torch.scatter(mask_cent[i, :], 0, i, replace_one).unsqueeze(0)
        else:
            logits_cent = mask_cent[i, :].unsqueeze(0)

        index = torch.arange(int(i) * d, (int(i) + 1) * d).to(args.gpu)
        for j in range(int(i) * d, (int(i) + 1) * d):
            if orl_lab[i] != NN_lab[j]:
                # print(j)
                index = torch.cat((index[:j], index[j+1:]), dim=0)
        logits_nn = torch.scatter(mask_nn[i, :], 0, index, replace).unsqueeze(0)
        if flag == 0:
            mask_cent_new = logits_cent
            mask_nn_new = logits_nn
        else:
            mask_cent_new = torch.cat((mask_cent_new, logits_cent), dim=0)
            mask_nn_new = torch.cat((mask_nn_new, logits_nn), dim=0)
        flag += 1
    max_all = torch.cat((mask_cent_new, mask_nn_new), dim=1)
    # print(max_all)
    return max_all




if __name__ == '__main__':
    main()


''' CIFAR10 : stage-2 200epoch
        t=0.2  0.90654 0.87
        t=0.3  0.53068 0.85
        t=0.5  0.69604 0.87
        t=1.0  0.74920 0.88
        
    ImageNet-10 : stage-2 200epoch
        t=0.2  0.59710  5.8906      0.897
        t=0.3  0.90654              0.878
        t=0.5  0.83966              0.868
        t=1.0  0.84366              0.884
'''