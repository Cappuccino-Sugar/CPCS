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
from loss import contrastive_loss
from utils import save_checkpoint, yaml_config_hook, accuracy, adjust_learning_rate
from utils.load_model_weights import load_model_weights
from tools import AverageMeter, ProgressMeter
from moco.resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from moco.resnet_stl import resnet18_stl, resnet34_stl, resnet50_stl
from moco.resnet_all import resnet18, resnet34, resnet50
from data_loader.CIFAR10 import CIFAR10
from utils.find_positive import find_cluste_center_stl
from utils.collate import collate_custom
from utils.memory_bank import MemoryBank

# import moco.builder
import tools.build_stage2


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  # 创建解析器
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    config = yaml_config_hook("config/main_kmeans.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code
    # prepare data

    # prepare data
    if args.dataset == "STL-10":
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
            transform=transform.Transforms(size=args.image_size),
        )
        cluster_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        instance_dataset = unlabeled_dataset
        class_num = 10
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler_cls = torch.utils.data.distributed.DistributedSampler(cluster_dataset)
        train_sampler_ins = torch.utils.data.distributed.DistributedSampler(instance_dataset)
    else:
        train_sampler_cls = None
        train_sampler_ins = None
        # train_sampler = None

    # dataclu_ins = torch.utils.data.DataLoader(
    #     instance_dataset,
    #     batch_size=500,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     # sampler=train_sampler,
    #     # collate_fn = collate_custom
    # )
    dataclu_clu = torch.utils.data.DataLoader(
        cluster_dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
        # sampler=train_sampler,
        # collate_fn = collate_custom
    )

    cluster_data_loader = torch.utils.data.DataLoader(
        cluster_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler_cls is None),
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler_cls
    )

    instance_data_loader = torch.utils.data.DataLoader(
        instance_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler_ins is None),
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler_ins
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

    model = tools.build_stage2.MoCo(
        base_model, class_num,
        args.moco_dim, args.moco_t, args.mlp, cluster=False)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_instance = contrastive_loss.Instance_bantch(args.batch_size, args.moco_t, args.gpu).cuda(args.gpu)
    # # criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.moco_t, args.gpu).cuda(args.gpu)

    criterion_clu_cent = contrastive_loss.clu_cent(class_num, args.moco_t, args.gpu).cuda(args.gpu)
    criterion_ins_tance = contrastive_loss.ins_U_cluloss(class_num, args.moco_t, args.gpu).cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.pretrained is not None:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.pth.tar".format(args.resume))
        load_model_weights(model, model_fp)


    cudnn.benchmark = True

    # creat memorybank
    with torch.no_grad():
        # memory_bank_base_ins = MemoryBank(len(instance_dataset),
        #                           model.module.encoder_q.layer4[2].conv1.weight.shape[0],
        #                           class_num)
        # memory_bank_base_ins.cuda()

        memory_bank_base_clu = MemoryBank(len(cluster_dataset),
                                        model.module.encoder_q.layer4[2].conv1.weight.shape[0],
                                        class_num)
        memory_bank_base_clu.cuda()
        print("=> Initializing memorybank ")


    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler_cls.set_epoch(epoch)
            train_sampler_ins.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # # creat memorybank
        # with torch.no_grad():
        #     memory_bank_base = MemoryBank(len(dataset),
        #                                   model.module.encoder_q.layer4[2].conv1.weight.shape[0],
        #                                   class_num)
        #     memory_bank_base.cuda()
        #     print("=> Initializing memorybank ")

        # clu_cen, clu_index, cent_NN, index_NN = find_cluste_center(dataloader_for_clu, model, memory_bank_base,
        #                                                            class_num, args)

        # # dynamic k-means
        # print("=> Initializing cluster center index")
        #
        # clu_index_ins, index_NN_ins, clu_index_clu, index_NN_clu = find_cluste_center_stl(dataclu_ins, dataclu_clu, model,
        #                                                          memory_bank_base_ins,memory_bank_base_clu, class_num, args)
        clu_index_clu, index_NN_clu = find_cluste_center_stl(dataclu_clu, model, memory_bank_base_clu,
                                                         class_num, args)
        print(clu_index_clu)
        # print("=> Initializing cluster center index")
        # if bestACC < ACC:
        print("=> set cluster center and NN ")
        # 找到原始数据集中聚类中心点的数据
        for i in range(class_num):
            item_clu = clu_index_clu[i].item()
            # item_ins = clu_index_ins[i].item()
            images_clu, _ = cluster_dataset.__getitem__(item_clu)
            # images_ins, _ = instance_dataset.__getitem__(item_ins)
            image_clu = images_clu[0].unsqueeze(0)
            # image_ins = images_ins[0].unsqueeze(0)
            if i == 0:
                clu_cent_clu = image_clu
                # clu_cent_ins = image_ins
            else:
                clu_cent_clu = torch.cat((clu_cent_clu, image_clu), dim=0)
                # clu_cent_ins = torch.cat((clu_cent_ins, image_ins), dim=0)
            # print(clu_cent.shape)

            # 找到原始数据集中聚类中心点近邻的数据
        for i in range(class_num):
            for j in range(args.N):
                item_clu = index_NN_clu[i, j].item()
                # item_ins = index_NN_ins[i, j].item()
                images_clu, _ = cluster_dataset.__getitem__(item_clu)
                # images_ins, _ = instance_dataset.__getitem__(item_ins)
                image_clu = images_clu[0].unsqueeze(0)
                # image_ins = images_ins[0].unsqueeze(0)
                if i == 0 and j == 0:
                    cent_NN_clu = image_clu
                    # cent_NN_ins = image_ins
                else:
                    cent_NN_clu = torch.cat((cent_NN_clu, image_clu), dim=0)
                    # cent_NN_ins = torch.cat((cent_NN_ins, image_ins), dim=0)
        # bestACC = ACC
        # # print(cent_NN.shape)

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # train for one epoch
        train(cluster_data_loader, instance_data_loader, model, criterion_instance, criterion_ins_tance, criterion_clu_cent, optimizer,
              epoch, clu_cent_clu, cent_NN_clu, args)


        # # init next clu cent
        # init_row = clu_index

        data_time = time.time() - end
        end = time.time()
        if epoch % 20 == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='checkpoint_{}.pth.tar'.format(epoch))
        # print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}\t time: {data_time}")
        print(f"Epoch [{epoch}/{args.epochs}]\t  time: {data_time}")

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='checkpoint_{}.pth.tar'.format(epoch+1))


# train(cluster_data_loader,instance_data_loader, model, criterion_instance, criterion_ins_tance, criterion_clu_cent, optimizer,
#               epoch, clu_cent_clu, clu_cent_ins, cent_NN_clu, cent_NN_ins, args)
def train(cluster_data_loader, instance_data_loader, model, criterion_instance, criterion_ins_cent, criterion_clu_cent, optimizer, epoch, clu_cent, cent_NN, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    loss_cent = AverageMeter('Loss_cent', ':.4e')
    loss_aug1 = AverageMeter('Loss_aug1', ':.4e')
    loss_orl = AverageMeter('Loss_orl', ':.4e')
    loss_batch = AverageMeter('loss_batch', ':.4e')

    progress = ProgressMeter(
        len(cluster_data_loader),
        [loss_cent, loss_batch, loss_aug1, loss_orl, batch_time],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()


    end = time.time()
    # for i, (batch, index) in enumerate(train_loader):

    for i, (images, _) in enumerate(instance_data_loader):
        # print(_)
        # images = batch['image']
        # index = batch['index']
        # measure data loading time
        # data_time.update(time.time() - end)

        if args.gpu is not None:
            # X_cent = clu_cent.cuda(args.gpu, non_blocking=True)
            # X_NN = cent_NN.cuda(args.gpu, non_blocking=True)
            x_aug1 = images[0].cuda(args.gpu, non_blocking=True)
            x_aug2 = images[1].cuda(args.gpu, non_blocking=True)

        _, _, z_i, z_j = model(x_aug1, x_aug1, x_aug1, x_aug2)
        # loss_1 = criterion_clu_cent(z_cent, z_nn)
        # loss_2 = criterion_ins_cent(z_cent, z_orl.detach(), z_nn, z_i)
        # loss_4 = criterion_ins_cent(z_cent, z_orl.detach(), z_nn, z_orl)
        loss = criterion_instance(z_i, z_j)
        # loss = loss_1 + loss_2 + loss_4 + loss_ins_2

        # output, target = model(im_q=images[0], im_k=images[1])
        # loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # loss_cent.update(loss_1.item(), images[0].size(0))
        # loss_aug1.update(loss_2.item(), images[0].size(0))
        # # loss_aug2.update(loss_3.item(), images[0].size(0))
        # loss_orl.update(loss_4.item(), images[0].size(0))
        loss_batch.update(loss.item(), images[0].size(0))

        # top1.update(acc1[0], images[0].size(0))
        # top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    for i, (images, _) in enumerate(cluster_data_loader):
        # print(_)
        # images = batch['image']
        # index = batch['index']
        # measure data loading time
        # data_time.update(time.time() - end)

        if args.gpu is not None:
            X_cent = clu_cent.cuda(args.gpu, non_blocking=True)
            X_NN = cent_NN.cuda(args.gpu, non_blocking=True)
            x_orl = images[0].cuda(args.gpu, non_blocking=True)
            x_aug1 = images[1].cuda(args.gpu, non_blocking=True)

        z_cent, z_nn, z_orl, z_i = model(X_cent, X_NN, x_orl, x_aug1)
        loss_1 = criterion_clu_cent(z_cent, z_nn)
        loss_2 = criterion_ins_cent(z_cent, z_orl.detach(), z_nn, z_i)
        loss_4 = criterion_ins_cent(z_cent, z_orl.detach(), z_nn, z_orl)
        loss_ins_2 = criterion_instance(z_orl, z_i)
        loss = loss_1 + loss_2 + loss_4 + loss_ins_2

        # output, target = model(im_q=images[0], im_k=images[1])
        # loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_cent.update(loss_1.item(), images[0].size(0))
        loss_aug1.update(loss_2.item(), images[0].size(0))
        # loss_aug2.update(loss_3.item(), images[0].size(0))
        loss_orl.update(loss_4.item(), images[0].size(0))
        loss_batch.update(loss_ins_2.item(), images[0].size(0))

        # top1.update(acc1[0], images[0].size(0))
        # top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    main()
