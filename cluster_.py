#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import warnings
import numpy as np
from evaluation import evaluation
from utils import yaml_config_hook
from sklearn.cluster import KMeans

import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils import data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from transfrom import transform
from moco.resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from moco.resnet_stl import resnet18_stl, resnet34_stl, resnet50_stl
from moco.resnet_all import resnet18, resnet34, resnet50

import moco.loader
import moco.builder

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  # 创建解析器
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    config = yaml_config_hook("config/main_cluster.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args = parser.parse_args()

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
    global best_acc1
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
        if args.dataset == "CIFAR-10":
            train_dataset = torchvision.datasets.CIFAR10(
                root=args.dataset_dir,
                train=True,
                download=True,
                transform=transform.Transforms(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=args.dataset_dir,
                train=False,
                download=True,
                transform=transform.Transforms(size=args.image_size).test_transform,
            )
            dataset = data.ConcatDataset([train_dataset, test_dataset])
            # dataset = test_dataset
            class_num = 10
        elif args.dataset == "CIFAR-100":
            train_dataset = torchvision.datasets.CIFAR100(
                root=args.dataset_dir,
                download=True,
                train=True,
                transform=transform.Transforms(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=args.dataset_dir,
                download=True,
                train=False,
                transform=transform.Transforms(size=args.image_size).test_transform,
            )
            dataset = data.ConcatDataset([train_dataset, test_dataset])
            # dataset = test_dataset
            class_num = 20
        elif args.dataset == "STL-10":
            train_dataset = torchvision.datasets.STL10(
                root=args.dataset_dir,
                split="train",
                download=True,
                transform=transform.Transforms(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.STL10(
                root=args.dataset_dir,
                split="test",
                download=True,
                transform=transform.Transforms(size=args.image_size).test_transform,
            )
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
            class_num = 10
        elif args.dataset == "ImageNet-10":
            dataset_dir = os.path.join(args.dataset_dir, "imagenet-10")
            dataset = torchvision.datasets.ImageFolder(
                root=dataset_dir,
                transform=transform.Transforms(size=args.image_size, blur=True).test_transform,
            )
            class_num = 10
        elif args.dataset == "ImageNet-dogs":
            dataset_dir = os.path.join(args.dataset_dir, "ImageNet-dogs")
            dataset = torchvision.datasets.ImageFolder(
                root=dataset_dir,
                transform=transform.Transforms(size=args.image_size, blur=True).test_transform,
            )
            class_num = 15
        elif args.dataset == "tiny-ImageNet":
            dataset_dir = os.path.join(args.dataset_dir, "tiny-imagenet-200/train")
            dataset = torchvision.datasets.ImageFolder(
                root=dataset_dir,
                transform=transform.Transforms(s=0.5, size=args.image_size).test_transform,
            )
            class_num = 200
        else:
            raise NotImplementedError
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=200,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
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

    model = moco.builder.MoCo(
        base_model, class_num,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)



    # # load from pre-trained, before DistributedDataParallel constructor
    # if args.pretrained:
    #     if os.path.isfile(args.pretrained):
    #         print("=> loading checkpoint '{}'".format(args.pretrained))
    #         checkpoint = torch.load(args.pretrained, map_location="cpu")
    #
    #         # rename moco pre-trained keys
    #         state_dict = checkpoint['state_dict']
    #         for k in list(state_dict.keys()):
    #             # retain only encoder_q up to before the embedding layer
    #             if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    #                 # remove prefix
    #                 state_dict[k[len("module.encoder_q."):]] = state_dict[k]
    #             # delete renamed or unused k
    #             del state_dict[k]
    #
    #         args.start_epoch = 0
    #         msg = model.load_state_dict(state_dict, strict=False)
    #         assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    #
    #         print("=> loaded pre-trained model '{}'".format(args.pretrained))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.pretrained))

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
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # # optimize only the linear classifier
    # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
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



    # if args.evaluate:
    #     validate(val_loader, model, args)
    #     return
    # evaluate on validation set
    X, Y = validate(data_loader, model, args)

    # k = KMeans(n_clusters=class_num).fit(X)
    # Y_ = k.predict(X)
    Y_ = X


    if args.dataset == "CIFAR-100":  # super-class
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
        ]
        Y_copy = copy.copy(Y)
        for i in range(20):
            for j in super_label[i]:
                Y[Y_copy == j] = i
    nmi, ari, f, acc = evaluation.evaluate(Y, Y_)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))


def validate(val_loader, model, args):

    # switch to evaluate mode
    model.eval()
    feature_vector = []
    labels_vector = []
    for i, (images, target) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model.module.forward_cluster_c(images)
        output = output.detach()
        feature_vector.extend(output.cpu().detach().numpy())
        labels_vector.extend(target.numpy())
        if i % 20 == 0:
            print(f"Step [{i}/{len(val_loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

if __name__ == '__main__':
    main()
