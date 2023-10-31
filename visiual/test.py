# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
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
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from visiual.ShowGradCam import ShowGradCam
from visiual.losses import Instance_bantch
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

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img,(32, 32))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input

def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec
def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  # 创建解析器
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=1, type=int,
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

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    # Data loading code
    # prepare data
    if args.dataset == "CIFAR-10":
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
        class_num = 20
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
        class_num = 200
    else:
        raise NotImplementedError

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

    gradCam = ShowGradCam(model.conv2)

    model.train()
    for i, (images, _) in enumerate(data_loader):
        if args.gpu is not None:
            # X_cent = clu_cent.cuda(args.gpu, non_blocking=True)
            # X_NN = cent_NN.cuda(args.gpu, non_blocking=True)
            x_orl = images[0].cuda(args.gpu, non_blocking=True)
            x_aug1 = images[1].cuda(args.gpu, non_blocking=True)

        z_cent, z_nn, z_orl, z_i = model(x_orl, x_orl, x_orl, x_aug1)
        # loss_1 = criterion_clu_cent(z_cent, z_nn)
        # loss_2 = criterion_ins_cent(z_cent, z_orl.detach(), z_nn, z_i)
        # loss_4 = criterion_ins_cent(z_cent, z_orl.detach(), z_nn, z_orl)
        # loss_ins_2 = criterion_instance(z_orl, z_i)

        loss = Instance_bantch(z_orl, z_i)
        # forward
        # backward
        net.zero_grad()
        loss.backward()
        gradCam.show_on_img(img)

if __name__ == '__main__':
    main()


    # 图片读取；网络加载


    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)
    net = Net()

    gradCam = ShowGradCam(net.conv2) #............................. def which layer to show

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # save result
    gradCam.show_on_img(img) #.......................... show gradcam on target pic

