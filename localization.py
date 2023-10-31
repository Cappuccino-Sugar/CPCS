from __future__ import print_function

import argparse
import os
from torch.utils import data
import torchvision
from transfrom import transform
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from moco.resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from moco.resnet_stl import resnet18_stl, resnet34_stl, resnet50_stl
from moco.resnet_all import resnet18, resnet34, resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import yaml_config_hook
from loss import contrastive_loss
import tools.build_stage2
from utils.load_model_weights import load_model_weights
from visiual.ShowGradCam import ShowGradCam
# from datasets.breeds import BREEDSFactory
# from models.util import create_model, load_model


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

    # train_dataset, class_num = get_datasets(args)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = tools.build_stage2.MoCo(
        base_model, class_num,
        args.moco_dim, args.moco_t, args.mlp, cluster=False)
    print(model)

    # model = create_model(args.model, class_num, args.only_base, args.head, args.dim)
    # load_model(model, args.model_path, not args.only_base)
    # optionally resume from a checkpoint
    if args.pretrained is not None:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.pth.tar".format(args.resume))
        load_model_weights(model, model_fp)


    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        cudnn.benchmark = True

    gradCam = ShowGradCam(model.encoder_q.layer4)
    criterion_instance = contrastive_loss.Instance_onegpu(args.batch_size, args.moco_t, args.gpu).cuda(args.gpu)


    for i, (images, labels) in enumerate(data_loader):
        if args.gpu is not None:
            img1 = images[0].cuda(args.gpu)
            img2 = images[1].cuda(args.gpu)

        _, _, z_orl, z_aug = model(img1, img1, img1, img2)

        model.zero_grad()
        loss = criterion_instance(z_orl, z_orl)
        loss.backward()

        gradCam.show_on_img(img1)

        if i == 10:
            break

        # def attention_forward(encoder, imgs):
        #     # hard-coded forward because we need the feature-map and not the finalized feature
        #     x = encoder.conv1(imgs)
        #     x = encoder.bn1(x)
        #     x = encoder.relu(x)
        #     x = encoder.maxpool(x)
        #     x = encoder.layer1(x)
        #     x = encoder.layer2(x)
        #     x = encoder.layer3(x)
        #     feats = encoder.layer4(x)
        #     feats_as_batch = feats.permute((0, 2, 3, 1)).contiguous().view((-1, feats.shape[1]))
        #     # reminder: "fc" layer outputs: (feature, class logits)
        #     feats_as_batch = encoder.fc(feats_as_batch)[0]
        #     feats_as_batch = feats_as_batch.view(
        #         (feats.shape[0], feats.shape[2], feats.shape[3], feats_as_batch.shape[1]))
        #     feats_as_batch = feats_as_batch.permute((0, 3, 1, 2))
        #     return feats_as_batch
        #
        # f_q = attention_forward(model, images)
        # visiual(images, f_q, args.batch_size, batch_id=i, img_size=448)



def localization(im_q, f_q, batch_size, batch_id, img_size):
    os.makedirs('imgs', exist_ok=True)
    for idd in range(batch_size):
        aa = torch.norm(f_q, dim=1)
        imgg = im_q[idd] * torch.Tensor([[[0.229, 0.224, 0.225]]]).view(
            (1, 3, 1, 1)).cuda() + torch.Tensor(
            [[[0.485, 0.456, 0.406]]]).view((1, 3, 1, 1)).cuda()
        heatmap = F.interpolate((aa[idd] / aa[0].max()).detach().unsqueeze(0).unsqueeze(0).repeat((1, 3, 1, 1)),
                                [img_size, img_size])
        thresh = 0
        heatmap[heatmap < thresh] = 0
        plt.imsave(f'imgs/bImg_{idd}_batch_{batch_id}.png',
                   torch.cat((imgg, heatmap * imgg), dim=3).squeeze(0).cpu().permute(
                       (1, 2, 0)).clamp(0, 1).numpy().astype(float))



if __name__ == '__main__':
    main()
