import argparse
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms

import moco.builder
from transfrom import transform
from utils import yaml_config_hook
from visiual.util import GradCAM, show_cam_on_image
import torchvision

from moco.resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from moco.resnet_stl import resnet18_stl, resnet34_stl, resnet50_stl
from moco.resnet_all import resnet18, resnet34, resnet50
import tools.build_stage2

import os


def main():
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # model = models.mobilenet_v3_large(pretrained=False)

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  # 创建解析器
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    config = yaml_config_hook("config/onegpu.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()


    data_transform = transform.Transforms(size=args.image_size, blur=True)

    dataset_dir = os.path.join(args.dataset_dir, "imagenet-10")
    dataset = torchvision.datasets.ImageFolder(
        root=dataset_dir,
        transform=transform.Transforms(size=args.image_size, blur=True),
    )
    class_num = 10
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
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

    target_layers = [model.encoder_q.fc[-1]]

    # load image
    img_path = r"/home/chenjunfen/workspace/Dataset/imagenet-10/n02056570_45.JPEG"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    for i, (images, _) in enumerate(data_loader):

        if args.gpu is not None:
            x_i = images[0].cuda(args.gpu, non_blocking=True)
            x_j = images[1].cuda(args.gpu, non_blocking=True)
        if i == 0:
            break


    # [C, H, W]
    img_tensor_i, img_tensor_j = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor_i = torch.unsqueeze(img_tensor_i, dim=0)
    input_tensor_j = torch.unsqueeze(img_tensor_j, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    # grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = cam(input_tensor_i, input_tensor_j, x_i, x_j)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()