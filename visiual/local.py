# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import tools.build_stage2
import argparse
from utils import save_checkpoint, yaml_config_hook
import os
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from moco.resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from moco.resnet_stl import resnet18_stl, resnet34_stl, resnet50_stl
from moco.resnet_all import resnet18, resnet34, resnet50

import tools.build_stage2

global features_grad
def extract(g):
    print('aaa')
    global features_grad
    print('111')
    features_grad = g

def draw_CAM(model,img_path, dataloader, save_path, visual_heatmap=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img).cuda()
    img = img.unsqueeze(0)
    # img = Variable(img).to(device)
    # # 获取模型输出的feature/score
    # #if use_gpu:
    # #    torch.cuda.set_device(CUDA_DEVICE)
    # model = torch.load(weight)
    # model.to(device)
    model.eval()
    # for i, (images, labels) in enumerate(dataloader):
    #     if args.gpu is not None:
    #         images = images.cuda(args.gpu)
    # images = images.cuda(args.gpu)
    x = model.encoder_q.conv1(img)
    x = model.encoder_q.bn1(x)
    # x = model.encoder_q.relu(x)
    x = model.encoder_q.layer1(x)
    x = model.encoder_q.layer2(x)
    x = model.encoder_q.layer3(x)
    x = model.encoder_q.layer4(x)
    features = model.encoder_q.avgpool(x)
    # features = model.avgpool(x)
    l, output, _, a = model(img,img,img,img)
    global features_grad
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    # pred_class.register_hook(extract)
    features.register_hook(extract)
    pred_class.backward() # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(128):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    os.makedirs('imgs', exist_ok=True)

    # plt.imsave(f'imgs/bImg_{idd}_batch_{batch_id}.png',
    #            torch.cat((imgg, heatmap * imgg), dim=3).squeeze(0).cpu().permute(
    #                (1, 2, 0)).clamp(0, 1).numpy().astype(float))

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

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
    # args = parse_option()

    train_dataset, class_num = get_datasets(args)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.workers)

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

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        cudnn.benchmark = True
    # augs = [
    #     transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    # ]
    # trans = transforms.Compose(augs)

    img_path = r'E:\workspace\moco-new\hotiamge\n02128757_215.JPEG'
    save_path = r'E:\workspace\moco-new\hotiamge\123.JPEG'
    draw_CAM(model, img_path, train_loader, save_path, visual_heatmap=True)


def get_datasets(args):
    augs = [
        transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ]
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transforms.Compose(augs),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transforms.Compose(augs),
        )
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset_dir = os.path.join(args.dataset_dir, "imagenet-10")
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir,
            transform=transforms.Compose(augs),
        )
        class_num = 10
    else:
        raise NotImplementedError(args.dataset)
    # dataset = data.ConcatDataset([train_dataset, test_dataset])
    # dataset = test_dataset

    return dataset, class_num

if __name__ == '__main__':
    main()
