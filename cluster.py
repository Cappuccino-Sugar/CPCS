import os
import argparse
import torch
import torchvision
import numpy as np
from utils import yaml_config_hook
# from modules import resnet, network, transform
from transfrom import transform
from evaluation import evaluation
from torch.utils import data
from sklearn.cluster import KMeans
import copy
from moco.resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from moco.resnet_stl import resnet18_stl, resnet34_stl, resnet50_stl
from moco.resnet_all import resnet18, resnet34, resnet50

import moco.loader
import moco.builder
import tools.build_stage2
import matplotlib.pyplot as plt


# from sklearn.manifold import TSNE

# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#

def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    #     for step, ((x, x_), y) in enumerate(loader):
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/main_cluster.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            transform=transform.Transforms_for2(size=args.image_size, blur=True).test_transform,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset_dir = os.path.join(args.dataset_dir, "imagenet-dogs")
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir,
            transform=transform.Transforms_for2(size=args.image_size, blur=True).test_transform,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset_dir = os.path.join(args.dataset_dir, "tiny-imagenet-200/train")
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir,
            transform=transform.Transforms_for2(s=0.5, size=args.image_size).test_transform,
        )
        class_num = 200
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
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

    # model = moco.builder.MoCo(
    #     base_model, class_num,
    #     args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, cluster=False)
    model = tools.build_stage2.MoCo(
        base_model, class_num,
        args.moco_dim, args.moco_t, args.mlp, cluster=False)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.pth.tar".format(args.start_epoch))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['state_dict'])
    # model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(torch.load(model_fp, map_location=device.type))
    # if args.reload:
    #     model_fp = os.path.join(args.model_path, "checkpoint_{}.pth.tar".format(args.resume))
    #     print("=> Initializing model '{}'".format(model_fp))
    #     pre_model = torch.load(model_fp, map_location=device)
    #     # rename moco pre-trained keys
    #     print("=> Initializing feature model '{}'".format(model_fp))
    #     state_dict = pre_model['state_dict']
    #     for k in list(state_dict.keys()):
    #         # Initialize the feature module with encoder_q of moco.
    #         if k.startswith('module.encoder_q'):
    #             # remove prefix
    #             state_dict["encoder_q.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
    #         # delete renamed or unused k
    #         # if not k.startswith('module.encoder_q'):
    #         del state_dict[k]
    #     msg = model.load_state_dict(state_dict, strict=False)
    #     # msg = model.load_state_dict(state_dict)
    #     print(msg)

    model.to(device)

    print("### Creating features from model ###")
    X, Y = inference(data_loader, model, device)

    feature = 'feature/{}/'.format(args.dataset)
    # Yture = 'feature/{}/'.format(args.dataset, 'stage2', args.start_epoch)

    if not os.path.exists(feature):
        os.makedirs(feature)
    # if not os.path.exists(Yture):
    #     os.makedirs(Yture)
    np.save('feature/{}/{}_Feature_{}.npy'.format(args.dataset, 'stage1', args.resume), X)
    # np.save('feature/Cifar-10/%s_Feature_200.npy' % name, z)
    np.save('feature/{}/{}_Ytrue_{}'.format(args.dataset, 'stage1', args.resume), Y)
    # Y_ = X

    k = KMeans(n_clusters=class_num).fit(X)
    Y_ = k.predict(X)

    # T-SNE
    # x_fea_train = X.to('cpu').data.n
    # umpy()
    # y_train = Y.to('cpu').data.numpy()
    # x_fea_train = X.reshape(-1, 1)
    # y_train = Y
    # lle = TSNE().fit_transform(X=x_fea_train)
    # x1 = (lle[:, 0])
    # x2 = (lle[:, 1])
    # plt.scatter(x1, x2, c=y_train, cmap='tab10')  # tab10
    # plt.savefig(r'D:\studysoftware\py\pytorch\DeepClustering-master\2c.svg', bbox_inches='tight')

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
    # print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))


