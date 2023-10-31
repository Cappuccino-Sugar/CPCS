import math

import numpy as np
import torch
import time
import copy
from utils.km_for_tensor import KMEANS
from sklearn.cluster import KMeans
from evaluation import evaluation
# find cluster_center

def find_cluste_center(loader, model, memory_bank_base, clus_num, args):
    """
    Args:
        dataset -> numpy
    Returns:
        index of cluster_center -> int
    """
    target_vector = []
    model.eval()
    N = args.N
    memory_bank_base.reset()
    # for step, ((x, x_), y) in enumerate(loader):
    # for i, (images, traget, index) in enumerate(loader):
    #     if args.gpu is not None:
    #         x = images[0].cuda(args.gpu, non_blocking=True)
    #     with torch.no_grad():
    #         c = model.forward_cluster_z(x)
    #     c = c.detach()
    #     feature_vector.extend(c.cpu().detach().numpy())
    #     index_vector.extend(index.numpy())
    #     # if step % 20 == 0:
    #     #     print(f"Step [{step}/{len(loader)}]\t Computing features...")
    # feature_vector = np.array(feature_vector)
    # index_vector = np.array(index_vector)
    # print("Features shape {}".format(feature_vector.shape))
    # # return center and index of cluster , TNN of center and index
    # clu_cen, clu_index, TNN, TNN_Iindex = K_means(feature_vector, index_vector)
    # return clu_cen, clu_index, TNN, TNN_Iindex
    end = time.time()
    feature_vector = []
    labels_vector = []
    # for i, (images, target, index) in enumerate(loader):
    for i, (images, target) in enumerate(loader):
        if args.gpu is not None:
            x = images[0].cuda(args.gpu, non_blocking=True).half()
        with torch.no_grad():
            output = model.forward_cluster(x)
        output = output.detach()
        memory_bank_base.update(output)
        feature_vector.extend(output.cpu().detach().numpy())
        labels_vector.extend(target.numpy())

        # memory_bank_base.update(c, index)
        if i % 20 == 0:
            data_time = time.time() - end
            end = time.time()
            print(f"===>fill memorybank [{i}/{len(loader)}]\t time: {data_time}")
        # feature_vector = torch.cat((feature_vector, c), dim=0)
        # index_vector = torch.cat((index_vector, index), dim=0)
        # feature_vector.extend(c.cpu().detach().numpy())
        # target_vector.extend(target.numpy())

        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")
    # feature_vector = np.array(feature_vector)
    X = np.array(feature_vector)
    Y = np.array(labels_vector)
    features = memory_bank_base.get()
    # Y = np.array(labels_vector)
    # print("Features shape {}, divice{}".format(feature_vector.shape, feature_vector.device))
    # return center and index of cluster , TNN of center and index

    # 使用sklean寻找中心点
    # x = np.array(feature_vector)
    k = KMeans(n_clusters=clus_num).fit(X)
    # print(k.cluster_centers_, k.labels_, k.inertia_)
    clu_center = torch.from_numpy(k.cluster_centers_).to(memory_bank_base.device).half()
    sim = torch.matmul(clu_center, features.T) / args.moco_t
    argsim_all, index_all = torch.sort(sim, dim=1, descending=True)
    clu_index_1 = index_all[:, 0]
    index_NN_1 = index_all[:, 1:N+1]

    Y_ = k.predict(X)
    # zero = torch.zeros([memory_bank_base[0], clus_num])
    # # Y_ = k.predict(X)
    # l = np.int64(Y_)
    # # print(l)
    # # a = torch.tensor(Y_[0]).unsqueeze(1)
    # entorpy = zero.scatter_(1, torch.tensor(l).unsqueeze(1), 1)
    # entorpy = entorpy.sum(0).view(-1)
    # entorpy /= entorpy.sum()
    # # print(entorpy)
    #
    # # print(math.log(entorpy.size(0)) + (entorpy * torch.log(entorpy)))
    # ne_i = math.log(entorpy.size(0)) + (entorpy * torch.log(entorpy)).sum()
    # Y_ = X
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
    nmi, ari, f, acc1 = evaluation.evaluate(Y, Y_)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC1 = {:.4f}'.format(nmi, ari, f, acc1))


    # # 使用GPU找中心点
    # # 使用K-means函数找聚类中心，
    # print("=> find cluster center")
    # k_means = KMEANS(clus_num, device=feature_vector.device)
    # k_means.fit(feature_vector, init_row)
    # clu_cen_2, clu_index, Y_ = k_means.get_cent_sample(feature_vector)
    #
    # data_time = time.time() - end
    # end = time.time()
    # print(f"=> find cluster center need time: {data_time}")
    #
    # # 计算datasets当前ACC
    # Y_ = np.array(Y_.cpu())
    # nmi, ari, f, acc2 = evaluation.evaluate(Y, Y_)
    # print('=> NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc2))
    # # print(clu_cen, clu_cen.shape, clu_index)
    #
    # # 寻找聚类中心点的N个近邻
    # sim = torch.matmul(clu_cen_2, feature_vector.T) / args.moco_t
    # argsim_all, index_all = torch.sort(sim, dim=1, descending=True)
    # index_NN_2 = index_all[:, 1:N+1]
    # # TNN = argsim_all[:, 1:N+1]
    # # TNN, index_NN = torch.topk(sim, args.N, dim=1)
    #
    # # TNN, TNN_index = torch.argmax(clu_cen * feature_vector[clu_index])
    # # print(f"=> return memorybank clu_cen: {clu_cen.shape}\t clu_index: {clu_index.shape}\t TNN {TNN.shape}\t "
    # #       f"TNN_index {index_NN.shape}")
    # # l3 = np.array(index_NN.to('cpu').detach().numpy())
    # # print(l3)
    # # return clu_cen, clu_index, TNN, index_NN


    index_NN = index_NN_1
    clu_index = clu_index_1

    # if acc1 >= acc2:
    #     index_NN = index_NN_1
    #     clu_index = clu_index_1
    # else:
    #     index_NN = index_NN_2
    #     clu_index = clu_cen_2


    # return clu_index, index_NN, acc1
    return clu_index, index_NN


def valid_z(data_loader, model, args, class_num):
    X, Y = validate(data_loader, model, args)

    k = KMeans(n_clusters=class_num).fit(X)
    Y_ = k.predict(X)
    # np.save('feature/Cifar-10/%s_Pred_1000.npy' % name, X)
    # np.save('feature/Cifar-10/%s_Feature_1000.npy' % name, z)
    # np.save('feature/Cifar-10/%s_TrueL_1000' % name, Y)
    # Y_ = X

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
            # images = images.cuda(args.gpu, non_blocking=True)
            images = images[0].cuda(args.gpu, non_blocking=True).half()
        # target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model.forward_cluster_z(images)
        output = output.detach()
        feature_vector.extend(output.cpu().detach().numpy())
        labels_vector.extend(target.numpy())
        if i % 40 == 0:
            print(f"Step [{i}/{len(val_loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def valid_c(data_loader, model, args, class_num):
    X, Y = validate_cluster(data_loader, model, args)

    # k = KMeans(n_clusters=class_num).fit(X)
    # Y_ = k.predict(X)
    Y_ = X
    # np.save('feature/Cifar-10/%s_Pred_1000.npy' % name, X)
    # np.save('feature/Cifar-10/%s_Feature_1000.npy' % name, z)
    # np.save('feature/Cifar-10/%s_TrueL_1000' % name, Y)
    # Y_ = X

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
    print('cluster ---> NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))


def validate_cluster(val_loader, model, args):

    # switch to evaluate mode
    model.eval()
    feature_vector = []
    labels_vector = []
    for i, (images, target) in enumerate(val_loader):
        if args.gpu is not None:
            # images = images.cuda(args.gpu, non_blocking=True)
            images = images[0].cuda(args.gpu, non_blocking=True).half()
        # target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model.forward_cluster_c(images)
        output = output.detach()
        feature_vector.extend(output.cpu().detach().numpy())
        labels_vector.extend(target.numpy())
        if i % 40 == 0:
            print(f"Step [{i}/{len(val_loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector