
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import torch
import torchvision
import argparse
# from modules import transform, resnet, network
from utils import yaml_config_hook, save_model
from torch.utils import data



def get_shuffle_classes(labels, features, num, class_num):
    names = locals()
    range_raw = len(labels) // class_num
    print(len(labels))
    print(len(features))
    print(class_num)
    print(num)
    print(range_raw)
    # exit()
    num = 8
    for raw in range(class_num):
        raw_n = random.sample(range(range_raw * raw, range_raw * (raw + 1)), num)
        exec('raw{} = raw_n'.format(raw))

    labels_new = []
    features_new = []
    for i in range(class_num):
        raw_n = names.get('raw{}'.format(i))
        for j in raw_n:
            labels_new.append(labels[j])
            features_new.append(features[j])
    return labels_new, features_new


def t_sne_plot(plot_path, features, labels, classes, class_num):
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(features)
    names = locals()  # 获取exec动态命名的变量
    plt.figure(figsize=(7, 6))

    # c_list = ['b', 'orange', 'navy', 'y', 'm', 'purple', 'k', 'aqua', 'g', 'c', 'peru', 'lightgreen', 'khaki', 'deeppink', 'olive']
    c_list = ['black','linen','gray','darkorange','limegreen','tan','lime','wheat','lightcoral','turquoise',
              'rosybrown','indianred','brown','red','tomato','coral','gold','khaki','olive','deepskyblue']
    for i in range(0, class_num):
        # 定义变量
        exec('tsnex{} = []'.format(i))
        exec('tsney{} = []'.format(i))
        # 添加tsne
        for index, j in enumerate(labels):
            if j == i:
                exec('tsnex{}.append(tsne[{}, {}])'.format(j, index, 0))
                exec('tsney{}.append(tsne[{}, {}])'.format(j, index, 1))

        tsnex = names.get('tsnex{}'.format(i))  # tsnex_list
        tsney = names.get('tsney{}'.format(i))  # tsney_list
        c = c_list[i]
        plt.scatter(tsney, tsnex, label=classes[i], c=c, s=10)
        # plt.scatter(tsnex, tsney, c=c, s=10)

    plt.legend(loc="upper left")
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.axis('off')
    # plt.title('t-sne')

    plt.savefig(plot_path)
    plt.show()


def t_sne_main(labels_true_path, features_path, class_number, class_names, source_dir, balance=False, each_class_point=1000):
    features = np.load(features_path)
    labels = np.load(labels_true_path)
    labels = np.array(labels, dtype=np.int)
    # features = features[:10]
    # labels = labels[:10]
    if balance:
        labels_new, features_new = get_shuffle_classes(labels, features, each_class_point, class_number)
        labels, features = labels_new, features_new
    data = np.reshape(features, (len(features), -1))
    t_sne_plot(os.path.join(source_dir, 't_sne.pdf'), data, labels, class_names, class_number)
    print('t_sne plot has finished!')


if __name__ == '__main__':

    source_dir = r'/home/chenjunfen/workspace/WWJ/MoCo/feature/'
    name = "CIFAR-10"
    # name = 'MNIST'
    labels_true_path = os.path.join(source_dir, '{}/{}_Ytrue_{}.npy'.format(name, 'stage2', 200))
    features_path = os.path.join(source_dir, '{}/{}_Feature_{}.npy'.format(name, 'stage2', 200))
    # print(len(labels_true_path))
    # print(len(features_path))
    # exit()
    class_names = [str(i) for i in range(10)]
    class_number = len(class_names)
    t_sne_main(source_dir=source_dir,
               labels_true_path=labels_true_path,
               features_path=features_path,
               class_number=class_number,
               class_names=class_names,
               balance=True,
               each_class_point=10000,
               )


