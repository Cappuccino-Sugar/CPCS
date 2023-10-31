import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import os



def get_shuffle_classes(labels, features, num, class_num):
    names = locals()
    range_raw = len(labels) // class_num
    print('labels', len(labels))
    print('feature',len(features))
    print('class_num',class_num)
    print('num',num)
    print('range',range_raw)
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

    # c_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # c_list = ['b', 'orange', 'navy', 'y', 'm', 'c', 'k', 'aqua', 'g', 'c', 'peru', 'lightgreen', 'khaki', 'deeppink',
    #           'olive']
    c_list = ['b', 'orange', 'navy', 'y', 'm', 'c', 'k', 'indigo', 'pink', 'yellow']
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
        plt.scatter(tsnex, tsney, label=classes[i], c=c, s=10)

    plt.legend(loc="upper left")
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.axis('off')
    # plt.title('CIFAR-10')

    plt.savefig(plot_path)
    plt.show()


def t_sne_main(labels_true_path, features_path, class_number, class_names, source_dir, balance=False, each_class_point=1):
    # features = values_save_and_load.load_values(features_path)
    # labels = values_save_and_load.load_values(labels_true_path)
    features = features_path
    print(f'main中features的shape：{features.shape}')
    labels = labels_true_path
    labels = np.array(labels, dtype=np.int)
    if balance:
        labels_new, features_new = get_shuffle_classes(labels, features, each_class_point, class_number)
        labels, features = labels_new, features_new
    data = np.reshape(features, (len(features), -1))
    t_sne_plot(os.path.join(source_dir, '{}/{}_{}.svg'.format(name, 'stage1', 200)), data, labels, class_names, class_number)
    print('t_sne plot has finished!')


if __name__ == '__main__':
    source_dir = r'/home/chenjunfen/workspace/WWJ/MoCo/feature/'
    name = "ImageNet-10"
    # name = 'MNIST'
    labels_true_path = os.path.join(source_dir, '{}/{}_Ytrue_{}.npy'.format(name, 'stage2', 200))
    features_path = os.path.join(source_dir, '{}/{}_Feature_{}.npy'.format(name, 'stage2', 200))
    labels_true_path = np.load(labels_true_path)
    features_path = np.load(features_path)
    # source_dir = r'E:\Visual_Max\feature-F\CIFAR-10'
    class_names = [str(i) for i in range(10)]
    class_number = len(class_names)
    # class_names = [str(i) for i in range(10)]
    t_sne_main(source_dir=source_dir,
               labels_true_path=labels_true_path,
               features_path=features_path,
               class_number=class_number,
               class_names=class_names,
               balance=True,
               each_class_point=1000)


