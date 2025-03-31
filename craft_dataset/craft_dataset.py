import numpy as np
from keras.datasets import cifar10, fashion_mnist
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import scipy.io as sio


PATH_DATA = "E:/githubAwesomeCode/1DLTesting/TestSelection/data/svhn/"
def load_svhn():

    train = sio.loadmat(os.path.join(PATH_DATA, 'svhn_train.mat'))
    test = sio.loadmat(os.path.join(PATH_DATA, 'svhn_test.mat'))

    X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])

    y_train = np.reshape(train['y'], (-1,)) - 1
    y_test = np.reshape(test['y'], (-1,)) - 1

    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)


    return X_train, y_train, X_test, y_test

def load_fashion_MNIST(one_hot=True, channel_first = False):

    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    if channel_first:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        # For output, it is important to change number to one-hot vector.
        y_train = np_utils.to_categorical(Y_train, num_classes=10)
        y_test = np_utils.to_categorical(Y_test, num_classes=10)

    return X_train, y_train, X_test, y_test

def load_MNIST(one_hot=True, channel_first=False):

    # Load data,可以不用下载，在keras.datasets包中直接调用
    mnist_path = 'E:\\githubAwesomeCode\\1DLTesting\\1dataset\\deepimportance_mnist_cifar\\mnist.npz'
    mnist_file = np.load(mnist_path)
    X_train, y_train = mnist_file['x_train'], mnist_file['y_train']
    X_test, y_test = mnist_file['x_test'], mnist_file['y_test']
    mnist_file.close()

    # Preprocess dataset
    # Normalization and reshaping of input.
    if channel_first:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        # For output, it is important to change number to one-hot vector.
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test

def craft_datasets(attack,dataset):

    if dataset == 'mnist':
        # 参数设置
        root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
        model_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/model/model_mnist.h5'

        # 判断攻击类型并加载数据
        if attack =='syn':
            x_target = np.load('./imagetrans/mnist_' + attack + '.npy')
        else:
            fgsm_x_target = np.load(root_path + 'adv_file/mnist/ConvNet_mnist_Adv_fgsm.npy')
            bima_x_target = np.load(root_path + 'adv_file/mnist/ConvNet_mnist_Adv_bim-a.npy')
            bimb_x_target = np.load(root_path + 'adv_file/mnist/ConvNet_mnist_Adv_bim-b.npy')

        (X_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype("float32").reshape(-1, 28, 28, 1)
        x_test = (x_test / 255.0)
        print('y_test:', y_test[0],y_test.shape)
        y_test = y_test.reshape((10000, 1))
        print('y_test:', y_test[0], y_test.shape)

        # adv_x = np.vstack((fgsm_x_target, bima_x_target, bimb_x_target))
        # adv_y = np.vstack((y_test, y_test, y_test))
        #
        # size_ratio = [10000, 8000, 6000, 4000, 2000, 0]
        # for s in size_ratio:
        #     # 原始数据集挑选 s 个
        #     origin_lst = np.random.choice(range(10000), replace=False, size=s)
        #     # 对抗样本数据集或者数据增强数据集挑选  10000-s 个
        #     mutated_lst = np.random.choice(range(30000), replace=False, size=10000 - s)
        #
        #     x_dest = np.append(x_test[origin_lst], adv_x[mutated_lst], axis=0)
        #     y_dest = np.append(y_test[origin_lst], adv_y[mutated_lst])
        #
        #     # 将数据集保存在目录下
        #     np.savez(root_path + 'adv_file/mnist/ConvNet_mnist_adv_compound_' + str(s) + '_random3.npz', x_test=x_dest,
        #              y_test=y_dest)

    elif dataset == 'mnist_lenet5':
        # 参数设置
        root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
        model_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/model/mnist_lenet5.h5'

        # 判断攻击类型并加载数据
        if attack == 'syn':
            x_target = np.load('./imagetrans/mnist_' + attack + '.npy')
        else:
            fgsm_x_target = np.load(root_path + 'adv_file/mnist_lenet5/mnist_lenet5_Adv_fgsm.npy')
            bima_x_target = np.load(root_path + 'adv_file/mnist_lenet5/mnist_lenet5_Adv_bim-a.npy')
            bimb_x_target = np.load(root_path + 'adv_file/mnist_lenet5/mnist_lenet5_Adv_bim-b.npy')

        (X_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype("float32").reshape(-1, 28, 28, 1)
        x_test = (x_test / 255.0)
        print('y_test:', y_test[0], y_test.shape)
        y_test = y_test.reshape((10000, 1))
        print('y_test:', y_test[0], y_test.shape)

        adv_x = np.vstack((fgsm_x_target, bima_x_target, bimb_x_target))
        adv_y = np.vstack((y_test, y_test, y_test))

        size_ratio = [10000, 8000, 6000, 4000, 2000, 0]
        for s in size_ratio:
            # 原始数据集挑选 s 个
            origin_lst = np.random.choice(range(10000), replace=False, size=s)
            # 对抗样本数据集或者数据增强数据集挑选  10000-s 个
            mutated_lst = np.random.choice(range(30000), replace=False, size=10000 - s)

            x_dest = np.append(x_test[origin_lst], adv_x[mutated_lst], axis=0)
            y_dest = np.append(y_test[origin_lst], adv_y[mutated_lst])

            # 将数据集保存在目录下
            np.savez(root_path + 'adv_file/mnist_lenet5/mnist_lenet5_adv_compound_' + str(s) + '_random1.npz', x_test=x_dest,
                     y_test=y_dest)

    elif dataset == "cifar10":

        root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
        CLIP_MAX = 0.5

        # 判断攻击类型并加载数据
        if attack == 'syn':
            x_target = np.load('./imagetrans/cifar_' + attack + '.npy')
            x_target = x_target.astype("float32")
            x_target = (x_target / 255.0) - (1.0 - CLIP_MAX)
        else:
            fgsm_x_target = np.load( root_path + 'adv_file/cifar_vgg/cifar10_vgg_Adv_cifar_fgsm.npy' )
            bima_x_target = np.load(root_path + 'adv_file/cifar_vgg/cifar10_vgg_Adv_cifar_bim-a.npy')
            bimb_x_target = np.load(root_path + 'adv_file/cifar_vgg/cifar10_vgg_Adv_cifar_bim-b.npy')

        # 加载原始数据集
        (_, _), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
        print('y_test.shape', y_test.shape)  #(10000,1)

        adv_x = np.vstack((fgsm_x_target, bima_x_target, bimb_x_target))
        adv_y = np.vstack((y_test, y_test, y_test))

        size_ratio = [10000, 8000, 6000, 4000, 2000, 0]
        for s in size_ratio:
            # 原始数据集挑选 s 个
            origin_lst = np.random.choice(range(10000), replace=False, size= s)
            # 对抗样本数据集或者数据增强数据集挑选  10000-s 个
            mutated_lst = np.random.choice(range(30000), replace=False, size= 10000 - s)

            x_dest = np.append(x_test[origin_lst], adv_x[mutated_lst], axis=0)
            y_dest = np.append(y_test[origin_lst], adv_y[mutated_lst])

            #将数据集保存在目录下
            np.savez(root_path + 'adv_file/cifar_vgg/cifar_vgg_adv_compound_'+str(s)+'_random1.npz', x_test=x_dest, y_test=y_dest)
    #
    elif dataset == 'fashion':

        root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'

        # 判断攻击类型并加载数据
        if attack =='syn':
            x_target = np.load('./imagetrans/mnist_' + attack + '.npy')
        else:
            fgsm_x_target = np.load(root_path + 'adv_file/fashion/ConvNet_Adv_fashion_bim-a.npy')
            bima_x_target = np.load(root_path + 'adv_file/fashion/ConvNet_Adv_fashion_bim-b.npy')
            bimb_x_target = np.load(root_path + 'adv_file/fashion/ConvNet_Adv_fashion_fgsm.npy')

        # 加载原始数据集
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.astype("float32").reshape(-1, 28, 28, 1)
        x_test = (x_test / 255.0)
        y_test = y_test.reshape((10000, 1))

        adv_x = np.vstack((fgsm_x_target, bima_x_target, bimb_x_target))
        adv_y = np.vstack((y_test, y_test, y_test))

        size_ratio = [10000, 8000, 6000, 4000, 2000, 0]
        for s in size_ratio:
            # 原始数据集挑选 s 个
            origin_lst = np.random.choice(range(10000), replace=False, size=s)
            # 对抗样本数据集或者数据增强数据集挑选  10000-s 个
            mutated_lst = np.random.choice(range(30000), replace=False, size=10000 - s)

            x_dest = np.append(x_test[origin_lst], adv_x[mutated_lst], axis=0)
            y_dest = np.append(y_test[origin_lst], adv_y[mutated_lst])

            # 将数据集保存在目录下
            np.savez(root_path + 'adv_file/svhn/ConvNet_fashion_adv_compound_' + str(s) + '_random3.npz', x_test=x_dest,
                     y_test=y_dest)

    elif dataset == 'svhn':

        root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'

        # 判断攻击类型并加载数据
        if attack =='syn':
            x_target = np.load('./imagetrans/mnist_' + attack + '.npy')
        else:
            fgsm_x_target = np.load(root_path + 'adv_file/svhn/ConvNet_svhn_Adv_bim-a.npy')
            bima_x_target = np.load(root_path + 'adv_file/svhn/ConvNet_svhn_Adv_bim-b.npy')
            bimb_x_target = np.load(root_path + 'adv_file/svhn/ConvNet_svhn_Adv_fgsm.npy')

        # 加载原始数据集
        X_train, Y_train, X_test, Y_test = load_svhn()
        Y_test = Y_test.reshape((26032, 1))

        adv_x = np.vstack((fgsm_x_target, bima_x_target, bimb_x_target))
        adv_y = np.vstack((Y_test, Y_test, Y_test))

        size_ratio = [10000, 8000, 6000, 4000, 2000, 0]
        for s in size_ratio:
            reals = round(s*0.0001*26032)
            # 原始数据集挑选 s 个
            origin_lst = np.random.choice(range(26032), replace=False, size=reals)
            # 对抗样本数据集或者数据增强数据集挑选  10000-s 个
            mutated_lst = np.random.choice(range(78096), replace=False, size=26032 - reals)

            x_dest = np.append(X_test[origin_lst], adv_x[mutated_lst], axis=0)
            y_dest = np.append(Y_test[origin_lst], adv_y[mutated_lst])

            # 将数据集保存在目录下
            np.savez(root_path + 'adv_file/svhn/ConvNet_svhn_adv_compound_' + str(s) + '_random1.npz', x_test=x_dest,
                     y_test=y_dest)

    elif dataset == 'svhn_vgg':

        root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'

        # 判断攻击类型并加载数据
        if attack =='syn':
            x_target = np.load('./imagetrans/mnist_' + attack + '.npy')
        else:
            fgsm_x_target = np.load(root_path + 'adv_file/svhn_vgg/vgg_svhn_Adv_bim-a.npy')
            bima_x_target = np.load(root_path + 'adv_file/svhn_vgg/vgg_svhn_Adv_bim-b.npy')
            bimb_x_target = np.load(root_path + 'adv_file/svhn_vgg/vgg_svhn_Adv_fgsm.npy')

        # 加载原始数据集
        X_train, Y_train, X_test, Y_test = load_svhn()
        Y_test = Y_test.reshape((26032, 1))

        adv_x = np.vstack((fgsm_x_target, bima_x_target, bimb_x_target))
        adv_y = np.vstack((Y_test, Y_test, Y_test))

        size_ratio = [10000, 8000, 6000, 4000, 2000, 0]
        for s in size_ratio:
            reals = round(s*0.0001*26032)
            # 原始数据集挑选 s 个
            origin_lst = np.random.choice(range(26032), replace=False, size=reals)
            # 对抗样本数据集或者数据增强数据集挑选  10000-s 个
            mutated_lst = np.random.choice(range(78096), replace=False, size=26032 - reals)

            x_dest = np.append(X_test[origin_lst], adv_x[mutated_lst], axis=0)
            y_dest = np.append(Y_test[origin_lst], adv_y[mutated_lst])

            # 将数据集保存在目录下
            np.savez(root_path + 'adv_file/svhn_vgg/svhn_vgg_adv_compound_' + str(s) + '_random3.npz', x_test=x_dest,
                     y_test=y_dest)


if __name__ == "__main__":
    attack = 'adv'
    dataset = "cifar10"
    craft_datasets( attack, dataset )