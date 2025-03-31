from multiprocessing import Pool
from keras.models import load_model, Model
from scipy.spatial import distance
import os
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import keras.backend as K
import numpy as np
from scipy.stats import gaussian_kde
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class SurpriseAdequacy:
    # (model, X_train, layer_names, upper_bound, dataset)
    def __init__(self,  model, train_inputs, layer_names, upper_bound, dataset):

        #self.surprise = surprise
        self.model = model
        self.train_inputs = train_inputs
        self.layer_names = layer_names
        self.upper_bound = upper_bound
        self.n_buckets = 1000
        self.dataset = dataset
        self.save_path='/home/EI_Neurons/cifar10_convnet/dsa/'
        if dataset == 'drive': self.is_classification = False   #处理非分类任务
        else: self.is_classification = True
        self.num_classes = 10
        self.var_threshold = 1e-5


    #sa.test(X_test, approach)
    def test(self, test_inputs, dataset_name, repeat, seed_selection_strategy, instance='dsa'):

        if instance == 'dsa':
            print('dataset_name: ', dataset_name)
            target_dsa = fetch_dsa(self.model, self.train_inputs, test_inputs,
                                   dataset_name, self.layer_names,
                                   self.num_classes, self.is_classification,
                                   self.save_path, self.dataset)

            #每一个 test input都计算出一个DSA值
            print("target_dsa shape: ",target_dsa)
            np.save(self.save_path + 'cifar10_convnet_adapt_NC_{}_{}_repeat.npy'.format(seed_selection_strategy, repeat), target_dsa)
            # coverage = get_sc(np.amin(target_dsa), self.upper_bound, self.n_buckets, target_dsa)

        if instance == 'lsa':
            print(len(test_inputs))
            target_lsa = fetch_lsa(self.model, self.train_inputs, test_inputs,
                                   dataset_name, self.layer_names,
                                   self.num_classes, self.is_classification,
                                   self.var_threshold, self.save_path, self.dataset)

        if instance == 'mdsa':
            print(len(test_inputs))
            target_mdsa = fetch_mdsa(self.model, self.train_inputs, test_inputs,
                                   dataset_name, self.layer_names,
                                   self.num_classes, self.is_classification,
                                   self.var_threshold, self.save_path, self.dataset)
            np.save(self.save_path + '/cifar_convnet_mdsa.npy', target_mdsa)



def fetch_mdsa(model, X_train, x_target, target_name, layer_names, num_classes,is_classification, var_threshold, save_path, dataset):

    prefix = "[" + target_name + "] "
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, X_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset)

    begin_time = time.time()

    class_matrix = {}
    if is_classification:
        for i, label in enumerate(train_pred):
            if label.argmax(axis=-1) not in class_matrix:
                class_matrix[label.argmax(axis=-1)] = []
            class_matrix[label.argmax(axis=-1)].append(i)
        print('yes')
    print(class_matrix.keys())  # dict_keys([5, 0, 4, 1, 9, 2, 3, 6, 7, 8])

    b4_mean_train_ats = {}
    b4_conv_matrix = {}
    for label in range(num_classes):
        train_ats_4_label = train_ats[class_matrix[label]]
        mean_train_ats = np.mean(train_ats_4_label, axis = 0)
        cov_matrix = np.cov(train_ats_4_label, rowvar = False)
        b4_mean_train_ats[label] = mean_train_ats
        b4_conv_matrix[label] = cov_matrix

    print('b4_mean_train_ats:', b4_mean_train_ats)
    mdsa = []
    if is_classification:
        print('start computing mdsa =======')
        for i, at in enumerate(target_ats):
            label = target_pred[i].argmax(axis = -1)
            mean_train_ats = b4_mean_train_ats[label]
            cov_matrix = b4_conv_matrix[label]
            temp_mdsa = distance.mahalanobis(at, mean_train_ats, np.linalg.inv(cov_matrix))
            mdsa.append(temp_mdsa)

    end_time = time.time()
    execution_time = end_time - begin_time
    print('execution_time: ', execution_time)
    return mdsa

def fetch_lsa(model, X_train, x_target, target_name, layer_names, num_classes,is_classification, var_threshold, save_path, dataset):
    """Likelihood-based SA

    Args:
        model (keras model): Subject model.
        X_train (list): Set of training inputs.
        x_target (list): Set of target (test or[] adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: Keyboard args.

    Returns:
        lsa (list): List of lsa for each target input.
    """

    prefix = "[" + target_name + "] "
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, X_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset)

    class_matrix = {}
    if is_classification:
        for i, label in enumerate(train_pred):
            if label.argmax(axis=-1) not in class_matrix:
                class_matrix[label.argmax(axis=-1)] = []
            class_matrix[label.argmax(axis=-1)].append(i)
        print('yes')
    print(class_matrix.keys()) #dict_keys([5, 0, 4, 1, 9, 2, 3, 6, 7, 8])

    kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix,
                                   is_classification, num_classes, var_threshold)

    lsa = []
    print(prefix + "Fetching LSA")
    if is_classification:
        for i, at in enumerate(target_ats):
            label = target_pred[i].argmax(axis=-1)
            #print('label: ', label)
            kde = kdes[label]
            #print('kde:' , kde)
            lsa.append(_get_lsa(kde, at, removed_cols))

    else:
        kde = kdes[0]
        for at in target_ats:
            lsa.append(_get_lsa(kde, at, removed_cols))


    return lsa

def _get_kdes(train_ats, train_pred, class_matrix, is_classification, num_classes, var_threshold):
    """Kernel density estimation

    Args:
        train_ats (list): List of activation traces in training set.
        train_pred (list): List of prediction of train set.
        class_matrix (list): List of index of classes.
        args: Keyboard args.

    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
    """

    #np.transpose 矩阵转置， trans_ats[class_matrix[label]]是标签为label的input的激活迹向量
    # train_ats[class_matrix[label]] shape:  (5947, 12)
    # col_vectors shape:  (12, 5947)
    is_classification =True
    removed_cols = []
    if is_classification:
        for label in range(num_classes):
            col_vectors = np.transpose(train_ats[class_matrix[label]])
            for i in range(col_vectors.shape[0]):
                if (
                        # np.var计算指定数据(数组元素)沿指定轴(如果有)的方差
                    np.var(col_vectors[i]) < var_threshold and i not in removed_cols
                ):
                    removed_cols.append(i)

        kdes = {}
        for label in range(num_classes):
            refined_ats = np.transpose(train_ats[class_matrix[label]])
            refined_ats = np.delete(refined_ats, removed_cols, axis=0)  #(1120, 9665)
            #print('refined_ats: ', refined_ats.shape)

            if refined_ats.shape[0] == 0:
                print("ats were removed by threshold {}".format(var_threshold))
                break
            kdes[label] = gaussian_kde(refined_ats)


    else:
        col_vectors = np.transpose(train_ats)
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < var_threshold:
                removed_cols.append(i)

        refined_ats = np.transpose(train_ats)
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        if refined_ats.shape[0] == 0:
            print("ats were removed by threshold {}".format(var_threshold))
        kdes = [gaussian_kde(refined_ats)]

    print("The number of removed columns: {}".format(len(removed_cols)))

    return kdes, removed_cols

def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    #print('refined_at: ', refined_at.shape)
    #print('-kde.logpdf(np.transpose(refined_at)):', -kde.logpdf(np.transpose(refined_at)))
    return np.asscalar(-kde.logpdf(np.transpose(refined_at)))

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

# fetch_dsa(self.model, self.train_inputs, test_inputs, dataset_name, self.layer_names,self.num_classes, self.is_classification, self.save_path, self.dataset)
def fetch_dsa(model, x_train, x_target, target_name, layer_names, num_classes, is_classification, save_path, dataset):


    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset
    )

    class_matrix = {}  #输入为训练数据集，模型的输出，每个类对应的数据下标
    all_idx = []
    for i, label in enumerate(train_pred):
        if label.argmax(axis=-1) not in class_matrix:  #label.argmax表示返回最大值的索引
            class_matrix[label.argmax(axis=-1)] = []
        class_matrix[label.argmax(axis=-1)].append(i)
        all_idx.append(i)

    dsa = []


    for i, at in enumerate(target_ats):
        label = target_pred[i].argmax(axis=-1)
        #print('train_ats[class_matrix[label]]: ', train_ats[class_matrix[label]].shape, train_ats[class_matrix[label]])
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))])
        dsa.append(a_dist / b_dist)

    return dsa

#  _get_train_target_ats(model, x_train, x_target, target_name, layer_names, num_classes,is_classification, save_path, dataset)
def _get_train_target_ats(model, x_train, x_target, target_name, layer_names,
                          num_classes, is_classification, save_path, dataset):

    saved_train_path = _get_saved_path(save_path, dataset, "train", layer_names)

    if os.path.exists(saved_train_path[0]):
        print("Found saved {} ATs, skip serving".format("train"))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0]) #train_ats:  (60000, 12)
        train_pred = np.load(saved_train_path[1]) #train_pred:  (60000, 10)
        print('train_ats: ', train_ats.shape)
        print('train_pred: ', train_pred.shape)

    else:
        train_ats, train_pred = get_ats_train(
            model,
            x_train,
            "train",
            layer_names,
            num_classes=num_classes,
            is_classification=is_classification,
            save_path=saved_train_path,
        )
        print("train ATs is saved at " + saved_train_path[0])

    saved_target_path = _get_saved_path(save_path, dataset, 'cifar10', layer_names)
    #Team DEEPLRP
    if os.path.exists(saved_target_path[0]):
        print("Found saved {} ATs, skip serving".format('cifar10'))
        # In case target_ats is stored in a disk
        print('saved_target_path: ', saved_target_path[0])
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])
        print('target_ats: ', target_ats.shape)
        print('target_pred: ', target_pred.shape)

    else:
        # target就是X_train
        target_ats, target_pred = get_ats_target(
            model,
            x_target, #X_test
            target_name,
            layer_names,
            num_classes=num_classes,
            is_classification=is_classification,
            save_path=saved_target_path,
        )


    return train_ats, train_pred, target_ats, target_pred

def get_ats_target( model, dataset, name, layer_names, save_path=None, batch_size=128, is_classification=True, num_classes=10, num_proc=10,):

    temp_model = Model(
        inputs=model.input, #Tensor("input_1:0", shape=(None, 28, 28, 1), dtype=float32)
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names], #layer_name 层的神经元激活值
    )

    print("============================")

    if is_classification:
        p = Pool(num_proc)  #Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果池还没有满，就会创建一个新的进程来执行请求。
        pred = model.predict(dataset, batch_size=batch_size, verbose=1) #（50000,10)
        print("pred:",pred.shape)

        final_ats = None
        srange = 1000
        for idx in range(0, 1):
            print('idx: ', str(srange * idx), str(srange * idx + srange))
            train_data = dataset[srange * idx:srange * idx + srange]

            if len(layer_names) == 1: #是否可以理解成，将该层作为一个模型，输入dataset，获取该层的输出。调用新建的“层模型”的predict方法
                layer_outputs = [temp_model.predict(train_data, batch_size=batch_size, verbose=1)]
            else:
                layer_outputs = temp_model.predict(train_data, batch_size=batch_size, verbose=1)
            K.clear_session()


            ats = None

            for layer_name, layer_output in zip(layer_names, layer_outputs):  #(1, 60000, 4, 4, 12)
                print("Layer: " + layer_name)  # block1_conv1
                print('layer_output: ', layer_output.shape)  #(60000, 24, 24, 4)
                if layer_output[0].ndim == 3:
                    # For convolutional layers，卷积层、池化层：ndim=3，flatten层：ndim=1
                    layer_matrix = np.array(p.map(_aggr_output, [layer_output[i] for i in range(len(train_data))]))
                    print('layer_matrix_1: ', layer_matrix.shape, layer_matrix)
                else:
                    layer_matrix = np.array(layer_output)
                    print('layer_matrix_2: ', layer_matrix.shape, layer_matrix)

                if ats is None:
                    ats = layer_matrix
                else:
                    ats = np.append(ats, layer_matrix, axis=1)
                    layer_matrix = None

            if idx == 0:
                final_ats = ats
            else:
                final_ats = np.concatenate((final_ats, ats), axis=0)

    print('final_ats shape: ', final_ats.shape)
    print('final_ats: ', final_ats)

    # if save_path is not None:
    #     np.save(save_path[0], final_ats)
    #     np.save(save_path[1], pred)

    return final_ats, pred

def get_ats_train( model, dataset, name, layer_names, save_path=None, batch_size=128, is_classification=True, num_classes=10, num_proc=10,):

    temp_model = Model(
        inputs=model.input, #Tensor("input_1:0", shape=(None, 28, 28, 1), dtype=float32)
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names], #layer_name 层的神经元激活值
    )

    print("============================")
    prefix = "[" + name + "] "

    if is_classification:
        p = Pool(num_proc)  #Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果池还没有满，就会创建一个新的进程来执行请求。
        print(prefix + "Model serving")
        pred = model.predict(dataset, batch_size=batch_size, verbose=1) #（50000,10)
        print("pred:",pred.shape)

        final_ats = None
        srange = 60000

        for idx in range(0, 1):
            print('idx: ', str(srange * idx), str(srange * idx + srange))
            train_data = dataset[srange * idx:srange * idx + srange]

            if len(layer_names) == 1: #是否可以理解成，将该层作为一个模型，输入dataset，获取该层的输出。调用新建的“层模型”的predict方法
                layer_outputs = [temp_model.predict(train_data, batch_size=batch_size, verbose=1)]
            else:
                layer_outputs = temp_model.predict(train_data, batch_size=batch_size, verbose=1)
            K.clear_session()

            print(prefix + "Processing ATs")
            ats = None

            for layer_name, layer_output in zip(layer_names, layer_outputs):  #(1, 60000, 4, 4, 12)
                print("Layer: " + layer_name)  # block1_conv1
                print('layer_output: ', layer_output.shape)  #(60000, 24, 24, 4)
                if layer_output[0].ndim == 3:
                    # For convolutional layers，卷积层、池化层：ndim=3，flatten层：ndim=1
                    layer_matrix = np.array(p.map(_aggr_output, [layer_output[i] for i in range(len(train_data))]))
                    print('layer_matrix_1: ', layer_matrix.shape, layer_matrix)
                else:
                    layer_matrix = np.array(layer_output)
                    print('layer_matrix_2: ', layer_matrix.shape, layer_matrix)

                if ats is None:
                    ats = layer_matrix
                else:
                    ats = np.append(ats, layer_matrix, axis=1)
                    layer_matrix = None

            if idx == 0:
                final_ats = ats
            else:
                final_ats = np.concatenate((final_ats, ats), axis=0)

    print('final_ats shape: ', final_ats.shape)
    print('final_ats: ', final_ats)


    if save_path is not None:
        np.save(save_path[0], final_ats)
        np.save(save_path[1], pred)

    return final_ats, pred

def _get_saved_path(base_path, dataset, dtype, layer_names):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )

#np.amin()一维数组a中的最小值。self.n_buckets在SADL论文中被默认设置为1000
#(np.amin(target_dsa), self.upper_bound,self.n_buckets, target_dsa)
def get_sc(lower, upper, k, sa):
    """Surprise Coverage

    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        sa (list): List of lsa or dsa.

    Returns:
        cov (int): Surprise coverage.
    """
    #np.linspace 创造等差数列
    #np.digitize 看sa值在哪个区间里。跟kmnc的思想有点类似，统计所有被覆盖到的区间的个数。
    #除以1000（计算了1000次），后面的*100是计算覆盖率的百分比
    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k) * 100

def find_closest_at(at, train_ats):
    #The closest distance between subject AT and training ATs.

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])

#计算1范数的距离，
def find_closest_at_ord1(at, train_ats):

    dist = np.linalg.norm(at - train_ats, ord=1, axis=1)  #二范数
    return (min(dist), train_ats[np.argmin(dist)])

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

def load_CIFAR(one_hot=True):
    CLIP_MAX = 0.5
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype("float32")
    X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
    X_test = X_test.astype("float32")
    X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)

    if one_hot:
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test

def load_MNIST(one_hot=True, channel_first=True):

    # Load data,可以不用下载，在keras.datasets包中直接调用
    mnist_path = '/home/data/mnist.npz'
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
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":

    # load the data:
    dataset='cifar10'
    X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改
    # X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)  # 在utils中修改

    # set the model
    model_path = '/home/model/model_cifar_b4.h5'
    # model_path = '/home/model/cifar10_vgg16.h5'
    # model_path = '/home/model/model_mnist.h5'
    # model_path = '/home/model/mnist_lenet5.h5'
    model = load_model(model_path )
    model.summary()

    upper_bound = 2000

    # skip flattern layers和inputlayers
    skip_layers = [ ]
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)
    print(skip_layers)

    # for lenet4: 除input、flatten和softmax外的所有层
    subject_layer = list(set(range(len(model.layers))) - set(skip_layers))[:-1]
    print('subject_layer: ', subject_layer)

    layer_names = []
    lyr = [20]  # activation_8:21; dense_2:20
    for ly_id in lyr:
        layer_names.append(model.layers[ly_id].name)
    print(layer_names)

    seed_selection_strategy = [ 'BE']
    repeat_times = 5
    file_path = '/home/EI_Neurons/cifar10_convnet/ADAPT/'

    sa = SurpriseAdequacy(model, X_train, layer_names, upper_bound, dataset)

    # ROBOT 数据加载
    # for seeds in seed_selection_strategy:
    #     for repeat in range(0, repeat_times):
    #
    #         all_gen_inps = []
    #         load_inps = np.load(file_path + 'robot_cifar10_convnet_{}Seed_all_gen_inps_200_5_{}_repeatTimes.npz'.format(seeds, repeat), allow_pickle=True)
    #         # load_dict = {key: load_inps[key] for key in load_inps.files}
    #         load_dict = load_inps.get('arr_0').item()
    #
    #         for key, value in load_dict.items():
    #             seed_inps = load_dict[key]
    #             for inp in seed_inps:
    #                 all_gen_inps.append(inp.numpy().reshape(32,32,3))
    #         test_inps = np.array(all_gen_inps)
    #         print('np.array(all_gen_inps):', test_inps.shape)
    #         sa.test(test_inps, dataset, repeat,seed_selection_strategy[0] )

    ##MOON 数据加载
    # for seeds in seed_selection_strategy:
    #     all_nlc_coverage = []
    #     for repeat in range(0, repeat_times):
    #         all_gen_inps = []
    #
    #         load_inps = np.load(file_path + 'MOON_all_gen_inps_{}_strategy_{}_repeat_V2.npz'.format(seeds, repeat),
    #                             allow_pickle=True)
    #         load_dict = {key: load_inps[key] for key in load_inps.files}
    #
    #         for key, value in load_dict.items():
    #             seed_inps = load_dict[key]
    #             for inp in seed_inps:
    #                 all_gen_inps.append(inp.reshape(32, 32, 3))
    #         test_inps = np.array(all_gen_inps)
    #         print('np.array(all_gen_inps):', test_inps.shape)
    #         sa.test(test_inps, dataset, repeat, seed_selection_strategy[0])
    #
    #     print('rs_val: ', all_nlc_coverage)

    # # ADAPT 数据加载
    for seeds in seed_selection_strategy:
        for repeat in range(0, repeat_times):
            all_gen_inps = []
            load_inps = np.load(file_path + 'Adapt_cifar_convnet_adaptive_strategy_{}Seed_NC_all_gen_inps_200_5_{}_repeat.npy'.format(seeds, repeat),
                                allow_pickle=True).item()

            for key, value in load_inps.items():

                seed_inps = load_inps[key]
                # each label对应的生成的inps构成的list
                for inp_list in seed_inps:
                    for inp in inp_list:
                        all_gen_inps.append(inp.numpy().reshape(32,32,3))
            test_inps = np.array(all_gen_inps)
            print('np.array(all_gen_inps):', test_inps.shape)
            sa.test(test_inps, dataset, repeat,seed_selection_strategy[0] )

