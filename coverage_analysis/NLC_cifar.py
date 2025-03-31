
from multiprocessing import Pool
from keras.models import load_model, Model
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os

class Estimator(object):
    def __init__(self, feature_num, num_class = 1):
        self.num_class = num_class
        self.CoVariance = tf.zeros([num_class, feature_num, feature_num], dtype=tf.float32)
        self.Ave = tf.zeros([num_class, feature_num], dtype=tf.float32)
        self.Amount = tf.zeros([num_class], dtype=tf.float32)

    def calculate(self, features, labels=None):
        N = tf.shape(features)[0]
        C = self.num_class
        A = tf.shape(features)[1]

        if labels is None:
            labels = tf.zeros([N], dtype=tf.int32)

        NxCxFeatures = tf.tile(tf.expand_dims(features, axis=1), multiples=[1, C, 1])
        onehot = tf.one_hot(labels, depth=C)
        NxCxA_onehot = tf.expand_dims(onehot, axis=2)

        features_by_sort = NxCxFeatures * NxCxA_onehot

        Amount_CxA = tf.reduce_sum(NxCxA_onehot, axis=0)
        Amount_CxA = tf.where(Amount_CxA == 0, 1, Amount_CxA)

        ave_CxA = tf.reduce_sum(features_by_sort, axis=0) / Amount_CxA

        var_temp = features_by_sort - tf.expand_dims(ave_CxA, axis=0) * NxCxA_onehot

        var_temp = tf.linalg.matmul(tf.transpose(var_temp, perm=[1, 2, 0]), tf.transpose(var_temp, perm=[1, 0, 2]))
        var_temp /= tf.expand_dims(Amount_CxA, axis=2)

        sum_weight_CV = tf.reduce_sum(onehot, axis=0, keepdims=True)
        sum_weight_AV = tf.reduce_sum(onehot, axis=0)

        weight_CV = sum_weight_CV / (sum_weight_CV + tf.expand_dims(tf.expand_dims(self.Amount, axis=1), axis=2))
        weight_CV = tf.where(tf.math.is_nan(weight_CV), 0.0, weight_CV)

        weight_AV = sum_weight_AV / (sum_weight_AV + tf.expand_dims(self.Amount, axis=1))
        weight_AV = tf.where(tf.math.is_nan(weight_AV), 0.0, weight_AV)

        additional_CV = weight_CV * (1 - weight_CV) * tf.linalg.matmul(
            tf.expand_dims(self.Ave - ave_CxA, axis=2),
            tf.expand_dims(self.Ave - ave_CxA, axis=1)
        )

        new_CoVariance = (self.CoVariance * (1 - weight_CV) + var_temp * weight_CV) + additional_CV

        new_Ave = (self.Ave * (1 - weight_AV) + ave_CxA * weight_AV)

        new_Amount = self.Amount + tf.reduce_sum(onehot, axis=0)

        return {
            'Ave': new_Ave,
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }



# class NLC():
#     def init_variable(self, hyper=None):
#         assert hyper is None, 'NLC has no hyper-parameter'
#         self.estimator_dict = {}
#         self.current = 1
#         for layer_name, layer_size in self.layer_size_dict.items():
#             self.estimator_dict[layer_name] = Estimator(feature_num=layer_size[0])
#
#     def calculate(self, data):
#         stat_dict = {}
#         layer_output_dict = get_layer_output(self.model, data)
#         for layer_name, layer_output in layer_output_dict.items():
#             info_dict = self.estimator_dict[layer_name].calculate(layer_output.numpy())
#             stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])
#         return stat_dict
#
#     def update(self, stat_dict, gain=None):
#         if gain is None:
#             for layer_name in stat_dict.keys():
#                 (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
#                 self.estimator_dict[layer_name].Ave.assign(new_Ave)
#                 self.estimator_dict[layer_name].CoVariance.assign(new_CoVariance)
#                 self.estimator_dict[layer_name].Amount.assign(new_Amount)
#             self.current = self.coverage(self.estimator_dict)
#         else:
#             (delta, layer_to_update) = gain
#             for layer_name in layer_to_update:
#                 (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
#                 self.estimator_dict[layer_name].Ave.assign(new_Ave)
#                 self.estimator_dict[layer_name].CoVariance.assign(new_CoVariance)
#                 self.estimator_dict[layer_name].Amount.assign(new_Amount)
#             self.current += delta
#
#     def coverage(self, stat_dict):
#         val = 0
#         for layer_name in stat_dict.keys():
#             CoVariance = stat_dict[layer_name].CoVariance.numpy()
#             val += self.norm(CoVariance)
#         return val
#
#     def gain(self, stat_new):
#         total = 0
#         layer_to_update = []
#         for layer_name in stat_new.keys():
#             (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
#             value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance.numpy())
#             if value > 0:
#                 layer_to_update.append(layer_name)
#                 total += value
#         if total > 0:
#             return (total, layer_to_update)
#         else:
#             return None
#
#     def norm(self, vec, mode='L1', reduction='mean'):
#         m = vec.shape[0]
#         assert mode in ['L1', 'L2']
#         assert reduction in ['mean', 'sum']
#         if mode == 'L1':
#             total = np.sum(np.abs(vec))
#         elif mode == 'L2':
#             total = np.sum(np.square(vec))**0.5
#         if reduction == 'mean':
#             return total / m
#         elif reduction == 'sum':
#             return total
#
#     def save(self, path):
#         print('Saving recorded NLC in %s...' % path)
#         stat_dict = {}
#         for layer_name in self.estimator_dict.keys():
#             stat_dict[layer_name] = {
#                 'Ave': self.estimator_dict[layer_name].Ave.numpy(),
#                 'CoVariance': self.estimator_dict[layer_name].CoVariance.numpy(),
#                 'Amount': self.estimator_dict[layer_name].Amount.numpy()
#             }
#         np.savez(path, stat=stat_dict)
#
#     def load(self, path):
#         print('Loading saved NLC from %s...' % path)
#         loaded_data = np.load(path, allow_pickle=True)
#         stat_dict = loaded_data['stat'].item()
#         for layer_name in stat_dict.keys():
#             self.estimator_dict[layer_name].Ave.assign(stat_dict[layer_name]['Ave'])
#             self.estimator_dict[layer_name].CoVariance.assign(stat_dict[layer_name]['CoVariance'])
#             self.estimator_dict[layer_name].Amount.assign(stat_dict[layer_name]['Amount'])


def norm(vec, mode='L2', reduction='mean'):
        m = vec.shape[0]
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = np.sum(np.abs(vec))
        elif mode == 'L2':
            total = np.sum(np.square(vec))**0.5
        if reduction == 'mean':
            return total / m
        elif reduction == 'sum':
            return total

# get_NLC(all_gen_inps, model, X_train, Y_train,  layer_names)
def get_NLC(rs, model, X_train, Y_train, layer_names):
    s = 1000
    upper_bound = 2
    dataset_name = 'cifar_convnet'
    ratio = 2000
    random ='random1'
    sa = SurpriseAdequacy(model, X_train, layer_names, upper_bound, dataset_name, str(ratio), s, random)
    train_ats, train_pred, target_ats, target_pred = sa.test(rs, dataset_name)
    print('train_ats: ', train_ats.shape)
    print('target_ats： ', target_ats.shape)
    layer_output_dict = {}
    layer_output_dict[layer_names[0]] = target_ats

    estimator_dict = {}
    estimator_dict[layer_names[0]] = Estimator(feature_num = 128)
    stat_dict = {}
    for layer_name, layer_output in layer_output_dict.items():
        info_dict = estimator_dict[layer_name].calculate(layer_output)
        stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])
    val = 0
    for layer_name in stat_dict.keys():
        CoVariance = stat_dict[layer_name][1]
        val += norm(CoVariance)

    print('val: ', val)

    return val


class SurpriseAdequacy:
   # sa = SurpriseAdequacy(model, X_train, layer_names, upper_bound, dataset, str(ratio), s, random)

    def __init__(self,  model, train_inputs, layer_names, upper_bound, dataset, cs, selected_size, random):

        #self.surprise = surprise
        self.model = model
        self.train_inputs = train_inputs
        self.layer_names = layer_names
        self.upper_bound = upper_bound
        self.n_buckets = 1000
        self.dataset = dataset
        self.save_path='E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/' \
                       '4retraining_complementary/4retraining_svhn/mdsa/'
        if dataset == 'drive': self.is_classification = False   #处理非分类任务
        else: self.is_classification = True
        self.num_classes = 10
        self.var_threshold = 1e-5
        self.cs = cs
        self.selected_size = selected_size
        self.random = random

    def test(self, test_inputs, dataset_name, instance='mdsa'):

        if instance == 'mdsa':
            train_ats, train_pred, target_ats, target_pred = fetch_mdsa(self.model, self.train_inputs, test_inputs,
                                   dataset_name, self.layer_names,
                                   self.num_classes, self.is_classification,
                                   self.var_threshold, self.save_path, self.dataset)
            return train_ats, train_pred, target_ats, target_pred

def fetch_mdsa(model, X_train, x_target, target_name, layer_names, num_classes,is_classification, var_threshold, save_path, dataset):

    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, X_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset)

    return train_ats, train_pred, target_ats, target_pred

def _get_train_target_ats(model, x_train, x_target, target_name, layer_names,
                          num_classes, is_classification, save_path, dataset):

    saved_train_path = _get_saved_path(save_path, dataset, "train", layer_names)

    if os.path.exists(saved_train_path[0]):

        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])  # train_ats:  (60000, 12)
        train_pred = np.load(saved_train_path[1])  # train_pred:  (60000, 10)

    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes = num_classes,
            is_classification=is_classification,
            save_path=saved_train_path,
        )

    saved_target_path = _get_saved_path(save_path, dataset, 'cifar', layer_names)
    if os.path.exists(saved_target_path[0]):
        # In case target_ats is stored in a disk
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])


    else:
        # target就是X_train
        target_ats, target_pred = get_ats(
            model,
            x_target, #X_test
            target_name,
            layer_names,
            num_classes=num_classes,
            is_classification=is_classification,
            save_path=saved_target_path,
        )

    return train_ats, train_pred, target_ats, target_pred

def get_ats( model, dataset, name, layer_names, save_path=None, batch_size=128, is_classification=True, num_classes=10, num_proc=10,):

    temp_model = Model(
        inputs=model.input, #Tensor("input_1:0", shape=(None, 28, 28, 1), dtype=float32)
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names], #layer_name 层的神经元输出的值
    )

    if is_classification:
        p = Pool(num_proc)  #Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果池还没有满，就会创建一个新的进程来执行请求。
        pred = model.predict(dataset, batch_size=batch_size, verbose=1)

        if len(layer_names) == 1:  #计算coverage的只有一层
            layer_outputs = [temp_model.predict(dataset, batch_size=batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)

        ats = None

        for layer_name, layer_output in zip(layer_names, layer_outputs):  # (1, 60000, 4, 4, 12)
            if layer_output[0].ndim == 3:
                # For convolutional layers and pooling layers 数据的维数是3维的
                layer_matrix = np.array(p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))]))
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None
    #
    # if save_path is not None:
    #     np.save(save_path[0], ats)
    #     np.save(save_path[1], pred)

        return ats, pred


def _get_saved_path(base_path, dataset, dtype, layer_names):
    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )
def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

def _get_saved_path(base_path, dataset, dtype, layer_names):
    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )



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

def filter_correct_classifications(model, X, Y):
    X_corr = []; X_corr_idx = []
    Y_corr = []
    X_misc = []; X_misc_idx = []
    Y_misc = []
    preds = model.predict(X)  # np.expand_dims(x,axis=0))

    for idx, pred in enumerate(preds):
        if np.argmax(pred) == np.argmax(Y[idx]):
            X_corr.append(X[idx])
            Y_corr.append(Y[idx])
            X_corr_idx.append(idx)
        else:
            X_misc.append(X[idx])
            Y_misc.append(preds[idx])
            X_misc_idx.append(idx)
            # X_misc, Y_misc以及X_misc_idx顺序是一一对应的
    return np.array(X_corr), np.array(Y_corr), np.array(X_misc), np.array(Y_misc)

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

    # load the data==============:
    dataset = 'cifar10'
    X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改

    # set the model====================
    model_path = '/workspace/model/model_cifar_b4.h5'
    model_name = (model_path).split('/')[-1]
    model = load_model(model_path)
    model.summary()

    X_test_corr, Y_test_corr, X_test_misc, Y_test_misc = filter_correct_classifications(model, X_test, Y_test)
    print(X_test_corr.shape)  # (9879, 28, 28, 1)
    print(X_test_misc.shape)  # (121, 28, 28, 1)

    layer_names = []
    lyr = [7]
    for ly_id in lyr:
        layer_names.append(model.layers[ly_id].name)

    file_path = '/workspace/EI_Neurons/cifar10_convnet/MOON/'
    repeat_times = 5
    approach_lst = ['MOON-Original']
    seed_selection_strategy = ['Random', 'RobotBe', 'RobotKM', 'Deepgini', 'MCP']
    version = ['V1']

    all_nlc_coverage = []
    for seeds in seed_selection_strategy:
        all_nlc_coverage = []
        for repeat in range(0, repeat_times):
            all_gen_inps = []

            load_inps = np.load(
                file_path + approach_lst[0] + '/MOON_all_gen_inps_Original_{}_strategy_{}_repeat_{}.npz'.format(seeds, repeat,version[0]),
                allow_pickle=True)
            load_dict = {key: load_inps[key] for key in load_inps.files}

            for key, value in load_dict.items():
                seed_inps = load_dict[key]
                for inp in seed_inps:
                    all_gen_inps.append(inp.reshape(32, 32, 3))

            rs_val = get_NLC(np.array(all_gen_inps), model, X_train, Y_train, layer_names)
            all_nlc_coverage.append(rs_val)

        print('rs_val: ', all_nlc_coverage)

    # ROBOT
    # for seeds in seed_selection_strategy:
    #     all_nlc_coverage = []
    #     for repeat in range(0, repeat_times):
    #         all_gen_inps = []
    #
    #         load_inps = np.load(
    #             file_path + 'robot_mnist_lenet5_{}Seed_all_gen_inps_200_5_{}_repeatTimes.npz'.format(seeds, repeat),
    #             allow_pickle=True)
    #         # load_dict = {key: load_inps[key] for key in load_inps.files}
    #         load_dict = load_inps.get('arr_0').item()
    #         for key, value in load_dict.items():
    #             seed_inps = load_dict[key]
    #             for inp in seed_inps:
    #                 all_gen_inps.append(inp.numpy().reshape(28,28,1))
    #         rs_val = get_NLC(np.array(all_gen_inps), model, X_train, Y_train,  layer_names)
    #         all_nlc_coverage.append(rs_val)
    #
    #     print('rs_val_{}: '.format(seeds), all_nlc_coverage)

    # ADAPT
    # for seeds in seed_selection_strategy:
    #     for repeat in range(0, repeat_times):
    #         all_gen_inps = []
    #         load_inps = np.load(file_path + 'Adapt_cifar_vgg_adaptive_strategy_{}Seed_TKNC_all_gen_inps_200_5_{}_repeat.npy'.format(seeds, repeat),
    #                             allow_pickle=True).item()
    #
    #         for key, value in load_inps.items():
    #
    #             seed_inps = load_inps[key]
    #             print('seed_inps: ', len(seed_inps))
    #             # each label对应的生成的inps构成的list
    #             for inp_list in seed_inps:
    #                 for inp in inp_list:
    #                     all_gen_inps.append(inp.numpy().reshape(32,32,3))
    #         rs_val = get_NLC(np.array(all_gen_inps), model, X_train, Y_train, layer_names)
    #         all_nlc_coverage.append(rs_val)
    #
    #     print('rs_val: ', all_nlc_coverage)



