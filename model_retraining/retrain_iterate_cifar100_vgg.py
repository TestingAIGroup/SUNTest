import xlwt
import numpy as np
from keras import optimizers
from keras.models import load_model
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
import os
#
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 70:
        lr *= 1e-3
    if epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def load_CIFAR(one_hot=True):
    CLIP_MAX = 0.5
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    X_train = X_train.astype("float32")
    X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
    X_test = X_test.astype("float32")
    X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)

    if one_hot:
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test

def write_results_to_file(all_result, root_path, dataset, approach, seed_app):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    i = 0
    for index_path in all_result:
        print('cs: ', index_path)
        acc = all_result[index_path]  # {des: [accuracy] }
        sheet1.write(i, 0, index_path)
        j = 1
        for key in acc:
            sheet1.write(i, j, key)
            j += 1
        i = i + 1

    f.save(root_path + 'retraining/{}_{}_{}_4000-3000_Random-Neuron-RMSprop_RetrainingResults.xls'.format(dataset, approach, seed_app))

def load_ROBOT_gen_data( Y_test, idx_path):
    gen_x = []
    gen_x_index = []
    gen_y = []
    all_gen_inps = []

    gen_test_file = np.load(idx_path, allow_pickle=True)
    load_dict = gen_test_file.get('arr_0').item()

    # robot 是根据200个seed对应的label构成的dict
    for key, value in load_dict.items():
        seed_inps = load_dict[key]
        for inp in seed_inps:
            all_gen_inps.append(inp.numpy().reshape(32,32,3))
            gen_x_index.append(key)
    gen_x = np.array(all_gen_inps)
    print('np.array(all_gen_inps):', gen_x.shape)

    gen_y = Y_test[gen_x_index]

    return gen_x, gen_y

def load_MOON_gen_data( Y_test, idx_path):
    gen_x = []
    gen_x_index = []
    gen_y = []
    all_gen_inps = []

    gen_test_file = np.load(idx_path, allow_pickle=True)
    load_dict = {key: gen_test_file[key] for key in gen_test_file.files}

    # MOON与ROBOT文件存储的原理一致 是根据200个seed对应的label构成的dict
    for key, value in load_dict.items():
        seed_inps = load_dict[key]
        for inp in seed_inps:
            all_gen_inps.append(inp.reshape(32,32,3))
            gen_x_index.append(int(key))
    gen_x = np.array(all_gen_inps)
    print('np.array(all_gen_inps):', gen_x.shape)

    gen_y = Y_test[gen_x_index]

    return gen_x, gen_y

def load_ADAPT_gen_data( Y_test, idx_path):
    gen_x = []
    gen_x_index = []
    gen_y = []
    all_gen_inps = []

    gen_test_file = np.load(idx_path, allow_pickle=True).item()

    for key, value in gen_test_file.items():
        seed_inps = gen_test_file[key]
        for inp_list in seed_inps:
            for inp in inp_list:
                all_gen_inps.append(inp.numpy().reshape(32,32,3))
                gen_x_index.append(key)
    gen_x = np.array(all_gen_inps)
    print('np.array(all_gen_inps):', gen_x.shape)

    gen_y = Y_test[gen_x_index]

    return gen_x, gen_y

def load_compound_cifar100(ratio, random):
    root_path = '/workspace/EI_Neurons/cifar100_vgg/retraining/compound/'
    data_path = 'cifar100_vgg_adv_compound_V2_%s_%s.npz' % (str(ratio), random)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = compound_data['y_test']
    return x_dest, y_dest


def load_compound_cifar(ratio, random):
    root_path = '/home/EI_Neurons/cifar10_convnet/retraining/compound/'
    data_path = 'cifar_convnet_adv_compound_V2_%s_%s.npz' % (str(ratio), random)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = compound_data['y_test']
    return x_dest, y_dest

def load_DLFuzz_gen_data( Y_test, idx_path):
    gen_x = []
    gen_x_index = []
    gen_y = []
    all_gen_inps = []

    gen_test_file = np.load(idx_path, allow_pickle=True)
    load_dict = gen_test_file.get('arr_0').item()

    # robot 是根据200个seed对应的label构成的dict
    for key, value in load_dict.items():
        seed_inps = load_dict[key]
        for inp in seed_inps:
            # all_gen_inps.append(inp.numpy().reshape(32,32,3)) # robot数据
            all_gen_inps.append(inp.reshape(32, 32, 3))
            gen_x_index.append(key)
    gen_x = np.array(all_gen_inps)
    print('np.array(all_gen_inps):', gen_x.shape)

    gen_y = Y_test[gen_x_index]

    return gen_x, gen_y

def load_CIFAR100(one_hot=True):

    CLIP_MAX =0.5
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    X_train = X_train.astype("float32")
    X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
    X_test = X_test.astype("float32")
    X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)

    if one_hot:
        y_train = to_categorical(y_train, num_classes=100)
        y_test = to_categorical(y_test, num_classes=100)

    return X_train, y_train, X_test, y_test

def retrain(model_name, dataset):
    # 载入模型
    if model_name == "cifar100_vgg":
        model_path = '/workspace/model/cifar100_vgg_v1.h5'
        model = load_model(model_path)

    if dataset == 'cifar100':
        X_train, Y_train, X_test, Y_test = load_CIFAR100()  # 在utils中修改

    _, original_acc = model.evaluate(X_test, Y_test)
    print('original_acc:', original_acc)

    root_saved_path = '/workspace/EI_Neurons/cifar100_vgg/'

    compound_ratio = [4000]
    random_lst = ['random1', 'random2']

    approach_lst = ['MOON-Random-Neuron']
    seed_selection_strategy = ['MOON']
    repeat_times = 3
    all_result = {}

    for app in approach_lst:
        print('approach name: ------------------------------', app)

        for seed_app in seed_selection_strategy:
            print('seed_app: ======', seed_app)

            for ratio in compound_ratio:
                for random in random_lst:
                    # 获取用于评估模型重训练效果的数据集
                    compound_x_eval, compound_y_eval = load_compound_cifar100(ratio, random)

                    for repeat in range(1, repeat_times):

                        # 加载各生成方法生成的测试数据集，与原始训练数据进行融合，用于模型重训练
                        if app == 'ROBOT':
                            idx_path = root_saved_path + app + '/robot_cifar10_convnet_{}Seed_all_gen_inps_200_5_{}_repeatTimes.npz'.format(seed_app, repeat)
                            res_name = root_saved_path + app + '/robot_cifar10_convnet_{}Seed_all_gen_inps_200_5_{}_repeatTimes_{}compound_{}random'.format(seed_app, repeat, ratio, random)
                            gen_x_train, gen_y_train = load_ROBOT_gen_data(Y_test, idx_path)
                        elif app == 'MOON':
                            idx_path = root_saved_path + app + '/MOON_all_gen_inps_{}_strategy_{}_repeat.npz'.format(seed_app, repeat)
                            res_name = root_saved_path + app + '/MOON_all_gen_inps_{}_strategy_{}_repeat_{}compound_{}random'.format(seed_app, repeat, ratio, random)
                            gen_x_train, gen_y_train = load_MOON_gen_data(Y_test, idx_path)

                        elif app == 'MOON-Random-Neuron':
                            idx_path = root_saved_path + 'MOON/' + app + '/MOON_all_gen_inps_RandomNeuron_{}_strategy_{}_repeat_V1.npz'.format(
                                seed_app, repeat)
                            res_name = root_saved_path + app + '/MOON_all_gen_inps_RandomMutation_{}_strategy_{}_repeat_{}compound_{}random'.format(
                                seed_app, repeat, ratio, random)
                            gen_x_train, gen_y_train = load_MOON_gen_data(Y_test, idx_path)


                        elif app == 'ADAPT':
                            idx_path = root_saved_path + app + '/Adapt_cifar_convnet_adaptive_strategy_{}Seed_NC_all_gen_inps_200_5_{}_repeat.npy'.format(seed_app, repeat)
                            res_name = root_saved_path + app + '/Adapt_cifar_convnet_adaptive_strategy_{}Seed_NC_all_gen_inps_200_5_{}_repeat_{}compound_{}random'.format(seed_app, repeat, ratio, random)
                            gen_x_train, gen_y_train = load_ADAPT_gen_data(Y_test, idx_path)
                        else:
                            idx_path = root_saved_path + app + '/DLFuzz_cifar10_convnet_all_gen_inps_200_5_{}_{}_repeatTimes.npz'.format(
                                seed_app, repeat)
                            res_name = root_saved_path + app + '/DLFuzz_cifar_convnet_all_gen_inps_200_5_{}_{}_repeatTimes_{}compound_{}random'.format(
                                seed_app, repeat, ratio, random)
                            gen_x_train, gen_y_train = load_DLFuzz_gen_data(Y_test, idx_path)

                        # 使用原始的训练集、生成的测试集，共同重训练模型
                        together_x = np.append(X_train, gen_x_train, axis = 0)
                        together_y = np.append(Y_train, gen_y_train, axis = 0)
                        print('together_x_y:', together_x.shape, together_y.shape)

                        retrain_acc_sum = 0
                        all_retrain_acc = [] # 存储所有重训练后模型的准确率

                        # 循环多次取average
                        for i in range(5):
                            model_path = '/workspace/model/cifar100_vgg_v1.h5'
                            model = load_model(model_path)

                            _, compound_eval_orig_acc = model.evaluate(compound_x_eval, compound_y_eval)
                            print('compound adv accuracy:', compound_eval_orig_acc)

                            opt = tf.keras.optimizers.legacy.RMSprop(lr=0.0001, decay=1e-6)
                            model.compile(loss='categorical_crossentropy',
                            optimizer = opt,
                            metrics=['accuracy'])

                            # model.compile(loss='categorical_crossentropy',
                            #               optimizer=Adam(learning_rate=lr_schedule(0)),
                            #               metrics=['accuracy'])

                            model.fit(together_x, together_y, batch_size = 500, epochs = 5, shuffle = True, verbose = 1, validation_data =(together_x, together_y))
                            _, retrain_acc = model.evaluate(compound_x_eval, compound_y_eval)
                            retrain_acc_sum += retrain_acc
                            all_retrain_acc.append(retrain_acc)

                        retrain_average_acc = retrain_acc_sum / 5   #重训练后的平均值
                        orig_acc_result = round(compound_eval_orig_acc, 4)    # 原始accuracy
                        retrain_acc_result = round(retrain_average_acc, 4)
                        acc_improvement_result = round(retrain_average_acc - compound_eval_orig_acc, 4)

                        all_result[res_name] = [orig_acc_result, all_retrain_acc[0], all_retrain_acc[1], all_retrain_acc[2], all_retrain_acc[3], all_retrain_acc[4],
                                              retrain_acc_result, acc_improvement_result]

        write_results_to_file(all_result, root_saved_path, dataset, app, seed_app)

if __name__ == "__main__":
    if tf.config.list_physical_devices('GPU'):
        print('GPU is available')
    else:
        print('GPU is not available')

    model_name = 'cifar100_vgg'
    dataset = 'cifar100'
    retrain(model_name, dataset)




