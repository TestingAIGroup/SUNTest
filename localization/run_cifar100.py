import time
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from dataset.utils_spectrum import filter_correct_classifications, get_class_matrix, get_layers_ats, construct_spectrum_matrices, find_unique_fault_type, \
                                     obtain_results, load_compound_CIFAR10, load_compound_img_trans_CIFAR10, load_compound_cifar100_vgg_img_adv, get_bin, get_bin_fifty
from dataset.utils_spectrum import relevance_analysis, relevance_all_analysis, find_EI_neurons_gradients, find_EI_neurons_multiply, find_EI_neurons_gradients_variant, find_EI_neurons, gradient_analysis, \
                                    load_compound_fashion, write_results_to_file, write_results_to_file_inconsist, load_compound_cifar_convnet_img_adv, find_EI_neurons_variant
from dataset.spectrum_analysis import tarantula_analysis, ochiai_analysis, dstar_analysis, other_analysis, other_star_analysis
import heapq
from keras.datasets import cifar100

def get_fixed_threshold():
    thres = np.zeros(512)
    for i in range(len(thres)):
        thres[i] = 0.75

    return thres

def load_CIFAR100(one_hot=True):
    CLIP_MAX = 0.5
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    X_train = X_train.astype("float32")
    X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
    X_test = X_test.astype("float32")
    X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=100)
        y_test = np_utils.to_categorical(y_test, num_classes=100)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # 载入模型
    # model = load_model('model/ConvNet_mnist.h5df')  # ConvNet模型 12层
    # model = load_model('./data/densenet_cifar10.h5df')

    # model_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/model/cifar10_vgg16.h5'
    # model_path = 'E:/githubAwesomeCode/1DLTesting/improve_DLtesting/neural_networks/cifar100_vgg_v1.h5'
    model_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/model/cifar100_resnet18.h5'
    model = load_model(model_path)
    model_name = 'CIFAR100_Vgg'
    model.summary()

    dataset = 'CIFAR100_Vgg'
    X_train, Y_train, X_test, Y_test = load_CIFAR100()  # 在utils中修改
    _, acc = model.evaluate(X_test, Y_test)
    print('acc: ', acc)

    # 被模型正确预测，正确的标签；被模型错误预测，错误的标签；被模型正确预测的样本下标，被模型错误预测的样本下标
    X_train_corr, Y_train_corr, X_train_misc, Y_train_misc, train_corr_idx, train_misc_idx = \
                            filter_correct_classifications(model, X_train, Y_train)
    print('X_train_misc.shape: ', X_train_misc.shape)


    # adv_name = 'imgTrans'
    adv_name = 'ADV'
    com_size = [0]
    susp_num = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    random_size = ['random1', 'random2', 'random3']

    trainable_layers = [66]

    layer_names = [model.layers[idx].name for idx in trainable_layers]
    train_ats, train_pred = get_layers_ats(model, X_train, layer_names, dataset)
    print('train_ats: ', train_ats.shape)
    train_misc_ats = train_ats[train_misc_idx]
    print('ats: ', train_misc_ats.shape)
    cumulative_activation = train_misc_ats.sum(axis = 0)
    print('cumulative_activation: ', cumulative_activation.shape)

    for random in random_size:
        all_results_acc = {}
        all_results_incon = {}

        for cs in com_size:
            # compound_x_test, compound_y_test = load_compound_CIFAR10(cs, adv_name)
            # compound_x_test, compound_y_test = load_compound_img_trans_CIFAR10(cs, random)
            compound_x_test, compound_y_test = load_compound_cifar100_vgg_img_adv(cs, random)
            score = model.evaluate(compound_x_test, compound_y_test, verbose=0)
            print('After mutation, %s, %s, Test accuracy: %s' %(cs, random, score[1]))

        #     activation_threshold = train_ats.mean(axis=0)  # neuron_act_sum
        # #   activation_threshold = train_ats.mean(axis=0) + train_ats.std(axis=0)
            activation_threshold = np.array(get_bin(train_ats))

            # activation_threshold = get_fixed_threshold()
          # print('activation_threshold: ', activation_threshold.shape)
            neuron_num = activation_threshold.shape[0]
            correct_classifications = train_corr_idx
            misclassifications = train_misc_idx

            spectrum_begin_time = time.time()
            spectrum_num = construct_spectrum_matrices(model, trainable_layers, correct_classifications, misclassifications,
                                train_ats, activation_threshold, model_name)

            #spectrum_num  = [num_ac, num_uc, num_af, num_uf]
            num_ac = spectrum_num[0]; num_uc = spectrum_num[1]; num_af = spectrum_num[2]; num_uf = spectrum_num[3]
            spectrum_end_time = time.time()

            spectrum_time = spectrum_end_time -  spectrum_begin_time
            print('spectrum_time:', spectrum_time)

            spectrum_approach = ['tarantula', 'ochiai', 'dstar', 'other', 'other_star']
            results_acc = { }   #存储所有accuracy的字典
            results_incon = { }   #存储所有inconsistency的字典


            # # # 获取神经元的可疑度，筛选可疑神经元
            for app in spectrum_approach:
                print('spectrum_approach: ', spectrum_approach)

                susMea_begin_time = time.time()

                if app == 'tarantula':
                    suspiciousness = tarantula_analysis( num_ac, num_uc, num_af, num_uf)

                elif app == 'ochiai':
                    suspiciousness = ochiai_analysis( num_ac, num_uc, num_af, num_uf)

                elif app == 'dstar':
                    suspiciousness = dstar_analysis( num_ac, num_uc, num_af, num_uf)

                elif app == 'other':
                    suspiciousness = other_analysis(num_ac, num_uc, num_af, num_uf)

                elif app == 'other_star':
                    suspiciousness = other_star_analysis(num_ac, num_uc, num_af, num_uf)

                susMea_end_time = time.time()
                susMea_time = susMea_end_time - susMea_begin_time
                print("susMea_time: ", susMea_time)


                ###################   suspiciousness    ####################
                total_test_incon = []
                for num in susp_num:
                    arr_max = heapq.nlargest( int(num * neuron_num), suspiciousness)
                    suspicious_neuron_idx = map(suspiciousness.index, arr_max)
                    test_ratio = obtain_results(model_path, trainable_layers, suspicious_neuron_idx, compound_x_test, compound_y_test)
                    total_test_incon.append(test_ratio)

                results_incon[ app + '_compound' + str(cs) + '_ratio'] = total_test_incon

                # ###################    compound: relevance + suspiciousness     #######################
                # total_test_incon = []
                # relevances = relevance_analysis(model_name)
                # for num in susp_num:
                #     EI_neuron_idx = find_EI_neurons(suspiciousness, relevances, int(num * neuron_num))
                #     test_ratio = obtain_results(model_path, trainable_layers, EI_neuron_idx, compound_x_test, compound_y_test)
                #     total_test_incon.append(test_ratio)
                #
                # results_incon[app + '_relevance_misc' + '_compound' + str(cs) + '_ratio'] = total_test_incon
                #
                # ###################    compound:  norm = (0.5 + norm) * suspiciousness        #######################
                # total_test_incon = []
                # relevances = relevance_analysis(model_name)
                # for num in susp_num:
                #     EI_neuron_idx = find_EI_neurons_variant(suspiciousness, relevances, int(num * neuron_num))
                #
                #     test_ratio = obtain_results(model_path, trainable_layers, EI_neuron_idx, compound_x_test, compound_y_test)
                #     total_test_incon.append(test_ratio)
                #
                # # results_acc[app + '_relevance_misc' + '_compound_variant' + str(cs)] = total_test_acc
                # results_incon[app + '_relevance_misc' + '_compound_varaint' + str(cs) + '_ratio'] = total_test_incon
                #
                #
                # ####################   only misclassification relevance neurons   #######################
                # model = load_model(model_path)
                # total_test_incon = []
                # for num in susp_num:
                #     relevances = relevance_analysis(model_name).flatten()
                #     relevance_neuron_idx = np.argsort(-relevances)[:int(num * neuron_num)]
                #
                #     test_ratio = obtain_results(model_path, trainable_layers, relevance_neuron_idx, compound_x_test,
                #                compound_y_test)
                #     total_test_incon.append(test_ratio)
                #
                # # results_acc['relevance_misc' + '_compound' + str(cs)] = total_test_acc
                # results_incon['relevance_misc_'  + str(cs) + '_ratio'] = total_test_incon

                ###################    compound: (0.5 + norm) * suspiciousness     #######################
                # total_test_incon = []
                # gradients = gradient_analysis(model_name, train_ats, train_misc_idx)
                # for num in susp_num:
                #     EI_neuron_idx = find_EI_neurons_gradients_variant(suspiciousness, gradients, int(num * neuron_num))
                #     test_ratio = obtain_results(model_path, trainable_layers, EI_neuron_idx, compound_x_test,
                #                                 compound_y_test)
                #     total_test_incon.append(test_ratio)
                #
                # results_incon[app + '_gradients_misc' + '_compound_variant_' + str(cs) + '_ratio'] = total_test_incon
                #
                #
                # ###################    compound: suspiciousness + gradients    #######################
                # total_test_incon = []
                # gradients = gradient_analysis(model_name, train_ats, train_misc_idx)
                # for num in susp_num:
                #     EI_neuron_idx = find_EI_neurons_gradients(suspiciousness, gradients, int(num * neuron_num))
                #     test_ratio = obtain_results(model_path, trainable_layers, EI_neuron_idx, compound_x_test,
                #                                 compound_y_test)
                #     total_test_incon.append(test_ratio)
                #
                # results_incon[app + '_gradients_misc' + '_compound' + str(cs) + '_ratio'] = total_test_incon
                #
                # ###################    compound: suspiciousness + gradients    #######################
                # total_test_incon = []
                # gradients = gradient_analysis(model_name, train_ats, train_misc_idx)
                # for num in susp_num:
                #     EI_neuron_idx = find_EI_neurons_multiply(suspiciousness, gradients, int(num * neuron_num))
                #     test_ratio = obtain_results(model_path, trainable_layers, EI_neuron_idx, compound_x_test,
                #                                 compound_y_test)
                #     total_test_incon.append(test_ratio)
                #
                # results_incon[app + '_gradients_misc' + '_compound_multiply_' + str(cs) + '_ratio'] = total_test_incon

            all_results_incon[cs] = results_incon


        # write_results_to_file(all_results_acc, all_results_incon, model_name, adv_name, susp_num)
        write_results_to_file_inconsist(all_results_incon, model_name, adv_name, susp_num, random)


     # cumulative_activation = train_ats.sum(axis=0)
     # # # all relevance neurons
        # model = load_model(model_path)
        # total_test_acc = []; total_test_incon = []
        # for num in susp_num:
        #     relevances = relevance_all_analysis(model_name).flatten()
        #     relevance_neuron_idx = np.argsort(-relevances)[:int(num * neuron_num)]
        #
        #     test_accuracy, test_ratio = \
        #         obtain_results(model_path, trainable_layers, relevance_neuron_idx, compound_x_test,
        #                        compound_y_test)
        #     total_test_acc.append(test_accuracy); total_test_incon.append(test_ratio)
        # results_acc['relevance_all' + '_compound' + str(cs)] = total_test_acc
        # results_incon['relevance_all' + '_compound' + str(cs) + '_ratio'] = total_test_incon
        #
            # all_results_acc[cs] = results_acc
