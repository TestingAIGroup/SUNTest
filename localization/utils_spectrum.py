
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras import backend as K
import sys
from sklearn.metrics import classification_report, confusion_matrix
from math import ceil
import numpy as np
import h5py
from os import path, makedirs
import traceback
from keras.models import Model, load_model
from multiprocessing import Pool
import xlwt
import os
import matplotlib.pyplot as plt


def load_CIFAR(one_hot=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test  = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def load_MNIST(one_hot=True, channel_first=True):
    """
    Load MNIST data
    :param one_hot:
    :return:
    """
    #Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Preprocess dataset
    #Normalization and reshaping of input.
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
        #For output, it is important to change number to one-hot vector.
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test  = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


# def load_model(model_name):
#     print(model_name)
#     json_file = open(model_name + '.json', 'r')
#     print('okx')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#     # load weights into model
#     model.load_weights(model_name + '.h5')
#
#     model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
#     print("Model structure loaded from ", model_name)
#     return model


def get_layer_outs_old(model, class_specific_test_set):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([class_specific_test_set, 1.]) for func in functors]

    return layer_outs


def get_layer_outs(model, test_input):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs

    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([test_input]) for func in functors]

    return layer_outs



def get_python_version():
    if (sys.version_info > (3, 0)):
        # Python 3 code in this block
        return 3
    else:
        # Python 2 code in this block
        return 2


def show_image(vector):
    img = vector
    plt.imshow(img)
    plt.show()


def calculate_prediction_metrics(Y_test, Y_pred, score):
    """
    Calculate classification report and confusion matrix
    :param Y_test:
    :param Y_pred:
    :param score:
    :return:
    """
    #Find test and prediction classes
    Y_test_class = np.argmax(Y_test, axis=1)
    Y_pred_class = np.argmax(Y_pred, axis=1)

    classifications = np.absolute(Y_test_class - Y_pred_class)

    correct_classifications = []
    incorrect_classifications = []
    for i in range(1, len(classifications)):
        if (classifications[i] == 0):
            correct_classifications.append(i)
        else:
            incorrect_classifications.append(i)


    # Accuracy of the predicted values
    print(classification_report(Y_test_class, Y_pred_class))
    print(confusion_matrix(Y_test_class, Y_pred_class))

    acc = sum([np.argmax(Y_test[i]) == np.argmax(Y_pred[i]) for i in range(len(Y_test))]) / len(Y_test)
    v1 = ceil(acc*10000)/10000
    v2 = ceil(score[1]*10000)/10000
    correct_accuracy_calculation =  v1 == v2
    try:
        if not correct_accuracy_calculation:
            raise Exception("Accuracy results don't match to score")
    except Exception as error:
        print("Caught this error: " + repr(error))


def get_dummy_dominants(model, dominants):
    import random
    # dominant = {x: random.sample(range(model.layers[x].output_shape[1]), 2) for x in range(1, len(model.layers))}
    dominant = {x: random.sample(range(0, 10), len(dominants[x])) for x in range(1, len(dominants)+1)}
    return dominant


def save_perturbed_test(x_perturbed, y_perturbed, filename):
    # save X
    with h5py.File(filename + '_perturbations_x.h5', 'w') as hf:
        hf.create_dataset("x_perturbed", data=x_perturbed)

    #save Y
    with h5py.File(filename + '_perturbations_y.h5', 'w') as hf:
        hf.create_dataset("y_perturbed", data=y_perturbed)

    return


def load_perturbed_test(filename):
    # read X
    with h5py.File(filename + '_perturbations_x.h5', 'r') as hf:
        x_perturbed = hf["x_perturbed"][:]

    # read Y
    with h5py.File(filename + '_perturbations_y.h5', 'r') as hf:
        y_perturbed = hf["y_perturbed"][:]

    return x_perturbed, y_perturbed


def save_perturbed_test_groups(x_perturbed, y_perturbed, filename, group_index):
    # save X
    filename = filename + '_perturbations.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        group.create_dataset("x_perturbed", data=x_perturbed)
        group.create_dataset("y_perturbed", data=y_perturbed)

    print("Classifications saved in ", filename)

    return


def load_perturbed_test_groups(filename, group_index):
    with h5py.File(filename + '_perturbations.h5', 'r') as hf:
        group = hf.get('group' + str(group_index))
        x_perturbed = group.get('x_perturbed').value
        y_perturbed = group.get('y_perturbed').value

        return x_perturbed, y_perturbed


def create_experiment_dir(experiment_path, model_name,
                            selected_class, step_size,
                            approach, susp_num, repeat):

    # define experiment name, create directory experiments directory if it
    # doesnt exist
    experiment_name = model_name + '_C' + str(selected_class) + '_SS' + \
    str(step_size) + '_' + approach + '_SN' + str(susp_num) + '_R' + str(repeat)


    if not path.exists(experiment_path):
        makedirs(experiment_path)

    return experiment_name


def save_classifications(correct_classifications, misclassifications, filename, group_index):
    filename = filename + '_classifications.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        group.create_dataset("correct_classifications", data=correct_classifications)
        group.create_dataset("misclassifications", data=misclassifications)

    print("Classifications saved in ", filename)
    return


def load_classifications(filename, group_index):
    filename = filename + '_classifications.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            correct_classifications = group.get('correct_classifications').value
            misclassifications = group.get('misclassifications').value

            print("Classifications loaded from ", filename)
            return correct_classifications, misclassifications
    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)


def save_layer_outs(layer_outs, filename, group_index):
    filename = filename + '_layer_outs.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        for i in range(len(layer_outs)):
            group.create_dataset("layer_outs_"+str(i), data=layer_outs[i])

    print("Layer outs saved in ", filename)
    return


def load_layer_outs(filename, group_index):
    filename = filename + '_layer_outs.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            layer_outs = []
            while True:
                layer_outs.append(group.get('layer_outs_'+str(i)).value)
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        traceback.print_exc()
        sys.exit(-1)
    except (AttributeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("Layer outs loaded from ", filename)
        return layer_outs


def save_suspicious_neurons(suspicious_neurons, filename, group_index):
    filename = filename + '_suspicious_neurons.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        for i in range(len(suspicious_neurons)):
            group.create_dataset("suspicious_neurons"+str(i), data=suspicious_neurons[i])

    print("Suspicious neurons saved in ", filename)
    return


def load_suspicious_neurons(filename, group_index):
    filename = filename + '_suspicious_neurons.h5'
    try:
        with h5py.File(filename, 'r') as hf:
            group = hf.get('group' + str(group_index))
            i = 0
            suspicious_neurons = []
            while True:
                suspicious_neurons.append(group.get('suspicious_neurons' + str(i)).value)
                i += 1

    except (IOError) as error:
        print("Could not open file: ", filename)
        sys.exit(-1)
    except (AttributeError) as error:
        # because we don't know the exact dimensions (number of layers of our network)
        # we leave it to iterate until it throws an attribute error, and then return
        # layer outs to the caller function
        print("Suspicious neurons  loaded from ", filename)
        return suspicious_neurons


def save_original_inputs(original_inputs, filename, group_index):
    filename = filename + '_originals.h5'
    with h5py.File(filename, 'a') as hf:
        group = hf.create_group('group'+str(group_index))
        group.create_dataset("x_original", data=original_inputs)

    print("Originals saved in ", filename)

    return


def filter_val_set(desired_class, X, Y):
    X_class = []
    Y_class = []
    for x,y in zip(X,Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)

    print("Validation set filtered for desired class: " + str(desired_class))

    return np.array(X_class), np.array(Y_class)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def get_trainable_layers(model):

    trainable_layers = []
    for layer in model.layers:
        try:
            weights = layer.get_weights()[0]
            trainable_layers.append(model.layers.index(layer))

        except:
            pass
    # print('trainable_layers[:-1]:', len(trainable_layers))
    # # trainable_layers = trainable_layers[:-1]  #ignore the output layer

    return trainable_layers


# layer_outs = train_ats
def construct_spectrum_matrices(model, trainable_layers,
                                correct_classifications, misclassifications,
                                layer_outs, activation_threshold, model_name ):

    save_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/trace/spectrum/' + model_name + '_spectrum_histogram_V2.npy'
    print('save_path: ', save_path)
    scores = []
    if os.path.exists(save_path):
        print("Found saved spectrum_num, skip serving")
        spectrum_num = np.load(save_path)
        print('spectrum_num: ', spectrum_num.shape)
        # scores.append(np.zeros(model.layers[0].output_shape[-1]))
        # print('scores: ', scores)

    else:
        num_ac = []; num_uc = []
        num_af = []; num_uf = []
        for tl in trainable_layers:
            print(model.layers[tl].output_shape)  # (None, 128)
            num_ac.append(np.zeros(model.layers[tl].output_shape[-1]))  # covered (activated) and failed
            num_uc.append(np.zeros(model.layers[tl].output_shape[-1]))  # uncovered (not activated) and failed
            num_af.append(np.zeros(model.layers[tl].output_shape[-1]))  # covered and succeeded
            num_uf.append(np.zeros(model.layers[tl].output_shape[-1]))  # uncovered and succeeded
            scores.append(np.zeros(model.layers[tl].output_shape[-1]))

        for tl in trainable_layers:
            layer_idx = trainable_layers.index(tl)
            all_neuron_idx = range(model.layers[tl].output_shape[-1])  #range(0, 128)
            test_idx = 0

            for l in layer_outs:   # for each test input
                for neuron_idx in all_neuron_idx:
                    # print('activation_threshold: ', activation_threshold[neuron_idx])
                    # print('l[neuron_idx]: ', l[neuron_idx])
                    # print('------------------')
                    if test_idx in correct_classifications and l[neuron_idx] > activation_threshold[neuron_idx]:
                        num_ac[layer_idx][neuron_idx] += 1   # activated & correct
                    elif test_idx in correct_classifications and l[neuron_idx] <= activation_threshold[neuron_idx]:
                        num_uc[layer_idx][neuron_idx] += 1   # unactivated & correct
                    elif test_idx in misclassifications and l[neuron_idx] > activation_threshold[neuron_idx]:
                        num_af[layer_idx][neuron_idx] += 1   # activated & fail
                    else:
                        num_uf[layer_idx][neuron_idx] += 1   # unactivated & fail

                test_idx += 1

        spectrum_num  = [num_ac, num_uc, num_af,num_uf]
        np.save(save_path, spectrum_num)

    return spectrum_num


    #         covered_idx   = list(np.where(l  > 0)[0])
    #         uncovered_idx = list(set(all_neuron_idx) - set(covered_idx))
    #         #uncovered_idx = list(np.where(l <= 0)[0])
    #         if test_idx  in correct_classifications:
    #             for cov_idx in covered_idx:
    #                 num_cs[layer_idx][cov_idx] += 1
    #             for uncov_idx in uncovered_idx:
    #                 num_us[layer_idx][uncov_idx] += 1
    #         elif test_idx in misclassifications:
    #             for cov_idx in covered_idx:
    #                 num_cf[layer_idx][cov_idx] += 1
    #             for uncov_idx in uncovered_idx:
    #                 num_uf[layer_idx][uncov_idx] += 1
    #         test_idx += 1
    #         '''

def construct_spectrum_matrices_class(model, trainable_layers,
                                correct_classifications, misclassifications,
                                layer_outs, activation_threshold, model_name, ft_tuple_str, activation_type, adv_name):

    save_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/trace/spectrum/spectrum_class/' + \
                model_name + '_' + str(adv_name) + '_' + ft_tuple_str + '_spectrum_' + str(activation_type) + '.npy'
    print('save_path: ', save_path)
    scores = []
    if os.path.exists(save_path):
        print("Found saved spectrum_num, skip serving")
        spectrum_num = np.load(save_path)
        print('spectrum_num: ', spectrum_num.shape)
        # scores.append(np.zeros(model.layers[0].output_shape[-1]))
        # print('scores: ', scores)

    else:
        num_ac = []; num_uc = []
        num_af = []; num_uf = []
        for tl in trainable_layers:
            print(model.layers[tl].output_shape)  # (None, 128)
            num_ac.append(np.zeros(model.layers[tl].output_shape[-1]))  # covered (activated) and failed
            num_uc.append(np.zeros(model.layers[tl].output_shape[-1]))  # uncovered (not activated) and failed
            num_af.append(np.zeros(model.layers[tl].output_shape[-1]))  # covered and succeeded
            num_uf.append(np.zeros(model.layers[tl].output_shape[-1]))  # uncovered and succeeded
            scores.append(np.zeros(model.layers[tl].output_shape[-1]))

        for tl in trainable_layers:
            layer_idx = trainable_layers.index(tl)
            all_neuron_idx = range(model.layers[tl].output_shape[-1])  #range(0, 128)
            test_idx = 0

            for l in layer_outs:   # for each test input
                for neuron_idx in all_neuron_idx:
                    # print('activation_threshold: ', activation_threshold[neuron_idx])
                    # print('l[neuron_idx]: ', l[neuron_idx])
                    # print('------------------')
                    if test_idx in correct_classifications and l[neuron_idx] > activation_threshold[neuron_idx]:
                        num_ac[layer_idx][neuron_idx] += 1   # activated & correct
                    elif test_idx in correct_classifications and l[neuron_idx] <= activation_threshold[neuron_idx]:
                        num_uc[layer_idx][neuron_idx] += 1   # unactivated & correct
                    elif test_idx in misclassifications and l[neuron_idx] > activation_threshold[neuron_idx]:
                        num_af[layer_idx][neuron_idx] += 1   # activated & fail
                    else:
                        num_uf[layer_idx][neuron_idx] += 1   # unactivated & fail

                test_idx += 1

        spectrum_num  = [num_ac, num_uc, num_af,num_uf]
        np.save(save_path, spectrum_num)

    return spectrum_num


    #         covered_idx   = list(np.where(l  > 0)[0])
    #         uncovered_idx = list(set(all_neuron_idx) - set(covered_idx))
    #         #uncovered_idx = list(np.where(l <= 0)[0])
    #         if test_idx  in correct_classifications:
    #             for cov_idx in covered_idx:
    #                 num_cs[layer_idx][cov_idx] += 1
    #             for uncov_idx in uncovered_idx:
    #                 num_us[layer_idx][uncov_idx] += 1
    #         elif test_idx in misclassifications:
    #             for cov_idx in covered_idx:
    #                 num_cf[layer_idx][cov_idx] += 1
    #             for uncov_idx in uncovered_idx:
    #                 num_uf[layer_idx][uncov_idx] += 1
    #         test_idx += 1
    #         '''


def neuron_albation(model, trainable_layers, neuron_idx):
    #  w,b = layer.get_weights()
    weights = model.layers[trainable_layers[0]].get_weights()
    # print('weights: ', weights[0].shape)  #weights:  (1024, 512)
    # print('weights: ', weights[1].shape)  #weights:  (512,)

    for idx in neuron_idx:
        weights[0][:, idx] = - weights[0][:, idx]
        weights[1][idx] = 0

    model.layers[trainable_layers[0]].set_weights(weights)

    return model



#ConvNet_CIFAR 10: dense_20 [index: 20]
#ConvNet_mnist: dense_5 [index: 7]
def relevance_analysis(model_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/trace/relevance_analysis/'
    relevance_path = root_path + model_name + '_' + 'dense_5' + '_misc_relevance.npy'
    # relevance_path = root_path + 'fashion_mnist_fc1_misc_relevance.npy'
    relevance_score = np.load(relevance_path)
    return  relevance_score

def relevance_all_analysis(model_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/trace/relevance_analysis/'
    relevance_path = root_path + model_name + '_' + 'dense' + '_all_relevance.npy'
    # relevance_path = root_path + 'fashion_mnist_fc1_misc_relevance.npy'
    relevance_score = np.load(relevance_path)
    return  relevance_score

def gradient_analysis(model_name, train_ats, train_misc_idx):

    train_misc_ats = train_ats[train_misc_idx]
    print('ats: ', train_misc_ats.shape)
    cumulative_activation = train_misc_ats.sum(axis = 0)
    print('cumulative_activation: ', cumulative_activation.shape)
    return  cumulative_activation

def find_EI_neurons_gradients(suspiciousness, misc_outputs, susp_num):
    suspiciousness = np.array(suspiciousness)
    indices = np.where(np.isnan(suspiciousness))
    suspiciousness[suspiciousness == 0] = 1
    np.nan_to_num(suspiciousness, nan=1, copy=False)

    gradients = misc_outputs / suspiciousness
    norm = gradients / gradients.max()

    all_sort_index = np.argsort(-norm)
    suspiciousness_neuron_idx = np.setdiff1d(all_sort_index, indices, True)
    suspicious_neuron_idx = suspiciousness_neuron_idx[:susp_num]
    return suspicious_neuron_idx

def find_EI_neurons_gradients_variant(suspiciousness, misc_outputs, susp_num):
    suspiciousness = np.array(suspiciousness)
    indices =np.where(np.isnan(suspiciousness))
    suspiciousness[suspiciousness==0]=1
    np.nan_to_num(suspiciousness, nan=1, copy=False)

    gradients = misc_outputs / suspiciousness
    norm = gradients / gradients.max()
    norm = (0.5 + norm) * suspiciousness

    all_sort_index = np.argsort(-norm)
    suspiciousness_neuron_idx = np.setdiff1d(all_sort_index,indices,True)
    suspicious_neuron_idx = suspiciousness_neuron_idx[:susp_num]

    return suspicious_neuron_idx

def find_EI_neurons (suspiciousness, relevances, susp_num):
    relevances = relevances.flatten()
    # print('relevances: ', relevances)

    suspiciousness = np.array(suspiciousness)
    # print('suspiciousness: ', suspiciousness)

    gradients = relevances / suspiciousness
    norm = gradients/ gradients.max()
    # norm = (0.5 + norm) * suspiciousness
    suspicious_neuron_idx = np.argsort(-norm)[:susp_num]

    return suspicious_neuron_idx


def find_EI_neurons_multiply (suspiciousness, relevances, susp_num):
    relevances = relevances.flatten()
    # print('relevances: ', relevances)

    suspiciousness = np.array(suspiciousness)
    # print('suspiciousness: ', suspiciousness)

    norm_relevance = relevances/relevances.max()
    norm_suspiciousness = suspiciousness/suspiciousness.max()

    norm = norm_relevance * norm_suspiciousness
    # norm = (0.5 + norm) * suspiciousness
    suspicious_neuron_idx = np.argsort(-norm)[:susp_num]

    return suspicious_neuron_idx

def find_EI_neurons_variant (suspiciousness, relevances, susp_num):
    relevances = relevances.flatten()
    # print('relevances: ', relevances)

    suspiciousness = np.array(suspiciousness)
    # print('suspiciousness: ', suspiciousness)

    gradients = relevances / suspiciousness
    norm = gradients / gradients.max()
    norm = (0.5 + norm) * suspiciousness
    # norm = (0.5 + norm) * suspiciousness
    suspicious_neuron_idx = np.argsort(-norm)[:susp_num]

    return suspicious_neuron_idx

# def find_EI_neurons_gradients (suspiciousness, activation, susp_num):
#
#     suspicious_neuron_idx = np.argsort(activation)[-susp_num:]
#     print('activation: ', activation)
#     print('suspicious_neuron_idx: ', suspicious_neuron_idx)
#     return suspicious_neuron_idx

def write_results_to_file_inconsist(all_results_incon, model_name, adv_name, susp_num, random, activation_type):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    for c in range(len(susp_num)):
        sheet1.write(0, c + 1, susp_num[c])

    i = 1

    for cs in all_results_incon:
        print('cs: ', cs)
        des_incon= all_results_incon[cs]

        for key in des_incon:
            sheet1.write(i, 0, key)
            column = 1
            for acc in des_incon[key]:
                sheet1.write(i, column, acc)
                column += 1
            i = i + 1
        i = i + 1

    f.save('E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/compound/cifar/'+
           model_name + '_' + adv_name +  '_' + random +  '_' + activation_type + '.xls')

def write_results_to_file(all_results_acc, all_results_incon, model_name, adv_name, susp_num):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    for c in range(len(susp_num)):
        sheet1.write(0, c + 1, susp_num[c])

    i = 1
    for cs in all_results_acc:
        print('cs: ', cs)
        des_acc = all_results_acc[cs]  # {des: [accuracy] }

        for key in des_acc:
            sheet1.write(i, 0, key)
            column = 1
            for acc in des_acc[key]:  # des[key]是存储accuracy的list
                sheet1.write(i, column, acc)
                column += 1
            i = i + 1
        i = i + 1

    for cs in all_results_incon:
        print('cs: ', cs)
        des_incon= all_results_incon[cs]

        for key in des_incon:
            sheet1.write(i, 0, key)
            column = 1
            for acc in des_incon[key]:
                sheet1.write(i, column, acc)
                column += 1
            i = i + 1
        i = i + 1

    f.save('E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/adv_file/svhn_vgg/'+ model_name + '_' + adv_name + '_histogram_results.xls')

def count_num(compound_x_test_ats, EI_neuron_idx, select_sample_index, activation_threshold):

    number = 0
    selectedX_ats = compound_x_test_ats[select_sample_index]   #(100,512)

    for idx in EI_neuron_idx:  #[20,30,22,29,19...]
        for inp in selectedX_ats:
            if inp[idx] > activation_threshold[idx]:
                number = number +1

    return number

def count_diversity(compound_x_test_ats, EI_neuron_idx):
    return EI_neuron_idx



def cone_of_influence_analysis(model, dominants):

    hidden_layers = [l for l in dominants.keys() if len(dominants[l]) > 0]
    target_layer = max(hidden_layers)

    scores = []
    for i in range(1, target_layer+1):
        scores.append(np.zeros(model.layers[i].output_shape[1]))
    for i in range(2, target_layer + 1)[::-1]:
        for j in range(model.layers[i].output_shape[1]):
            for k in range(model.layers[i - 1].output_shape[1]):
                relevant_weights = model.layers[i].get_weights()[0][k]
                if (j in dominants[i] or scores[i-1][j] > 0) and relevant_weights[j] > 0:
                    scores[i-2][k] += 1
                elif (j in dominants[i] or scores[i-1][j] > 0) and relevant_weights[j] < 0:
                    scores[i-2][k] -= 1
                elif j not in dominants[i] and scores[i-1][j] < 0 and relevant_weights[j] > 0:
                    scores[i-2][k] -= 1
                elif j not in dominants[i] and scores[i-1][j] < 0 and relevant_weights[j] < 0:
                    scores[i-2][k] += 1
    print(scores)
    return scores


def weight_analysis(model, target_layer):
    threshold_weight = 0.1
    deactivatables = []
    for i in range(2, target_layer + 1):
        for k in range(model.layers[i - 1].output_shape[1]):
            neuron_weights = model.layers[i].get_weights()[0][k]
            deactivate = True
            for j in range(len(neuron_weights)):
                if neuron_weights[j] > threshold_weight:
                    deactivate = False

            if deactivate:
                deactivatables.append((i,k))

    return deactivatables

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

def get_layers_compound_ats( model, X_train, layer_names, dataset, cs):
    save_path =  'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/trace/ats/'
    saved_train_path = _get_saved_path(save_path, dataset, 'compound'+ str(cs), layer_names)

    if os.path.exists(saved_train_path[0]):
        print("Found saved {} ATs, skip serving".format('compound'+ str(cs)))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            X_train,
            layer_names,
            saved_train_path,
            batch_size=128,
            is_classification=True,
            num_proc=10,
        )
        print("train ATs is saved at " + saved_train_path[0])

    return train_ats,train_pred

def get_layers_ats( model, X_train, layer_names, dataset):
    save_path =  'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/trace/ats/'
    saved_train_path = _get_saved_path(save_path, dataset, "train", layer_names)

    if os.path.exists(saved_train_path[0]):
        print('saved_train_path[0]: ', saved_train_path[0])
        print("Found saved {} ATs, skip serving".format("train"))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            X_train,
            layer_names,
            saved_train_path,
            batch_size=128,
            is_classification=True,
            num_proc=10,
        )
        print("train ATs is saved at " + saved_train_path[0])

    return train_ats,train_pred

def get_ats( model, dataset, layer_names, save_path, batch_size=128, is_classification=True, num_proc=10):

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    if is_classification:
        p = Pool(num_proc)  # Pool类可以提供指定数量的进程供用户调用
        pred = model.predict(dataset, batch_size=batch_size, verbose=1)

        if len(layer_names) == 1:  # 计算coverage的只有一层
            layer_outputs = [temp_model.predict(dataset, batch_size=batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)

        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):  # (1, 60000, 4, 4, 12)
            if layer_output[0].ndim == 3:
                layer_matrix = np.array(
                    p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
                )
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred

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
    return np.array(X_corr), np.array(Y_corr), np.array(X_misc), np.array(Y_misc), X_corr_idx, X_misc_idx

def find_unique_fault_type(train_misc_idx, Y_train, Y_train_misc):
    index = 0
    true_label_list = []
    misc_label_list = []

    for misc_idx in train_misc_idx:
        real_label = np.argmax(Y_train[misc_idx])
        true_label_list.append(real_label)
        misc_label = np.argmax(Y_train_misc[index])
        misc_label_list.append(misc_label)
        index = index + 1

    fault_zip = zip(true_label_list, misc_label_list)
    unique_fault_zip = list(set([i for i in fault_zip]))

    index = 0
    ft_results = {}
    for ft in unique_fault_zip:
        ft_results[ft] = []
    for misc_idx in train_misc_idx:
        real_label = np.argmax(Y_train[misc_idx])
        misc_label = np.argmax(Y_train_misc[index])
        index = index + 1
        fault_type_tuple = (real_label, misc_label)
        for ft in unique_fault_zip:
            if fault_type_tuple == ft:
                ft_results[ft].append(misc_idx)

    return ft_results

def get_class_matrix(train_pred):

    class_matrix = {}  # 输入为训练数据集，模型的输出，每个类对应的数据下标
    all_idx = []
    for i, label in enumerate(train_pred):
        if label.argmax(axis=-1) not in class_matrix:  # label.argmax表示返回最大值的索引
            class_matrix[label.argmax(axis=-1)] = []
        class_matrix[label.argmax(axis=-1)].append(i)
        all_idx.append(i)

    return class_matrix, all_idx

def compute_inconsistency(model, X, Y):

    X_misc_idx = []
    preds = model.predict(X)  # np.expand_dims(x,axis=0))

    for idx, pred in enumerate(preds):
        if np.argmax(pred) != np.argmax(Y[idx]):
            X_misc_idx.append(idx)

    ratio = len(X_misc_idx)/X.shape[0]

    return ratio

def find_all_fault_type(model, X_train_misc, Y_train_misc):
    Y_pred = model.predict(X_train_misc)
    Y_pred_class = np.argmax(Y_pred, axis=1)   #预测标签
    Y_test_class = np.argmax(Y_train_misc, axis=1)  #真实标签
    fault_zip = zip(Y_test_class, Y_pred_class)   # 包含重复的
    unique_fault_zip = list(set([i for i in fault_zip]))
    return unique_fault_zip


def get_configuration(dataset):

    if dataset == 'cifar10_m2':
        adv_name = 'fgsm'
        com_size = [10000]
        susp_num = [0.2]
        trainable_layers = [20]
        return adv_name, com_size, susp_num, trainable_layers

    # data_path = 'adv_file/cifar10/cifar_%s_compound_%s.npz' % (adv_name, str(cs))
def load_compound_mnist(cs, adv_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/'
    data_path = 'adv_file/mnist_ConvNet/ConvNet_mnist_Adv_%s_compound_%s.npz' % (adv_name, str(cs))
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    print('x_dest:', x_dest.shape)

    return x_dest, y_dest


# compound adversarial examples
def load_compound_mnist_convnet_img_adv(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
    data_path = 'adv_file/mnist/ConvNet_mnist_adv_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest

def load_compound_mnist_lenet5(cs, adv_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/'
    data_path = 'adv_file/mnist_lenet5/mnist_lenet5_%s_compound_%s.npz' % (adv_name, str(cs))
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    print('x_dest:', x_dest.shape)

    return x_dest, y_dest

def load_compound_mnist_lenet5_img_trans(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/compound/'
    data_path = 'mnist/mnist_lenet5_imgTrans_compound_%s_%s.npz' %(str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    print('x_dest:', x_dest.shape)

    return x_dest, y_dest

# compound adversarial examples
def load_compound_mnist_lenet5_img_adv(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
    data_path = 'adv_file/mnist_lenet5/mnist_lenet5_adv_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest

def load_compound_svhn(cs, adv_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/'
    data_path = 'adv_file/svhn/ConvNet_svhn_%s_compound_%s.npz' % (adv_name, str(cs))
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest

def load_compound_svhn_convnet_img_adv(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
    data_path = 'adv_file/svhn/ConvNet_svhn_adv_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest

# compound image transformation
def load_compound_svhn_img_trans(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/compound/'
    data_path = 'svhn/ConvNet_svhn_imgTrans_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest

# compound adversarial examples
def load_compound_svhn_vgg_img_adv(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
    data_path = 'adv_file/svhn_vgg/svhn_vgg_adv_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest


# 没有compound前的对抗样本文件
def load_compound_svhn_vgg(cs, adv_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/'
    data_path = 'adv_file/svhn_vgg/vgg_svhn_%s_compound_%s.npz' % (adv_name, str(cs))
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest

def load_compound_CIFAR10_vgg(cs, adv_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/'
    data_path = 'adv_file/cifar10_vgg/cifar10_vgg_%s_compound_%s.npz' % (adv_name, str(cs))
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    print('x_dest:', x_dest.shape)
    print('y_dest:', y_dest.shape)

    return x_dest, y_dest

def load_compound_cifar_convnet_img_adv(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
    data_path = 'adv_file/cifar10/cifar_adv_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest


def load_compound_cifar100_vgg_img_adv(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/adv_file/cifar100_vgg/'
    data_path = 'cifar100_vgg_adv_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 100)

    return x_dest, y_dest

def load_compound_CIFAR10(cs, adv_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/'
    data_path = 'adv_file/cifar10/before/cifar_%s_compound_%s.npz' % (adv_name, str(cs))
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    print('x_dest:', x_dest.shape)
    print('y_dest:', y_dest.shape)

    return x_dest, y_dest

def load_compound_img_trans_CIFAR10(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/compound/'
    data_path = '/cifar/cifar_imgTrans_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    print('x_dest:', x_dest.shape)
    print('y_dest:', y_dest.shape)

    return x_dest, y_dest


def load_compound_cifar_vgg_img_adv(cs, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
    data_path = 'adv_file/cifar_vgg/cifar_vgg_adv_compound_%s_%s.npz' % (str(cs), random)
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)

    return x_dest, y_dest


def load_compound_fashion(cs, adv_name):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelection/dataset/'
    data_path = 'adv_file/fashion/fashion_LeNet5_%s_compound_%s.npz' % (adv_name, str(cs))
    print('data_path: ', data_path)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    return x_dest, y_dest

def obtain_results(model_path, trainable_layers, neuron_idx, compound_x_test, compound_y_test):
    model = load_model(model_path)
    fup_model = neuron_albation(model, trainable_layers, neuron_idx)

    # _, test_accuracy = fup_model.evaluate(compound_x_test, compound_y_test)
    test_ratio = compute_inconsistency(fup_model, compound_x_test, compound_y_test)
    print('test_ratio: ', test_ratio)
    return test_ratio

def get_bin_fifty(train_ats):
    average_list = []

    for i in range(len(train_ats[0])):
        x = train_ats[:, i]
        n = len(x)
        nbins = 1000
        freq, bins = np.histogram(x, bins=nbins)
        total_average = 0
        high_index = np.argsort(-freq)[:50]

        for idx  in high_index:
            average = (bins[idx] + bins[idx-1])/2
            total_average = total_average + average
        average_list.append(total_average/50)
    return average_list

def get_bin_twenty(train_ats):

    average_list = []

    for i in range(len(train_ats[0])):
        x = train_ats[:, i]
        nbins = 1000
        freq, bins = np.histogram(x, bins=nbins)
        total_average = 0
        high_index = np.argsort(-freq)[:20]

        for idx  in high_index:
            average = (bins[idx] + bins[idx-1])/2
            total_average = total_average + average
        average_list.append(total_average/20)

    return average_list

def get_bin(train_ats):

    average_list = []

    for i in range(len(train_ats[0])):
        x = train_ats[:, i]
        nbins = 1000
        freq, bins = np.histogram(x, bins=nbins)
        total_average = 0
        high_index = np.argsort(-freq)[:100]

        for idx  in high_index:
            average = (bins[idx] + bins[idx-1])/2
            total_average = total_average + average
        average_list.append(total_average/100)

    return average_list
    #
    # ax = sns.distplot(x, bins=nbins,
    #                   hist=True,  # Whether to plot a (normed) histogram.
    #                   kde=False,
    #                   norm_hist=True,  # norm_hist = norm_hist or kde or (fit is not None); 如果为False且kde=False, 则高度为频数
    #                   #                 kde_kws={"label": "density_est_by_sns",
    #                   #                          "bw": bin_w}
    #                   )
    # ax.grid(True)
    # ax.set_yticks(np.arange(0.16, step=0.01))
    # plt.show()



    # from sklearn.neighbors import KernelDensity
    # import matplotlib.pyplot as plt
    # model = KernelDensity()
    #
    # for n in refined_ats:
    #     model.fit(n.reshape(-1,1))
    #     score = model.score_samples(n.reshape(-1,1))
    #     plt.fill(n, np.exp(score), c='cyan')
    #     plt.show()
    #     break

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # for n in refined_ats:
    #     sns.kdeplot(refined_ats[20], color="b", linewidth=2, fill=True)
    #     plt.show()
    #     break
