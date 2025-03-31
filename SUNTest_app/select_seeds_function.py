import random
import numpy as np
from tensorflow import keras
import tensorflow as tf

def select_seeds(test_ats, final_EI_neuron_idx, seeds_num, test_corr_idx):

    EI_ats_test_orig = test_ats[:, final_EI_neuron_idx]
    EI_ats_sum = np.sum(EI_ats_test_orig, axis=1)

    # 使用argsort函数获取元素值从小到大的下标
    sorted_indices = np.argsort(EI_ats_sum)
    temp_set = set(test_corr_idx)
    sorted_indices = [x for x in sorted_indices if x in temp_set]

    # 取得最高的前seeds num个元素的下标
    top_indices = sorted_indices[-seeds_num:]

    return top_indices

def select_seeds_art(test_ats, final_EI_neuron_idx, seeds_num, test_corr_idx):
    EI_ats_test_orig = test_ats[:, final_EI_neuron_idx]
    top_index = np.flipud(select_seeds(test_ats, final_EI_neuron_idx, seeds_num, test_corr_idx))[0]
    max_EI_ats = EI_ats_test_orig[top_index]
    temp_corr_idx = test_corr_idx

    top_indices = []
    top_indices.append(top_index)

    # 计算max_EI_ats 与EI_ats_test_orig中各元素间的欧式距离，找到与max_EI_ats距离最大的
    for num in range(1, seeds_num):
        distances = np.linalg.norm(EI_ats_test_orig - max_EI_ats, axis=1)
        distance_indices = np.argsort(distances)
        print('distance_indices: ', distance_indices)

        temp_set = set(temp_corr_idx)
        top_index = [x for x in distance_indices if x in temp_set][-1]
        top_indices.append(top_index)

        # temp_corr_idx = test_corr_idx，需要删除已经被选为seeds的样本
        elements_to_remove_set = set(top_indices)
        temp_corr_idx = [item for item in temp_corr_idx if item not in elements_to_remove_set]
        max_EI_ats = EI_ats_test_orig[top_index]

    return top_indices


def select_seeds_art_random(test_ats, final_EI_neuron_idx, seeds_num, test_corr_idx):
    top_index = random.sample(test_corr_idx, 1)[0]  # 第一个seed 随机选取，后面的seed 基于SUS信息ART选取
    EI_ats_test_orig = test_ats[:, final_EI_neuron_idx]
    max_EI_ats = EI_ats_test_orig[top_index]
    temp_corr_idx = test_corr_idx

    top_indices = []
    top_indices.append(top_index)

    # 计算max_EI_ats 与EI_ats_test_orig中各元素间的欧式距离，找到与max_EI_ats距离最大的
    for num in range(1, seeds_num):
        distances = np.linalg.norm(EI_ats_test_orig - max_EI_ats, axis=1)
        distance_indices = np.argsort(distances)
        print('distance_indices: ', distance_indices)

        temp_set = set(temp_corr_idx)
        top_index = [x for x in distance_indices if x in temp_set][-1]
        top_indices.append(top_index)
        print('top_indices： ', top_indices)

        # temp_corr_idx = test_corr_idx，需要删除已经被选为seeds的样本
        elements_to_remove_set = set(top_indices)
        temp_corr_idx = [item for item in temp_corr_idx if item not in elements_to_remove_set]
        max_EI_ats = EI_ats_test_orig[top_index]

    return top_indices


def select_seeds_ats_art(test_ats, seeds_num, test_corr_idx):
    top_index = random.sample(test_corr_idx, 1)[0]  # 第一个seed 随机选取，后面的seed 基于SUS信息ART选取
    max_EI_ats = test_ats[top_index]
    temp_corr_idx = test_corr_idx

    top_indices = []
    top_indices.append(top_index)

    # 计算max_EI_ats 与EI_ats_test_orig中各元素间的欧式距离，找到与max_EI_ats距离最大的
    for num in range(1, seeds_num):
        distances = np.linalg.norm(test_ats - max_EI_ats, axis=1)
        distance_indices = np.argsort(distances)
        print('distance_indices: ', distance_indices)

        temp_set = set(temp_corr_idx)
        top_index = [x for x in distance_indices if x in temp_set][-1]
        top_indices.append(top_index)
        print('top_indices： ', top_indices)

        # temp_corr_idx = test_corr_idx，需要删除已经被选为seeds的样本
        elements_to_remove_set = set(top_indices)
        temp_corr_idx = [item for item in temp_corr_idx if item not in elements_to_remove_set]
        max_EI_ats = test_ats[top_index]

    return top_indices


def select_seeds_ats_sus_art(test_ats, final_EI_neuron_idx, seeds_num, test_corr_idx):
    top_index = np.flipud(select_seeds(test_ats, final_EI_neuron_idx, seeds_num, test_corr_idx))[0]
    max_ats = test_ats[top_index]
    temp_corr_idx = test_corr_idx

    top_indices = []
    top_indices.append(top_index)

    # 计算max_EI_ats 与EI_ats_test_orig中各元素间的欧式距离，找到与max_EI_ats距离最大的
    for num in range(1, seeds_num):
        distances = np.linalg.norm(test_ats - max_ats, axis=1)
        distance_indices = np.argsort(distances)
        print('distance_indices: ', distance_indices)

        temp_set = set(temp_corr_idx)
        top_index = [x for x in distance_indices if x in temp_set][-1]
        top_indices.append(top_index)
        print('top_indices： ', top_indices)

        # temp_corr_idx = test_corr_idx，需要删除已经被选为seeds的样本
        elements_to_remove_set = set(top_indices)
        temp_corr_idx = [item for item in temp_corr_idx if item not in elements_to_remove_set]
        max_ats = test_ats[top_index]

    return top_indices

# 从各类别的样本中筛选sus outputs值最高的样本
def select_seeds_sus_class(test_ats, final_EI_neuron_idx, seeds_num, test_corr_idx, X_test, Y_test):
    class_num = 10
    select_from_each_class = int(seeds_num/class_num)
    class_matrix = {}
    EI_ats_test_orig = test_ats[:, final_EI_neuron_idx]
    EI_ats_sum = np.sum(EI_ats_test_orig, axis=1)
    sorted_EI_ats_idx = np.flip(np.argsort(EI_ats_sum))

    top_indices = []

    for idx in range(0, len(X_test)):
        idx_label = np.argmax(Y_test[idx])  # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
        if idx_label not in class_matrix:
            class_matrix[idx_label] = []
        class_matrix[idx_label].append(idx)

    for key_label in class_matrix:
        idx_each_label = class_matrix[key_label]
        corr_idx_each_label = [x for x in idx_each_label if x in test_corr_idx]
        sorted_test_idx = [x for x in sorted_EI_ats_idx if x in corr_idx_each_label]
        select_idx_each_label = sorted_test_idx[:select_from_each_class]
        top_indices.append(select_idx_each_label)

    from itertools import chain
    top_indices = list(chain(*top_indices))
    return top_indices


# 从各类别的样本中筛选adpative 筛选样本
def select_seeds_sus_class_art(test_ats, final_EI_neuron_idx, seeds_num, test_corr_idx, X_test, Y_test):
    class_num = 10
    select_from_each_class = int(seeds_num/class_num)  # 每个class中需要筛选出的样本的数量
    class_matrix = {}
    EI_ats_test_orig = test_ats[:, final_EI_neuron_idx]
    EI_ats_sum = np.sum(EI_ats_test_orig, axis=1)
    sorted_EI_ats_idx = np.flip(np.argsort(EI_ats_sum))

    top_indices = []

    for idx in range(0, len(X_test)):
        idx_label = np.argmax(Y_test[idx])  # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
        if idx_label not in class_matrix:
            class_matrix[idx_label] = []
        class_matrix[idx_label].append(idx)

    for key_label in class_matrix:
        idx_each_label = class_matrix[key_label]
        corr_idx_each_label = [x for x in idx_each_label if x in test_corr_idx]
        top_sus_test_idx = [x for x in sorted_EI_ats_idx if x in corr_idx_each_label][0]  # 各class中EI ats值最高的样本
        max_ats = test_ats[top_sus_test_idx]
        top_indices.append(top_sus_test_idx)

        for num in range (1, select_from_each_class):
            distances = np.linalg.norm(test_ats - max_ats, axis=1)
            distance_indices = np.argsort(distances)

            temp_set = set(corr_idx_each_label)
            top_index = [x for x in distance_indices if x in temp_set][-1]
            top_indices.append(top_index)

            # temp_corr_idx = test_corr_idx，需要删除已经被选为seeds的样本
            elements_to_remove_set = set(top_indices)
            corr_idx_each_label = [item for item in corr_idx_each_label if item not in elements_to_remove_set]
            max_ats = test_ats[top_index]

    return top_indices

def select_seeds_random_V2(seeds_num, test_corr_idx, select_lst):
    import random
    print('test_corr_idx:', len(test_corr_idx))
    print('select_lst: ', len(select_lst))
    union = [value for value in select_lst if value in test_corr_idx]
    print('union: ', len(union))
    random_index = random.sample(union, seeds_num)
    return random_index

def select_seeds_random(seeds_num, test_corr_idx):
    import random
    random_index = random.sample(test_corr_idx, seeds_num)
    return random_index

def select_seeds_deepgini_V2(model, X_test, seeds_num, test_corr_idx, select_lst):
    # 取gini不纯度高的
    all_gini=[]

    for idx in range(X_test.shape[0]):
        print('idx:', idx)
        temp_img = X_test[[idx]]
        logits = model(temp_img)

        pro_sum = 0
        for pro in logits[0]:
            pro_sum = pro_sum + pro*pro
        t_gini = 1 - pro_sum
        all_gini.append(t_gini)

    # gini_idx时升序排列的索引数组
    gini_idx = np.argsort(all_gini)
    temp_set = set(test_corr_idx)
    temp_select_set = set(select_lst)
    sorted_indices = [x for x in gini_idx if x in temp_set]
    sorted_indices = [x for x in sorted_indices if x in temp_select_set]

    # 取得最高的前seeds num个元素的下标
    top_indices = sorted_indices[-seeds_num:]

    return top_indices

    # top_indices.append(top_index)


def select_seeds_deepgini(model, X_test, seeds_num, test_corr_idx):
    # 取gini不纯度高的
    all_gini=[]

    for idx in range(X_test.shape[0]):
        temp_img = X_test[[idx]]
        logits = model(temp_img)

        pro_sum = 0
        for pro in logits[0]:
            pro_sum = pro_sum + pro*pro
        t_gini = 1 - pro_sum
        all_gini.append(t_gini)

    # gini_idx时升序排列的索引数组
    gini_idx = np.argsort(all_gini)
    temp_set = set(test_corr_idx)
    sorted_indices = [x for x in gini_idx if x in temp_set]

    # 取得最高的前seeds num个元素的下标
    top_indices = sorted_indices[-seeds_num:]

    return top_indices

    # top_indices.append(top_index)

def fol_Linf(model, X_test, ep, Y_test, adv_file):
    xi = adv_file
    x = X_test
    y = Y_test

    x, target = tf.Variable(x), tf.constant(y)
    fols = []
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(target, model(x))
        grads = tape.gradient(loss, x)
        grad_norm = np.linalg.norm(grads.numpy().reshape(x.shape[0], -1), ord=1, axis=1)
        grads_flat = grads.numpy().reshape(x.shape[0], -1)
        diff = (x.numpy() - xi).reshape(x.shape[0], -1)
        for i in range(x.shape[0]):
            i_fol = -np.dot(grads_flat[i], diff[i]) + ep * grad_norm[i]
            fols.append(i_fol)

    return np.array(fols)


def fol_L2(model, x, y):
    """
    x: perturbed inputs, shape of x: [batch_size, width, height, channel]
    y: ground truth, one hot vectors, shape of y: [batch_size, N_classes]
    """
    x, target = tf.Variable(x), tf.constant(y)
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(target, model(x))
        grads = tape.gradient(loss, x)
        grads_norm_L2 = np.linalg.norm(grads.numpy().reshape(x.shape[0], -1), ord=2, axis=1)

    return grads_norm_L2


def select_seeds_robotBe(model, X_test, Y_test, adv_file, seeds_num, test_corr_idx):
    ep = 1e-18
    grads_norm_L2 = fol_Linf(model, X_test, ep, Y_test, adv_file)

    # 根据样本的FOL值，从小到大排序的索引数组
    ranks = np.argsort(grads_norm_L2)
    print('ranks: ', ranks)
    temp_set = set(test_corr_idx)
    sorted_indices = [x for x in ranks if x in temp_set]

    h = seeds_num // 2
    result_idx = np.concatenate((sorted_indices[-h:], sorted_indices[:h]))
    print('result_idx:', result_idx)
    return result_idx

def select_seeds_robotBe_V2(model, X_test, Y_test, adv_file, seeds_num, test_corr_idx, select_lst):
    ep = 1e-18
    grads_norm_L2 = fol_Linf(model, X_test, ep, Y_test, adv_file)

    # 根据样本的FOL值，从小到大排序的索引数组
    ranks = np.argsort(grads_norm_L2)
    print('ranks: ', ranks)
    temp_set = set(test_corr_idx)
    temp_select_lst = set(select_lst)
    sorted_indices = [x for x in ranks if x in temp_set]
    sorted_indices = [x for x in sorted_indices if x in temp_select_lst]

    h = seeds_num // 2
    result_idx = np.concatenate((sorted_indices[-h:], sorted_indices[:h]))
    print('result_idx:', result_idx)
    return result_idx

def select_seeds_robotKM(model, X_test, Y_test, adv_file, seeds_num, test_corr_idx):
    ep = 1e-18
    grads_norm_L2 = fol_Linf(model, X_test, ep, Y_test, adv_file)
    print('grads_norm_L2: ', grads_norm_L2, grads_norm_L2.shape)
    ranks = np.argsort(grads_norm_L2)
    print('ranks: ', ranks)
    temp_set = set(test_corr_idx)
    ranks = [x for x in ranks if x in temp_set]

    n = seeds_num
    index = []
    k = 20
    section_w = len(ranks) // k
    section_nums = n // k   # 每个section中收取section_nums个元素
    indexes = random.sample(list(range(k)), section_nums)
    print('indexes:', indexes)
    for i in indexes:
        block = ranks[i * k: (i + 1) * k]
        index.append(block)

    result_idx = np.concatenate(np.array(index))
    return result_idx

def select_seeds_robotKM_V2(model, X_test, Y_test, adv_file, seeds_num, test_corr_idx, select_lst):
    ep = 1e-18
    grads_norm_L2 = fol_Linf(model, X_test, ep, Y_test, adv_file)
    print('grads_norm_L2: ', grads_norm_L2, grads_norm_L2.shape)
    ranks = np.argsort(grads_norm_L2)
    print('ranks: ', ranks)
    temp_set = set(test_corr_idx)
    temp_select_lst = set(select_lst)
    ranks = [x for x in ranks if x in temp_set]
    ranks = [x for x in ranks if x in temp_select_lst]

    n = seeds_num
    index = []
    k = 20
    section_w = len(ranks) // k
    section_nums = n // k   # 每个section中收取section_nums个元素
    indexes = random.sample(list(range(k)), section_nums)

    for i in indexes:
        block = ranks[i * k: (i + 1) * k]
        index.append(block)

    result_idx = np.concatenate(np.array(index))
    return result_idx