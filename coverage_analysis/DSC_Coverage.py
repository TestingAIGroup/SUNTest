import numpy as np
import random
def get_coverage(lower, upper, k, sa):

    print('sa value of all test inputs: ', len(sa), sa.sort)
    print('lower and upper: ', lower, upper)
    buckets  = np.digitize(sa, np.linspace(lower, upper, k))

    return len(list(set(buckets))) / float(k) * 100

def get_coverage_tn(lower, upper, k, sa):

    print('sa value of all test inputs: ', len(sa), sa.sort)
    print('lower and upper: ', lower, upper)
    buckets  = np.digitize(sa, np.linspace(lower, upper, k))
    tn_buckets = list(set(buckets))
    print('tn_buckets',tn_buckets)
    count = 0
    for i in tn_buckets:
        if i >= 100:
            count = count+1
    return count/ 400 *100


def get_LDSC_coverage(original_lid):

    upper_bound = 3; n_buckets = 500
    # print('real upper: ', np.amax(original_lid))
    original_coverage = get_coverage(np.amin(original_lid), upper_bound, n_buckets, original_lid)
    # print('original_coverage======', original_coverage)
    return original_coverage

def get_TNSC_coverage(original_lid):

    upper_bound = 3; n_buckets = 500
    print('real upper: ', np.amax(original_lid))
    original_coverage = get_coverage_tn(np.amin(original_lid), upper_bound, n_buckets, original_lid)
    print('original_coverage======', original_coverage)
    return original_coverage

def get_LSC_coverage(original_lsa):

    #mnist upper_bound = 2000, cifar upper_bound = 100
    upper_bound = 2000; n_buckets = 1000
    print('real upper and lower: ', np.amax(original_lsa), np.amin(original_lsa))
    original_coverage = get_coverage(np.amin(original_lsa), upper_bound, n_buckets, original_lsa)
    # print('original_coverage======', original_coverage)
    return original_coverage

def get_MDSC_coverage(original_mdsa):

    #mnist upper_bound = 2000, cifar upper_bound = 100
    upper_bound = 2000; n_buckets = 1000
    print('real upper and lower: ', np.amax(original_mdsa), np.amin(original_mdsa))
    original_coverage = get_coverage(np.amin(original_mdsa), upper_bound, n_buckets, original_mdsa)
    # print('original_coverage======', original_coverage)
    return original_coverage

def get_DSC_coverage(original_dsa, bound):

    upper_bound = bound; n_buckets = 1000
    lower_bound = 0
    print('real upper: ', np.amax(original_dsa))
    original_coverage = get_coverage(lower_bound, upper_bound, n_buckets, original_dsa)

    return original_coverage

def get_DeepGini_coverage(original_gini):
    upper_bound = 1 ;  n_buckets = 100
    print('real upper: ', np.amax(original_gini))

    original_coverage = get_coverage(np.amin(original_gini), upper_bound, n_buckets, original_gini)
    print('original_coverage======', original_coverage)
    return original_coverage


def get_var_coverage(original_gini):
    upper_bound = 1
    n_buckets = 2000
    real_upper_bound = np.amax(original_gini)
    print('real upper: ', np.amax(original_gini))

    original_coverage = get_coverage(0, real_upper_bound, n_buckets, original_gini)
    print('original_coverage======', original_coverage)
    return original_coverage

def find_nat_fault_type(model, X_test_misc, Y_test_misc):
    Y_pred = model.predict(X_test_misc)
    Y_pred_class = np.argmax(Y_pred, axis=1)   #预测标签
    Y_test_class = np.argmax(Y_test_misc, axis=1)  #真实标签
    fault_zip = zip(Y_test_class, Y_pred_class)   # 包含重复的
    unique_fault_zip = list(set([i for i in fault_zip]))
    return unique_fault_zip

if __name__ == "__main__":
    rs_dsc =[];   rs_ldsc=[]

    path = '/home/EI_Neurons/cifar10_convnet/dsa/'
    # path = '/home/EI_Neurons/cifar10_convnet/dsa/'
    # path ='/home/EI_Neurons/mnist_convnet/dsa/'
    repeat_times = 5
    seed_selection_strategy = ['Random', 'Deepgini', 'KM',  'BE' ]
    dsc_coverage = []
    upper_bound_lst = [5, 10]

    for seeds in seed_selection_strategy:

        for bound in upper_bound_lst:
            rs_dsc = []
            for repeat in range(0, repeat_times):
                # dsc_load = np.load(path + 'cifar10_vgg_moon_{}_{}_repeat.npy'.format(seeds, repeat))
                dsc_load = np.load(path + 'cifar10_convnet_adapt_NC_{}_{}_repeat.npy'.format(seeds, repeat))
                # cov = get_DSC_coverage(dsc_load)
                # print('cov: ', cov)
                # print('dsc_load: ', dsc_load)
                rs_dsc.append(get_DSC_coverage(dsc_load, bound))
                # rs_dsc.append(get_LSC_coverage(dsc_load))
            print('dsc_moon_{}_{}: '.format(seeds, bound), rs_dsc)

