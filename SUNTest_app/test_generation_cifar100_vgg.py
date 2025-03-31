import time
import xlwt
from keras.models import load_model
from tensorflow.keras.datasets import cifar10
from keras.utils import to_categorical
from GenSOTA.MOON.utils_gen import get_layers_ats, get_layers_new_img_ats
from GenSOTA.MOON.dflare_img_mutations import get_img_mutations
from GenSOTA.MOON.dflare_probability_img_mutations import ProbabilityImgMutations as ImgMutations
from GenSOTA.MOON.triggering_distances import TriggeringInfo
from GenSOTA.MOON.dflare_fitnessValue import DiffProbFitnessValue as FitnessValue
from GenSOTA.MOON.select_seeds_function import *
import os
from tensorflow.keras.datasets import cifar100

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# def test_gen(model, X_test):
def get_random_p(num_EI):
    ratio = 0.8
    total_num = round(num_EI * ratio)
    ran_p = np.zeros(num_EI)
    s = np.random.choice(range(num_EI), replace=False, size= total_num)
    for i in range(len(s)):
        ran_p[s[i]] = 1

    return ran_p


def predict_gen(model, new_img):
    gen_index = np.argmax(model(new_img))
    return gen_index

def ed(m, n):
 return np.sqrt(np.sum((m - n) ** 2))

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
    return np.array(X_corr), np.array(Y_corr), X_corr_idx


def summary_attack_results(seed_strategy, repeat, trigger_obj):
    summary_str = " "
    summary_str = summary_str + "seed_strategy: " + seed_strategy + "\n"
    summary_str = summary_str + "repeat times: " + str(repeat) + "\n"

    summary_str = summary_str + "len(all generated inputs): {}\n".format(str(sum(len(lst) for lst in trigger_obj.all_gen_inps.values())))
    summary_str = summary_str + "len(triggering inputs): {}\n".format(len(trigger_obj.trigger_input_archive))
    summary_str = summary_str + "len(triggering inputs ats): {}\n".format(len(trigger_obj.trigger_inputs_ats_archive))
    summary_str = summary_str + "all triggering inputs: {}\n".format(str(trigger_obj.all_trigger_inps_archive))
    summary_str = summary_str + "len(fault type): {}\n".format(len(trigger_obj.fault_type_archive))
    summary_str = summary_str + "fault type: {}\n".format(', '.join(trigger_obj.fault_type_archive))
    summary_str = summary_str + "average time: {}".format(str(trigger_obj.average_time_archive))

    print(summary_str)
    return summary_str


def write2excel(ALL_results, repeat_times, seed_selection_strategy, saved_path):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    seed_selection_strategy = ['Random', 'RobotBe', 'RobotKM', 'Deepgini', 'MCP']

    for repeat in range(0, repeat_times):
        i = 0 # i 控制seed strategy - 行; repeat 控制repeat times -列
        for seed in seed_selection_strategy:
            all_seed_strategy_summary_list = ALL_results[str(repeat) + seed]
            for seed_strategy_summary in all_seed_strategy_summary_list:
                # j 控制每个seed strategy 里面的行
                j = 0
                # 拆分后的summary元素列表
                seed_strategy_summary_split = seed_strategy_summary.split("\n")
                print('seed_strategy_summary_split: ', seed_strategy_summary_split)
                for j in range(0, len(seed_strategy_summary_split)):
                    line_idx = i * 11 + j
                    sheet1.write( line_idx, repeat, seed_strategy_summary_split[j] )
                    j = j + 1
                i = i + 1

    f.save(saved_path + 'MOON_fitC_results_Original_200_5_V1.xls')


def load_CIFAR100(one_hot=True):
    CLIP_MAX = 0.5
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    X_train = X_train.astype("float32")
    X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
    X_test = X_test.astype("float32")
    X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)

    if one_hot:
        y_train = to_categorical(y_train, num_classes=100)
        y_test = to_categorical(y_test, num_classes=100)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_CIFAR100()  # 在utils中修改

    model_path = '/workspace/model/cifar100_vgg_v1.h5'
    model = keras.models.load_model(model_path)
    model.summary()

    trainable_layers = [41]      #cifar10_convnet 和 cfar10_vgg ： target layer都是20
    layer_names = [model.layers[idx].name for idx in trainable_layers]
    model.summary()

    # obtain data
    dataset = 'cifar100_vgg'
    X_test_corr, Y_test_corr, test_corr_idx =  filter_correct_classifications(model, X_test, Y_test)

    # get outputs of EI neurons
    saved_path = '/workspace/EI_Neurons/cifar100_vgg/EI_neurons/'
    EI_neurons_idx = np.load(saved_path + 'cifar100_vgg_ochiai_EI_neurons_idx.npy')
    print('EI_neurons_idx: ', EI_neurons_idx)
    saved_file_path = '/workspace/EI_Neurons/cifar100_vgg/MOON/MOON-Original/'

    test_ats, test_pred = get_layers_ats(model, X_test, layer_names, dataset)
    print('test_ats:', test_ats.shape)

    random_seed = 1
    seeds_num = 200   #种子样本数量
    iteration_max = 5  #迭代次数
    timeout_seconds = 10
    repeat_times = 3 # 重复实验次数
    select_lst = np.load('/workspace/EI_Neurons/adv_file/select_lst.npy')
    remain_lst = np.load('/workspace/EI_Neurons/adv_file/remain_lst.npy')

    seed_root_path = '/workspace/EI_Neurons/cifar100_vgg/seed_index/'
    # 记录所有结果
    ALL_results = {}
    All_Operator_Results = {}

    # 选择fitness function 作为引导，fitA和fitB分别表示只使用fitness function1和fitness function2；fitC表示同时使用两个fitness function
    fitness_mode = 'fitC'
    seed_selection_strategy = ['Be', 'KM']

    Random_seed_index = {}
    Random_seed_index[0] = np.load(seed_root_path + 'Random_seed_index_V1_0.npy')
    Random_seed_index[1] = np.load(seed_root_path + 'Random_seed_index_V1_1.npy')
    Random_seed_index[2] = np.load(seed_root_path + 'Random_seed_index_V1_2.npy')
    Random_seed_index[3] = np.load(seed_root_path + 'Random_seed_index_V1_3.npy')
    Random_seed_index[4] = np.load(seed_root_path + 'Random_seed_index_V1_4.npy')

    BE_seed_index = {}
    BE_seed_index[0] = np.load(seed_root_path + 'Be_seed_index_V2_0.npy')
    BE_seed_index[1] = np.load(seed_root_path + 'Be_seed_index_V2_1.npy')
    BE_seed_index[2] = np.load(seed_root_path + 'Be_seed_index_V2_2.npy')
    BE_seed_index[3] = np.load(seed_root_path + 'Be_seed_index_V2_3.npy')
    BE_seed_index[4] = np.load(seed_root_path + 'Be_seed_index_V2_4.npy')

    KM_seed_index = {}
    KM_seed_index[0] = np.load(seed_root_path + 'KM_seed_index_V2_0.npy')
    KM_seed_index[1] = np.load(seed_root_path + 'KM_seed_index_V2_1.npy')
    KM_seed_index[2] = np.load(seed_root_path + 'KM_seed_index_V2_2.npy')
    KM_seed_index[3] = np.load(seed_root_path + 'KM_seed_index_V2_3.npy')
    KM_seed_index[4] = np.load(seed_root_path + 'KM_seed_index_V2_4.npy')

    Deepgini_seed_index = {}
    Deepgini_seed_index[0] = np.load(seed_root_path + 'Deepgini_seed_index_V1_0.npy')
    Deepgini_seed_index[1] = np.load(seed_root_path + 'Deepgini_seed_index_V1_1.npy')
    Deepgini_seed_index[2] = np.load(seed_root_path + 'Deepgini_seed_index_V1_2.npy')
    Deepgini_seed_index[3] = np.load(seed_root_path + 'Deepgini_seed_index_V1_3.npy')
    Deepgini_seed_index[4] = np.load(seed_root_path + 'Deepgini_seed_index_V1_4.npy')

    MCP_seed_index = {}
    MCP_seed_index[0] = np.load(seed_root_path + 'mcp_seed_index_V1_0.npy')
    MCP_seed_index[1] = np.load(seed_root_path + 'mcp_seed_index_V1_1.npy')
    MCP_seed_index[2] = np.load(seed_root_path + 'mcp_seed_index_V1_2.npy')
    MCP_seed_index[3] = np.load(seed_root_path + 'mcp_seed_index_V1_3.npy')
    MCP_seed_index[4] = np.load(seed_root_path + 'mcp_seed_index_V1_4.npy')

    for repeat in range(0, repeat_times):
        # ALL_results[str(repeat) + 'Random'] =[]
        ALL_results[str(repeat) + 'RobotBe'] = []
        ALL_results[str(repeat) + 'RobotKM'] = []
        # ALL_results[str(repeat) + 'Deepgini'] = []
        # ALL_results[str(repeat) + 'MCP'] = []

        for seed_strategy in seed_selection_strategy:
            trigger_obj = TriggeringInfo()

            if seed_strategy == 'Random':
                top_indices = Random_seed_index[repeat]
                print('Random strategy selected seeds, top_indices:', top_indices)

            elif seed_strategy == 'Deepgini':
                top_indices = Deepgini_seed_index[repeat]
                print('Deepgini strategy selected seeds, top_indices:', top_indices)

            elif seed_strategy == 'Be':
                top_indices = BE_seed_index[repeat]
                print('Robot-Best strategy selected seeds, top_indices:', top_indices)

            elif seed_strategy == 'KM':
                top_indices = KM_seed_index[repeat]
                print('Robot-KM strategy selected seeds, top_indices:', top_indices)

            elif seed_strategy == 'MCP':
                top_indices = MCP_seed_index[repeat]
                print('MCP strategy selected seeds, top_indices:', top_indices)

            for idx in top_indices:
                print('start testing with seed------------------:', idx)
                seed_start_time = time.time()

                # 记录每个seed 生成的全部inputs
                trigger_obj.all_gen_inps[idx] = []
                # 记录每个seed 生成的triggering inputs
                trigger_inp_seed = []
                # 记录每个seed 开始迭代，直到发现第一个triggering input的时间
                find_trigger_inp_st = time.time()
                trigger1st = False

                raw_seed_input = X_test[idx]
                raw_seed_label = np.argmax(Y_test[idx])   # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

                mutation = get_img_mutations()

                p_mutation = ImgMutations(mutation, random_seed)

                last_mutation_operator = None
                latest_img = np.copy(raw_seed_input)
                best_fitness_value = FitnessValue(0)

                for iteration in range(1, iteration_max + 1):

                    print("Start Iteration-------------------{}".format(iteration))

                    m = p_mutation.choose_mutator(last_mutation_operator)
                    m.total += 1   # mutator被选中的次数
                    print('Mutator: ', m.name)

                    new_img = m.mut(np.copy(latest_img)).reshape(-1,32,32,3)
                    mutated_seed_label = predict_gen(model, new_img)
                    trigger_obj.all_gen_inps[idx].append(new_img)   # 所有生成的inps都存储

                    if mutated_seed_label != raw_seed_label:
                        print("Found triggering inputs== org: {} vs cps: {}".format(raw_seed_label, mutated_seed_label))

                        trigger_obj.trigger_input_archive.append(new_img)
                        trigger_inp_seed.append(new_img)
                        new_test_ats, _ = get_layers_new_img_ats(model, new_img, layer_names, dataset)
                        trigger_obj.trigger_inputs_ats_archive.append(new_test_ats)

                        # mutator 更新
                        new_fault_type = str(raw_seed_label) + str(mutated_seed_label)
                        m.trigger_num +=1
                        if new_fault_type not in trigger_obj.fault_type_archive:
                            m.fault_type_num +=1  # mutation 引入新的fault type的能力
                            trigger_obj.fault_type_archive.append(new_fault_type)
                        # m.delta_bigger_than_zero += 1

                    else:
                        # 1st fitness value，获取新生成的img的EI neuron outputs
                        ran_p = get_random_p(len(EI_neurons_idx))
                        temp = ran_p * EI_neurons_idx
                        final_EI_neuron_idx = temp[temp != 0].astype(int)

                        new_test_ats, new_test_pred = get_layers_new_img_ats(model, new_img, layer_names, dataset)
                        img_EI_neurons_ats = new_test_ats[:, final_EI_neuron_idx]
                        img_EI_neurons_ats_value = np.sum(new_test_ats[:, final_EI_neuron_idx], axis=1)  # [11.986867]

                        # 2nd fitness value
                        triggering_dis = []
                        if not trigger_obj.trigger_input_archive:
                            print('No triggering inputs, using fitness A!!')
                            fitness_value = FitnessValue((img_EI_neurons_ats_value)[0])
                        else:
                            for trigger_inp_ats in trigger_obj.trigger_inputs_ats_archive:
                                dist = np.linalg.norm(trigger_inp_ats - new_test_ats, ord=1, axis=1)  # 二范数
                                triggering_dis.append(dist)

                            triggering_outputs = np.min(triggering_dis)

                        # 2nd fitness value
                        # get_fault_pattern(new_test_ats, final_EI_neuron_idx)
                        # triggering_dis = []
                        # if not trigger_obj.trigger_input_archive:
                        #     print('No triggering inputs, using fitness A!!')
                        #     fitness_value = FitnessValue((img_EI_neurons_ats_value)[0])
                        # else:
                        #     for trigger_inp_ats in trigger_obj.trigger_inputs_ats_archive:
                        #         trigger_EI_neurons_ats = trigger_inp_ats[:, final_EI_neuron_idx]
                        #         dist = np.linalg.norm(trigger_EI_neurons_ats - img_EI_neurons_ats, ord=1, axis=1)  # 二范数
                        #         triggering_dis.append(dist)
                        #
                        #     triggering_outputs = np.min(triggering_dis)

                            # fitness function
                            if fitness_mode == 'fitC':
                                fitness_value = FitnessValue((img_EI_neurons_ats_value + triggering_outputs)[0])
                            elif fitness_mode == 'fitA':
                                fitness_value = FitnessValue(img_EI_neurons_ats_value[0])
                            else:
                                fitness_value = FitnessValue(triggering_outputs)

                        if fitness_value.better_than(best_fitness_value):
                            print("update fitness value from {} to {}".format(best_fitness_value,
                                                                              fitness_value.diff_value))
                            best_fitness_value = fitness_value
                            m.delta_bigger_than_zero += 1

                        last_mutation_operator = m
                        latest_img = np.squeeze(np.copy(new_img))  # 若新生成的img的fitness值高，则下次迭代以该new_img为基础进行迭代

                    if len(trigger_inp_seed) == 1 and trigger1st == False:
                        trigger1st = True
                        find_trigger_inp_et = time.time()
                        trigger_obj.average_time_archive[idx] = float(find_trigger_inp_et - find_trigger_inp_st)
                        print('finding the 1st triggering input: {}, iteration: {}'.format(
                            trigger_obj.average_time_archive[idx], iteration))

                trigger_obj.all_trigger_inps_archive[idx] = len(trigger_inp_seed)

                trigger_obj.fault_type_archive = list(set(trigger_obj.fault_type_archive))

            summary_str = summary_attack_results(seed_strategy, repeat, trigger_obj)
            # ALL_results[str(repeat)+seed_strategy].append(summary_str)

            # 生成的inputs存储在文件中
            trigger_obj.all_gen_inps  = {str(key): value for key, value in trigger_obj.all_gen_inps.items()}

            np.savez(saved_file_path + 'MOON-Original_{}_strategy_{}_V2.npz'.format(seed_strategy, str(repeat)),
                     **trigger_obj.all_gen_inps)

    #         np.savez(saved_file_path + 'MOON_all_gen_inps_Original_{}_strategy_{}_repeat_V1.npz'.format(seed_strategy, str(repeat)),
    #                  **trigger_obj.all_gen_inps)

    #         # np.save(saved_file_path + 'MOON_all_gen_triggering_inps_RandomMutation_{}_strategy_{}_repeat_V2.npy'.format(seed_strategy, str(repeat)),
    #         #         trigger_obj.trigger_input_archive)
    #
    # write2excel(ALL_results, repeat_times, seed_selection_strategy, saved_file_path)




