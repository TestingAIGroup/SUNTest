import numpy as np
import math





def tarantula_analysis(num_ac, num_uc, num_af, num_uf):

    result = []
    i = 0
    for j in range(len(num_af[0])):
        result.append(
            float(float(num_af[i][j]) / (num_af[i][j] + num_uf[i][j])) / \
        (float(num_af[i][j]) / (num_af[i][j] + num_uf[i][j]) + float(num_ac[i][j]) / (num_ac[i][j] + num_uc[i][j]))
        )

    return result


def ochiai_analysis( num_ac, num_uc, num_af, num_uf):

    result = []
    i = 0
    for j in range(len(num_af[0])):
        result.append(
            float(num_af[i][j]) / ((num_af[i][j] + num_uf[i][j]) * (num_af[i][j] + num_ac[i][j])) ** (.5)
        )

    return result

def dstar_analysis(num_ac, num_uc, num_af, num_uf):

    star = 3
    result = []
    i = 0
    for j in range(len(num_af[0])):
        result.append(
            float(num_af[i][j] ** star) / (num_ac[i][j] + num_uf[i][j])
        )

    return result

def other_analysis(num_ac, num_uc, num_af, num_uf):

    result = []
    i = 0
    for j in range(len(num_af[0])):
        result.append(
            float(num_af[i][j] + num_uc[i][j]) / (num_ac[i][j] + num_uf[i][j])
        )

    return result

def other_star_analysis(num_ac, num_uc, num_af, num_uf):

    star = 3
    result = []
    i = 0
    for j in range(len(num_af[0])):
        result.append(
            float(num_af[i][j]**star + num_uc[i][j]**star) / (num_ac[i][j] + num_uf[i][j])
        )

    return result


def scores_with_foo(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, foo):

    for i in range(len(scores)):
        for j in range(len(scores[i])):
            score = foo(i, j)
            if np.isnan(score):
                score = 0
            scores[i][j] = score

    flat_scores = [float(item) for sublist in scores for item in sublist if not math.isnan(float(item))]

    # grab the indexes of the highest suspicious_num scores
    if suspicious_num >= len(flat_scores):
        flat_indexes = range(len(flat_scores))
    else:
        flat_indexes = np.argpartition(flat_scores, -suspicious_num)[-suspicious_num:]

    suspicious_neuron_idx = []
    for idx in flat_indexes:
        # unflatten idx
        i = 0
        accum = idx
        while accum >= len(scores[i]):
            accum -= len(scores[i])
            i += 1
        j = accum

        if trainable_layers is None:
            suspicious_neuron_idx.append((i, j))
        else:
            suspicious_neuron_idx.append((trainable_layers[i], j))

    return suspicious_neuron_idx