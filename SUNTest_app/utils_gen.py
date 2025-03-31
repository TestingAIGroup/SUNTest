import os
import numpy as np
from keras.models import Model, load_model
from multiprocessing import Pool

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

def get_layers_ats( model, X_train, layer_names, dataset):
    save_path =  '/workspace/EI_Neurons/cifar10_vgg/MOON/ats_gen/'
    saved_train_path = _get_saved_path(save_path, dataset, "test_seed", layer_names)

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

# get_layers_new_img_ats(model, new_img, layer_names, dataset)
def get_layers_new_img_ats( model, X_train, layer_names, dataset):

    saved_train_path = None
    train_ats, train_pred = get_ats(
            model,
            X_train,
            layer_names,
            saved_train_path,
            batch_size=128,
            is_classification=True,
            num_proc=10,
    )

    return train_ats, train_pred

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
