#!/usr/bin/env python
# Based on https://www.kaggle.com/alphasis/xgboost-with-context-label-data-acc-99-637
# Make sure you have dataset/dataset1 and dataset/dataset2 in the right place in the folder above where this file is

from __future__ import print_function
import sys
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..",)))
from get_time_series import get_file


#FREQUENCY_UNIT_COUNT = 4432
FREQUENCY_UNIT_COUNT = 1000

dataset_path1 = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset1"))
dataset_path4 = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset4"))

out_path = os.path.realpath(os.path.join(os.path.dirname(__file__)))

labels = ["up", "down", "silence"]


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def load_file(file_path, class_path):
    file_path = os.path.join(class_path, file_path)
    # Load wav files
    x_data_from_file =  get_file(file_path)

    # Get the class from folder name

    class_name = os.path.basename(class_path)
    if class_name.startswith("all_"):
        class_name = class_name[len("all_"):]
    return x_data_from_file, labels.index(class_name)


def load_sound_file_for_network(file_path):
    """
    Loads a file for the network, should match what we are doing in the function load_dataset and load_file
    :param file_path:
    :return:
    """
    # Load file, just give some label
    data = load_file(file_path, labels[0])[0]
    x_data = np.zeros((1, FREQUENCY_UNIT_COUNT))
    # import code; code.interact(local=dict(globals(), **locals()))

    x_data[0] = np.array(data[1][:min(FREQUENCY_UNIT_COUNT, len(data[1]))])

    return_value =  np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    print(return_value.shape)
    return return_value


def load_dataset(path):
    outputs = []
    print("Loading dataset" + str(path))
    for class_path in os.listdir(path):
        class_path = os.path.join(path, class_path)
        if os.path.isdir(class_path):
            #for file_path in os.listdir(class_path):
            out = Parallel(n_jobs=4, backend="threading")(delayed(load_file)(file_path, class_path)
                                                              for file_path in os.listdir(class_path))
            outputs += out

    # Just arrange this stuff back
    x_data = np.zeros((len(outputs), FREQUENCY_UNIT_COUNT))
    y_data = np.zeros(len(outputs))

    for i, output in enumerate(outputs):
        # Dump wave data in to array, if its too long cut it in to WAV_LENGTH
        #import code; code.interact(local=dict(globals(), **locals()))
        x_data[i] = np.array(output[0][1][:min(FREQUENCY_UNIT_COUNT, len(output[0][1]))])
        y_data[i] = output[1]
    return x_data, y_data


if __name__ == "__main__":
    # Load dataset
    x_data, y_data = load_dataset(dataset_path1)
    x_data_valid, y_data_valid = load_dataset(dataset_path4)

    print('Total number of samples:', len(x_data))

    x_train = x_data
    y_train = y_data

    # Add all of dataset 4 as validation, sperate recording from dataset 1
    x_train = np.concatenate([x_train, x_data_valid])
    y_train = np.concatenate([y_train, y_data_valid])

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=0.1, random_state=2017)

    #x_valid = x_data_valid
    #y_valid = y_data_valid


    # Load dataset to network for training

    # Number of classes to train
    num_classes = len(labels)

    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_valid = keras.utils.to_categorical(y_valid, num_classes)

    print("current x train shape:" + str(x_train.shape))

    gc.collect()

    from sklearn.svm import LinearSVC

    y_train = y_train.astype(np.uint8)
    from sklearn.metrics import classification_report
    print(x_train.shape, y_train.shape)

    # import code; code.interact(local=dict(globals(), **locals()))

    #x_train = np.array(list(map(lambda x: movingaverage(x, 3), x_train)))
    # x_valid = np.array(list(map(lambda x: movingaverage(x, 3), x_valid)))

    print(x_train.shape)
    # import code;     code.interact(local=dict(globals(), **locals()))

    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(x_train, y_train)
    LinearSVC(C=1.0, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0, class_weight="balanced")
    print(clf.coef_)
    print(clf.intercept_)
    y_pred = clf.predict(x_valid)

    print(classification_report(y_valid, y_pred))

    import pickle

    with open("/tmp/svc.pkl",  'wb') as f:
        pickle.dump(clf, f)

    print("done")
