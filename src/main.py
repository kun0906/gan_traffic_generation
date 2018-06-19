# -*- coding: utf-8 -*-
"""
 @ main : entry point

 created on 20180615
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

__author__ = 'Learn-Live'

import logging as lg
import os
import sys
import time
import numpy as np

import torch

from utilties.data_preprocess import load_data_from_file, select_features, normalize_data, change_label, \
    handle_abnormal_values, divide_training_testing_data, save_data, merge_data_into_one_file
from wgan_gp_class import WGAN_GP

lg_level = lg.DEBUG
#
# file_subfix=time.strftime('%Y-%H-%d %h:%m:%s', time.localtime())
# logging.basicConfig(filename=file_subfix+'.log',level=lg_level)
# # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

file_subfix = '../.logs/' + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
file_handler = lg.FileHandler(filename=file_subfix + '.log')
stdout_handler = lg.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

lg.basicConfig(
    level=lg_level,
    # format=ColoredFormatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s -> %(message)s',reset=True),
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s -> %(message)s',
    # format='%(message)s <-[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s',
    handlers=handlers
)


def preprocess_data(X, Y):
    # step 1. handle abnormal
    new_X, new_Y = handle_abnormal_values(X, Y, abnormal_list=['-', 'NaN', 'Infinity'],
                                          except_columns=[0])  # remove abnormal valuse, such as '-', 'Nan'
    #
    # # step 2. sample data
    # new_X, new_Y = sample_data(new_X, new_Y, sample_rate=0.1)

    # step 3. selcet features
    new_X = select_features(new_X, selected_features_index_list=[10, 9])
    lg.debug(new_X)

    # step 4. normalize data
    new_X = normalize_data(new_X)
    # lg.debug(new_X)

    # step 5. change label
    new_Y = change_label(new_Y, label_dict={'BENIGN': 1, 'Others': 0})
    lg.debug(new_Y)

    return new_X, new_Y


def build_generate_model(training_set, data_flg='BENIGN', distance_threshold=0.5):
    X = torch.Tensor(training_set['X'])
    Y = torch.Tensor(training_set['Y'])
    in_size = X.shape[1]
    h_size = 12
    out_size = 1
    mini_batch_size = 10
    epochs = 0  # invalid for my function, can't be str
    nn_size_lst = [in_size, h_size, out_size, g_input_size]
    critic = 3  # D update times per G
    # distance_threshold = 0.1
    show_flg = True

    # step 1. initialization
    wgan = WGAN_GP((X, Y), nn_size_lst, mini_batch_size, epochs, show_flg, data_flg,
                   critic, distance_threshold)

    # step 2. train model
    wgan.train_gp()

    return wgan


def build_model(training_set):
    X = training_set['X']
    Y = training_set['Y']
    # model = LogisticRegression()
    model = RandomForestClassifier(max_depth=2, random_state=0)
    score = cross_val_score(model, X, Y, cv=10)
    model.fit(X, Y)

    return model


def evaluate_model(model, testing_set):
    X = testing_set['X']
    Y_true = testing_set['Y']
    Y_preds = model.predict(X)

    acc = accuracy_score(Y_true, Y_preds)
    cm = confusion_matrix(Y_true, Y_preds)

    return cm, acc


if __name__ == '__main__':
    lg.debug('begin')
    input_file = '../data/Wednesday-workingHours.pcap_ISCX_demo.csv'
    results_data_dir = '../results_data'  # the results data stores directory
    if not os.path.exists(results_data_dir):
        os.mkdir(results_data_dir)

    # step 1. load data
    X, Y = load_data_from_file(input_file)
    lg.debug('%s,%s' % (X[0][0], Y[0]))
    new_X, new_Y = preprocess_data(X, Y)

    # step 1.2. get training set and testing set
    training_set_percent = 0.3
    training_testing_data_out_dir = os.path.join(results_data_dir, 'training_testing_data')
    if not os.path.exists(training_testing_data_out_dir):
        os.mkdir(training_testing_data_out_dir)

    training_set, testing_set = divide_training_testing_data(new_X, new_Y, training_set_percent, repeatable=False)
    training_set_file = os.path.join(training_testing_data_out_dir, str(training_set_percent) + '_training_set.txt')
    save_data(training_set['X'], training_set['Y'], output_file=training_set_file)
    save_data(testing_set['X'], testing_set['Y'], output_file=os.path.join(training_testing_data_out_dir, str(
        1 - training_set_percent) + '_testing_set.txt'))

    # step 2. build generated model
    # wgan_gp_model=build_model(training_set)
    generated_data_num = 1000
    g_input_size = 3
    generated_results_out_dir = os.path.join(results_data_dir, str(generated_data_num) + '_generated_samples')
    if not os.path.exists(generated_results_out_dir):
        os.mkdir(generated_results_out_dir)

    training_set_with_generated_data_file = os.path.join(results_data_dir, str(
        training_set_percent) + '_training_set_with_generated_data.txt')
    generated_data = {'BENIGN': [], 'ATTACK': []}
    for i in range(2):
        if i == 0:
            data_flg = 'BENIGN'
            label = 1
        else:
            data_flg = 'ATTACK'
            label = 0
        generated_data_file = os.path.join(generated_results_out_dir, data_flg + '.txt')
        wgan_gp_model = build_generate_model(training_set, data_flg, distance_threshold=0.9)

        # step 2.1 generated data
        noise_data = torch.randn((generated_data_num, g_input_size))
        generated_X = wgan_gp_model.G(noise_data)
        # step 2.2 save data
        generated_Y = [label for j in range(generated_data_num)]
        save_data(generated_X.tolist(), generated_Y, output_file=generated_data_file)
        generated_data[data_flg].append([generated_X.tolist(), generated_Y])

    # generated_data_file = os.path.join(generated_results_out_dir, 'generated_data.txt')
    # if os.path.exists(generated_data_file):
    #     os.remove(generated_data_file)
    # save_data()
    # merge_data_into_one_file(training_set_file, generated_data_file, output_file=training_set_with_generated_data_file)
    # X_with_generated_data,Y_with_generated_data = load_data_from_file(training_set_with_generated_data_file)
    # X_with_generated_data=np.asarray(X_with_generated_data,dtype='float')
    # Y_with_generated_data=np.asarray(Y_with_generated_data, dtype='int')
    # #  normalize data
    # new_X_with_generated_data = normalize_data(X_with_generated_data)
    # # lg.debug(new_X)
    new_X_with_generated_data = training_set['X'] + list(map(lambda x: x[0], generated_data['BENIGN']))[0] + \
                                list(map(lambda x: x[0], generated_data['ATTACK']))[0]
    Y_with_generated_data = training_set['Y'] + list(map(lambda x: x[1], generated_data['BENIGN']))[0] + \
                            list(map(lambda x: x[1], generated_data['ATTACK']))[0]
    training_set_with_generated_data = {'X': new_X_with_generated_data, 'Y': Y_with_generated_data}
    # step 2.1 build classifier
    classifier_model = build_model(training_set)
    classifier_model_with_generated_data = build_model(training_set_with_generated_data)

    # step 3. evaluate model
    # step 3.1 evaluate model on training set
    lg.info('---evaluate model on training set')
    cm, acc = evaluate_model(classifier_model, training_set)
    lg.info('Confusion Matrix:%s' % cm)
    lg.info('Accuracy:%.4f' % acc)
    cm, acc = evaluate_model(classifier_model_with_generated_data, training_set_with_generated_data)
    lg.info('Confusion Matrix:%s' % cm)
    lg.info('Accuracy:%.4f\n' % acc)

    # step 3.2 evaluate model on testing set
    lg.info('***evaluate model on testing set')
    cm, acc = evaluate_model(classifier_model, testing_set)
    lg.info('Confusion Matrix:%s' % cm)
    lg.info('Accuracy:%.4f' % acc)
    cm, acc = evaluate_model(classifier_model_with_generated_data, testing_set)
    lg.info('Confusion Matrix:%s' % cm)
    lg.info('Accuracy:%.4f\n' % acc)
