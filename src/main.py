# -*- coding: utf-8 -*-

"""
 @ main

 created on 20180615
"""

import logging as lg
import os
import random
import sys
import time
import torch

import numpy as np

from utilties.data_preprocess import load_data_from_file, select_features, normalize_data, change_label, \
    handle_abnormal_values, remove_features, sample_data, divide_training_testing_data, save_data
from colorlog import ColoredFormatter

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


def build_model(training_set, data_flg='BENIGN'):
    X=torch.Tensor(training_set['X'])
    Y=torch.Tensor(training_set['Y'])
    in_size = X.shape[1]
    h_size = 12
    out_size = 1
    mini_batch_size = 10
    epochs = 0  # invalid for my function, can't be str
    nn_size_lst = [in_size, h_size, out_size, g_input_size]
    critic = 3  # D update times per G
    distance_threshold = 0.05
    show_flg=True

    # step 1. initialization
    wgan = WGAN_GP((X,Y), nn_size_lst, mini_batch_size, epochs, show_flg, data_flg,
                   critic, distance_threshold)

    # step 2. train model
    wgan.train_gp()

    return wgan


def evaluate_model(testing_set):
    pass


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

    save_data(training_set['X'], training_set['Y'],
              output_file=os.path.join(training_testing_data_out_dir, str(training_set_percent) + '_training_set.txt'))
    save_data(testing_set['X'], testing_set['Y'], output_file=os.path.join(training_testing_data_out_dir, str(
        1 - training_set_percent) + '_testing_set.txt'))

    # step 2. build model
    # wgan_gp_model=build_model(training_set)
    generated_data_num = 1000
    g_input_size = 3
    generated_results_out_dir_tmp = os.path.join(results_data_dir, str(generated_data_num) + 'generated_samples')
    if not os.path.exists(generated_results_out_dir_tmp):
        os.mkdir(generated_results_out_dir_tmp)
    for i in range(2):
        if i == 0:
            data_flg = 'BENIGN'
            label = 1
        else:
            data_flg = 'ATTACK'
            label = 0
        output_file = os.path.join(generated_results_out_dir_tmp, data_flg + '.txt')

        wgan_gp_model = build_model(training_set, data_flg)

        # step 2.1 generated data
        noise_data = torch.randn((generated_data_num, g_input_size))
        generated_X = wgan_gp_model.G(noise_data)
        # step 2.2 save data
        generated_Y = [label for j in range(generated_data_num)]
        save_data(generated_X.tolist(), generated_Y, output_file)

    # step 3. evaluat model
    evaluate_model(testing_set)
