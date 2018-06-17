# -*- coding: utf-8 -*-
"""
 Purpose:
    do data preprocessing, such as load data, normalization

"""

__author__ = 'Learn-Live'

import os
import logging
import random

import numpy as np

lg = logging.getLogger(__name__)


def load_data_from_file(input_file, *args, **kwargs):
    lg.debug(input_file)
    if not os.path.exists(input_file):
        lg.error('%s does not exist.' % input_file)
        exit(-1)

    X = []
    Y = []
    with open(input_file, 'r') as f_in:
        line = f_in.readline()

        while line:
            if line and line.strip()[0].isalpha():  # match any alpha character [a-zA-Z]
                lg.info('line:%s' % line.split())
                line = f_in.readline()
                continue
            line_arr = line.split(',')
            X.append(line_arr[:-1])
            Y.append(line_arr[-1])

            line = f_in.readline()

    return X, Y


def remove_features(X, remove_features_index_list=[0, 1], *args, **kwargs):
    new_X = []
    for row in range(len(X)):
        tmp_arr = []
        for column in range(len(remove_features_index_list)):  # save the column
            for j in range(len(X[0])):
                if j == remove_features_index_list[column]:
                    continue
                else:
                    tmp_arr.append(float(X[row][j]))
        new_X.append(tmp_arr)

    return new_X


def select_features(X, selected_features_index_list=[3, 1], *args, **kwargs):
    new_X = []
    for row in range(len(X)):
        tmp_arr = []
        for column in range(len(selected_features_index_list)):  # save the column
            tmp_arr.append(float(X[row][selected_features_index_list[column]]))
        new_X.append(tmp_arr)

    return new_X


def handle_abnormal_values(X, Y, abnormal_list=['-', 'NaN', 'Infinity'], except_columns=[0]):
    new_X = []
    new_Y = []
    for i in range(len(X)):
        abnormal_flg = False
        for j in range(len(X[0])):
            if j in except_columns:  # do not check this columns
                continue
            for t in abnormal_list:  # ignore negative values
                if t in X[i][j]:
                    abnormal_flg = True
                    break  # break for t
            if abnormal_flg:
                break  # break for j
        if abnormal_flg:
            continue
        new_X.append(X[i])
        new_Y.append(Y[i])

    return new_X, new_Y


def sample_data(X, Y, sample_rate=0.1):
    pass


def normalize_data(X, range_value=[-1, 1], eps=1e-5):  # down=-1, up=1

    new_X = np.copy(X)

    mins = new_X.min(axis=0)  # column
    maxs = new_X.max(axis=0)

    rng = maxs - mins
    for i in range(rng.shape[0]):
        if rng[i] == 0.0:
            rng[i] += eps

    new_X = (new_X - mins) / rng * (range_value[1] - range_value[0]) + range_value[0]

    return new_X


def change_label(Y, label_dict={'BENIGN': 1, 'Others': 0}):
    new_Y = []

    for i in range(len(Y)):
        if Y[i].strip() == 'BENIGN':
            new_Y.append(label_dict['BENIGN'])
        else:
            new_Y.append(label_dict['Others'])

    return new_Y


def divide_training_testing_data(X, Y, training_set_percent=0.3, repeatable=False):
    new_X = np.copy(X).tolist()
    new_Y = np.copy(Y).tolist()

    training_set = {'X': [], 'Y': []}
    testing_set = {'X': [], 'Y': []}
    sample_size = int(len(X) * training_set_percent)
    if repeatable:
        for i in range(sample_size):
            index = random.randint(0, len(new_X))
            training_set['X'].append(new_X[index])
            training_set['Y'].append(new_Y[index])
        for j in range(len(X) - sample_size):
            index = random.randint(0, len(new_X))
            testing_set['X'].append(new_X[index])
            testing_set['Y'].append(new_Y[index])
    else:
        for i in range(sample_size):
            index = random.randint(0, len(new_X))
            print(index, len(new_X))
            training_set['X'].append(new_X[index])
            training_set['Y'].append(new_Y[index])
            new_X.pop(index)
            new_Y.pop(index)
        testing_set['X'] = new_X
        testing_set['Y'] = new_Y

    print('training percent', training_set_percent, ':training_set (Benigin and Attack)',
          (len(training_set['X']), len(training_set['X'][0]),
           ', testing_set (Benigin and Attack)', (len(testing_set['X'])), len(testing_set['Y'])))

    return training_set, testing_set


def save_data(X, Y, output_file='results.txt'):
    with open(output_file, 'w') as f_out:
        for i in range(len(X)):
            line_tmp = ''
            for j in range(len(X[0])):
                line_tmp += str(X[i][j]) + ','
            line_tmp += str(Y[i])
            f_out.write(line_tmp + '\n')

    return output_file


if __name__ == '__main__':
    input_file = '../../data/Wednesday-workingHours.pcap_ISCX_demo.csv'
    X, Y = load_data_from_file(input_file)
    new_X = select_features(X, selected_features_index_list=[3, 1])
    lg.debug(new_X)
    new_X = normalize_data(new_X)
    lg.debug(new_X)
