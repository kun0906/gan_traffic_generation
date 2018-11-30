# -*- coding: utf-8 -*-
"""
    load data from csv file
"""
import os
from collections import Counter

import numpy as np


def csv_dataloader(input_file):
    """

    :param input_file:
    :return:
    """
    X = []
    y = []
    with open(input_file, 'r') as f_in:
        line = f_in.readline()
        while line:
            if line.startswith('Flow'):
                line = f_in.readline()
                continue
            line_arr = line.split(',')
            X.append(line_arr[:-1])
            # X.append(line_arr[7:40])
            # if line_arr[-1] == '2\n':
            #     y.append('1')
            # else:
            #     y.append('0')
            y.append(line_arr[-1].strip())
            line = f_in.readline()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    print('input_data size is ', Counter(y))

    return (X, y)


def open_file(input_file, label='0', start_feat_idx=[5,9]):
    """

    :param input_file:
    :param label:
    :param start_feat_idx: start feature's index
    :return:
    """
    print('input_file:', input_file)
    X = []
    y = []
    with open(input_file, 'r') as file_in:
        line = file_in.readline()
        while line:
            if line.strip().startswith('ts'):
                print(line.strip())
                line = file_in.readline()
                continue

            line_arr = line.strip().split(',')
            line_arr = [f.strip() for f in line_arr]  # delete ' ' (space) in each features
            if start_feat_idx[0] =='-':
                start_feat_idx[0] =0
            if start_feat_idx[1] =='-':
                X.append(line_arr[start_feat_idx[0]:])
            else:
                X.append(line_arr[start_feat_idx[0]:start_feat_idx[1]])
            y.append(label)
            line = file_in.readline()

    return X, y


def mix_normal_attack_and_label(normal_f='', attack_f='', label_dict={'normal': '0', 'attack': '1'}, start_feat_idx=['-','-'],
                                output_f='./mix_data.csv'):
    """

    :param normal_f:
    :param attack_f:
    :param label_dict:
    :param start_feat_idx:
    :param output_f:
    :return:
    """
    assert os.path.exists(normal_f)
    assert os.path.exists(attack_f)
    print('start features index is ', start_feat_idx)
    X = []
    y = []
    X_normal, y_normal = open_file(normal_f, label=label_dict['normal'], start_feat_idx=start_feat_idx)
    print('X_normal[0]: features=',len(X_normal[0]), ',',X_normal[0])
    X.extend(X_normal)
    y.extend(y_normal)

    X_attack, y_attack = open_file(attack_f, label=label_dict['attack'], start_feat_idx=start_feat_idx)
    print('X_attack[0]: features=',len(X_attack[0]),',', X_attack[0])
    X.extend(X_attack)
    y.extend(y_attack)

    with open(output_f, 'w') as out_f:
        for i in range(len(y)):
            line = ','.join(X[i]) + ',' + y[i] + '\n'
            out_f.write(line)

        out_f.flush()

    return (X, y), output_f


def save_data(data, output_f=''):
    with open(output_f, 'w') as out_f:
        for line in data:
            line_str = ','.join([str(v) for v in line]) + '\n'
            out_f.write(line_str)

    return output_f


def save_numpy_data(data, output_f=''):
    with open(output_f, 'w') as out_f:
        for x, y in zip(data[0], data[1]):
            line_str = ','.join([str(v) for v in x]) + ',' + str(int(y)) + '\n'
            out_f.write(line_str)

    return output_f
