# -*- coding: utf-8 -*-
"""
    @ csv to arff : convert csv to arff.

    return:

csv format:
    f1, f2, ..., f60, 0, 0, 1, 0

    in which,
    features= f1, f2, ..., f60
    label = one-hot coding: 0,0,1,0

"""

from __future__ import print_function, division

import os



def append_data_to_file(all_in_one_file, new_file):
    """

    :param all_in_one_file:
    :param new_file:
    :return:
    """
    with open(all_in_one_file, 'a') as fid_out:
        with open(new_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                line_arr = line.split('|')
                # print(line_arr[-4], ','.join([str(v) for v in line_arr[-4]]))
                # line_tmp = first_n_pkts_list+flow_duration+interval_time_list+label
                # print(IP2Int(line_arr[2]), line_arr[2])
                ### srcport, dstport, len(pkts), pkts_lst, flow_duration, intr_time_lst, label
                line_tmp = str(IP2Int(line_arr[2])) + ',' + str(IP2Int(line_arr[3])) + ',' + str(
                    line_arr[-6]) + ',' + str(line_arr[-5]) + ',' + str(len(eval(line_arr[-4]))) + ',' + ','.join(
                    [str(v) for v in eval(line_arr[-4])]) + ',' + line_arr[-3] + ',' + ','.join(
                    [str(v) for v in eval(line_arr[-2])]) + ',' + line_arr[
                               -1]  # line_arr[-4]='[1140,1470]', so use eval() to change str to list
                # print(line_tmp)
                fid_out.write(line_tmp)
                line = fid_in.readline()


def append_data_to_file_with_mean(all_in_one_file, new_file):
    """

    :param all_in_one_file:
    :param new_file:
    :return:
    """
    with open(all_in_one_file, 'a') as fid_out:
        with open(new_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                line_arr = line.split('|')
                # print(line_arr[-4], ','.join([str(v) for v in line_arr[-4]]))
                # line_tmp = first_n_pkts_list+flow_duration+interval_time_list+label
                # line_tmp = ','.join([str(v) for v in eval(line_arr[-4])]) + ',' + line_arr[-3]+',' + ','.join(
                #     [str(v) for v in eval(line_arr[-2])]) + ',' + line_arr[-1]   # line_arr[-4]='[1140,1470]', so use eval() to change str to list
                # # print(line_tmp)
                pkts_mean = compute_mean(eval(line_arr[-4]))
                flow_dur = float(line_arr[-3])
                intr_tm_mean = compute_mean(eval(line_arr[-2])[1:])  # line_arr[first_n+1] always is 0
                line_tmp = str(IP2Int(line_arr[2])) + ',' + str(IP2Int(line_arr[3])) + ',' + str(
                    line_arr[-6]) + ',' + str(line_arr[-5]) + ',' + str(len(eval(line_arr[-4]))) + ',' + str(
                    pkts_mean) + ',' + str(flow_dur) + ',' + str(intr_tm_mean) + ',' + line_arr[-1]
                # line_tmp = str(pkts_mean) + ',' + str(flow_dur) + ',' + line_arr[-1]
                fid_out.write(line_tmp)
                line = fid_in.readline()


def add_arff_header(all_in_one_file, attributes_num=2, label=['a', 'b', 'c']):
    """

    :param all_in_one_file:
    :param attributes_num:
    :param label:
    :return:
    """
    output_file = os.path.splitext(all_in_one_file)[0] + '.arff'
    print(output_file)
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w') as fid_out:
        fid_out.write('@Relation test\n')
        for i in range(attributes_num):
            fid_out.write('@Attribute feature_%s numeric\n' % i)
        label_tmp = ','.join([str(v) for v in label])
        print('label_tmp:', label_tmp)
        fid_out.write('@Attribute class {%s}\n' % (label_tmp))
        fid_out.write('@data\n')
        with open(all_in_one_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                fid_out.write(line)
                line = fid_in.readline()


def merge_features_labels(features_file, labels_file):
    data_file = os.path.splitext(features_file)[0] + '_merged.csv'
    i = 0
    with open(data_file, 'w') as fid_out:
        # with open(features_file, 'r') as fid_in:
        #     with open(labels_file, 'r') as fid_in2:
        #         line = fid_in.readline() + ',' + str(int(float(fid_in2.readline().strip())))
        #         while line :
        #             i +=1
        #             fid_out.write(line)
        #             tmp_label=fid_in2.readline().strip()
        #             print('i=%d, tmp_label=%s'%(i,tmp_label))
        #             # line = fid_in.readline() + ',' + str(int(float(fid_in2.readline().strip())))
        #             line = fid_in.readline() +',' + str(int(float(tmp_label)))
        #             print(line)
        features_data = []
        with open(features_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                features_data.append(line.strip())
                line = fid_in.readline()

        with open(labels_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                lab = int(float(line.strip()))
                fid_out.write(features_data[i] + ',' + str(lab) + '\n')
                i += 1
                line = fid_in.readline()
    return data_file

def save_to_arff(input_file, output_file, features_num=60, labels=[0, 1, 2, 3]):
    """

    :param input_file:
    :param output_file:
    :param features_num:
    :return:
    """
    with open(output_file, 'w') as fid_out:
        fid_out.write('@Relation "demo"\n')
        for i in range(features_num):
            fid_out.write('@Attribute feature_%s numeric\n' % i)
        label_tmp = ','.join([str(v) for v in labels])
        # print('label_tmp:', label_tmp)
        fid_out.write('@Attribute class {%s}\n' % (label_tmp))
        fid_out.write('@data\n')

        with open(input_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                fid_out.write(line)
                line = fid_in.readline()


if __name__ == '__main__':

    root_dir = '../results/arff'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    first_n_pkts = 10

    data_root_dir = '../data/'
    tmp_dir = 'data_split_train_v2_711'
    csv_root_dir = os.path.join(data_root_dir, tmp_dir)
    if not os.path.exists(csv_root_dir):
        os.mkdir(csv_root_dir)
    arff_dir = os.path.join(root_dir, tmp_dir)
    if not os.path.exists(arff_dir):
        os.mkdir(arff_dir)

    # features_file = '../data/data_split_train_v2_711/train_1pkt_images.csv'
    # labels_file = '../data/data_split_train_v2_711/train_1pkt_labels.csv'
    # input_file=merge_features_labels(features_file, labels_file)
    # output_file = os.path.join(arff_dir, 'pkt1_merge.arff')
    # save_to_arff(input_file, output_file, features_num=60, labels=[0,1,2,3])

    # for i in range(1, first_n_pkts + 1,2):
    for i in [1, 3, 5, 8, 10]:
        features_file_i = '../data/data_split_train_v2_711/train_%dpkt_images.csv' % i
        labels_file_i = '../data/data_split_train_v2_711/train_%dpkt_labels.csv' % i
        input_file_i = merge_features_labels(features_file_i, labels_file_i)
        # input_file_i = os.path.join(csv_root_dir, 'pkt%d.csv' % i)
        print('input_file_%d=%s' % (i, input_file_i))
        output_file_i = os.path.join(arff_dir, 'pkt%d.arff' % i)
        save_to_arff(input_file_i, output_file_i, features_num=60 * i + (i - 1), labels=[0, 1, 2, 3])
