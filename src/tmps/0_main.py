# -*- coding: utf-8 -*-

"""
entry point

created on 20180601

"""
import os
import random
import time
from collections import Counter
import numpy as np

import torch

from autoencoder import AutoEncoder, load_data_from_file, change_labels, save_data, select_features_from_file, \
    save_data_in_autoencoder
from utils.data_loader import normalize_data, load_data_from_files
from wgan_model_sample import WGAN, save_tensor_data, merge_files, split_data


def achieve_reduced_features_data(input_file,epochs=1):

    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    st = time.time()
    print('It starts at ', start_time)
    # ----be careful with the ' ' in items
    # all_features= "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, Flow Duration," \
    #                " Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets," \
    #                " Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std," \
    #                "Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s," \
    #                " Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean," \
    #                " Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min," \
    #                "Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length," \
    #                "Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std," \
    #                " Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count," \
    #                " URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size," \
    #                " Avg Bwd Segment Size, Fwd Header Length,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate," \
    #                " Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes," \
    #                " Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd," \
    #                " min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min," \
    #                " Label"
    selected_features = " Source Port, Destination Port, Protocol, Flow Duration," \
                        " Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets," \
                        " Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std," \
                        "Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s," \
                        " Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean," \
                        " Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min," \
                        "Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length," \
                        "Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std," \
                        " Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count," \
                        " URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size," \
                        " Avg Bwd Segment Size, Fwd Header Length,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate," \
                        " Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes," \
                        " Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd," \
                        " min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min"

    # input_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX_demo.csv'
    output_file = '../original_data_no_sample/features_selected_Wednesday-workingHours.pcap_ISCX.csv'
    invalid_file = '../original_data_no_sample/invalid_data_Wednesday-workingHours.pcap_ISCX.csv'
    selected_features_list = selected_features.split(',')
    _, _, output_file= select_features_from_file(input_file, selected_features_list, output_file, invalid_file)
    X, Y = load_data_from_file(output_file)
    new_X = normalize_data(X, axis=0, low=-1, high=1, eps=1e-5)
    new_Y = change_labels(Y, labels=[1, 0])  # 'BENIGN=1, others=0'
    output_file = '../original_data_no_sample/features_selected_Normalized_Wednesday-workingHours.pcap_ISCX.csv'
    save_data_in_autoencoder(new_X, new_Y, output_file)

    model = AutoEncoder(new_X, new_Y, epochs)
    # 1. train model
    model.train()
    # torch.save(model.state_dict(), './sim_autoencoder.pth')

    # 2. encoding data and save the encoding data
    reduced_output_file = '../original_data_no_sample/features_selected_Normalized_Reduced_data_Wednesday-workingHours.pcap_ISCX.csv'
    reduced_features_data = model.encoder(torch.Tensor(new_X))
    reduced_features_data = normalize_data(reduced_features_data.tolist(), axis=0, low=0, high=1, eps=1e-5)
    save_data_in_autoencoder(reduced_features_data, new_Y, reduced_output_file)

    end_time = time.strftime('%Y-%h-%d %H:%M:%S', time.localtime())
    print('It ends at ', end_time)
    print('All takes %.4f s', time.time() - st)

    return reduced_output_file


def load_data_from_files_tmp(input_selected_features_file,sample_ratio=0.1, preprocess=True):

    benigin_data = []
    attack_data = []
    invalid_data = []
    X=[]
    Y=[]
    with open(input_selected_features_file, 'r') as in_file:
        line = in_file.readline()
        while line:
            line_arr = line.split(',')
            if line.startswith('Fl'):
                print(line)
                line = in_file.readline()
                continue

            if line_arr[-1] == '1\n':
                benigin_data.append(line_arr[:-1])
            else:
                attack_data.append(line_arr[:-1])
                # print('unknow label:',line,end='')

            line = in_file.readline()

    print('original_label: 1: %d, 0: %d'%(len(benigin_data), len(attack_data)))
    sample_size= int(len(benigin_data)*sample_ratio)
    print('sample size = ',sample_size,', int(len(benigin_data)*sample_ratio)')
    benigin_data=[ benigin_data[i] for i in sorted(random.sample(range(len(benigin_data)), sample_size)) ]
    attack_data=[ attack_data[i] for i in sorted(random.sample(range(len(attack_data)), sample_size)) ]

    if len(benigin_data) > len(attack_data):
        benigin_data=benigin_data[:len(attack_data)]
    else:
        attack_data=attack_data[:len(benigin_data)]

    X=benigin_data+attack_data
    for i in range(len(X)):
        X[i]= list(map(lambda x:float(x), X[i]))
        if i < len(benigin_data):
            Y.append(1)   # begin_data == 1
        else:
            Y.append(0)   # attack == 0

    return X, Y


def achieved_wgan_generated_data(input_file,sample_ratio=1,training_set_percent=0.9,critic=1,distance_threshold=0.05):

    X, Y = load_data_from_files_tmp(input_file, sample_ratio, preprocess=True)
    # print('X.shape:', X.shape, ' Y.shape:', np.asarray(Y).shape)
    print('X[0]:', X[0])
    X = normalize_data(X, axis=0, low=-1, high=1, eps=1e-5)
    print('Normalized X[0]:', X[0])
    print('X.shape:', X.shape, ' Y.shape:', np.asarray(Y).shape)
    print('label:', Counter(Y))

    features_selected = [str(i) for i in range(X.shape[1])]  # re-define new features name, E.g. '0', '1', ...

    show_flg = True
    save_flg = True
    in_size = X.shape[1]
    h_size = 12
    out_size = 1
    dtype = torch.float
    percent = training_set_percent

    mini_batch_size = 50
    epochs = 0  # invalid for my function, can't be str

    root_dir = './original_data_wgan_data_' + '%.2f' % percent + 'percent_' + time.strftime("%Y%m%d-%H:%M:%S",
                                                                                            time.localtime())
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    training_set, testing_set = split_data(X, Y, percent)
    print('training percent', percent, ':training_set (Benigin and Attack)', training_set[0].shape,
          ', testing_set (Benigin and Attack)', testing_set[0].shape)

    if save_flg:
        test_percent = '%.2f' % (1 - percent)
        save_tensor_data(training_set, output_file=os.path.join(root_dir, str(percent) + '_training_set.csv'))
        save_tensor_data(testing_set, output_file=os.path.join(root_dir, str(test_percent) + '_testing_set.csv'))

        training_file_lst = [os.path.join(root_dir, str(percent) + '_origin_training_set.arff'),
                             os.path.join(root_dir, str(percent) + '_training_set.csv')]
        merge_files(training_file_lst, header='header.txt', feature_lst=features_selected)  # add_arff_header
        testing_file_lst = [os.path.join(root_dir, str(test_percent) + '_origin_testing_set.arff'),
                            os.path.join(root_dir, str(test_percent) + '_testing_set.csv')]
        merge_files(testing_file_lst, header='header.txt', feature_lst=features_selected)  # add_arff_header

    nn_size_lst = [in_size, h_size, out_size]

    for i in range(2):
        if i == 0:
            data_flg = 'benign_data'
        else:
            data_flg = 'attack_data'
        wgan = WGAN((training_set[0], training_set[1]), nn_size_lst, mini_batch_size, epochs, show_flg, data_flg,critic,distance_threshold)

        # wgan.train()
        wgan.train_gp()

        input = torch.randn((1000, in_size // 10))
        gen_data = wgan.G(input)

        output_file = os.path.join(root_dir, str(percent) + '_gen_' + data_flg + '.csv')
        # print('data_flg ',data_flg)
        if data_flg == 'benign_data':
            data_type = '1'
            save_data(gen_data, data_type, output_file)
        else:
            data_type = '0'
            save_data(gen_data, data_type, output_file)

        ## merge files
    all_file_lst = [os.path.join(root_dir, str(percent) + '_all_in_one_file.csv.arff'),  # all_in_one_arff_file
                    os.path.join(root_dir, str(percent) + '_training_set.csv'),      # original_training_set_file
                    os.path.join(root_dir, str(percent) + '_gen_benign_data.csv'),        # gen_benign_data_file
                    os.path.join(root_dir, str(percent) + '_gen_attack_data.csv')]        # gen_attack_data_file
    merge_files(all_file_lst, header='header.txt', feature_lst=features_selected)

    return all_file_lst


if __name__ == '__main__':
    torch.manual_seed(1)

    demo_flg=0
    if demo_flg:
        input_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX_demo.csv'
        autoencoder_epochs=1
        sample_ratio=1
        training_set_percent=0.8
        critic=2
        distance_threshold=0.05
        # reduced_features_data_file = achieve_reduced_features_data(input_file, epochs=1)
        # print(reduced_features_data_file)
        #
        # all_file_lst = achieved_wgan_generated_data(reduced_features_data_file, sample_ratio=1,
        #                                             training_set_percent=0.8, critic=2,distance_threshold)
        # for file_tmp in all_file_lst:
        #     print(file_tmp)
    else:
        input_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX.csv'
        autoencoder_epochs = 1
        sample_ratio = 0.1
        training_set_percent = 0.1
        critic = 2
        distance_threshold = 0.04

    reduced_features_data_file=achieve_reduced_features_data(input_file,autoencoder_epochs)
    print(reduced_features_data_file)

    all_file_lst=achieved_wgan_generated_data(reduced_features_data_file,sample_ratio,training_set_percent,critic,distance_threshold)
    for file_tmp in all_file_lst:
        print(file_tmp)



