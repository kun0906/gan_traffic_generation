# -*- coding: utf-8 -*-
"""
    autoencoder:
            reduce features' dimensions

    ref:  https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
          (__author__ = 'SherlockLiao')

"""
import time

from utils.data_loader import normalize_data
from utils.show_save import save_data

__author__ = 'SherlockLiao'

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def print_net(net, describe_str='Net'):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


class AutoEncoder(nn.Module):
    def __init__(self, X, Y, epochs=10):
        super().__init__()
        self.dataset = Data.TensorDataset(torch.Tensor(X), torch.Tensor(Y))

        self.epochs = epochs
        self.learning_rate = 1e-3
        self.batch_size = 50

        self.show_flg = True

        self.num_features_in = len(X[0])
        self.h_size = 16
        self.num_features_out = 10

        self.encoder = nn.Sequential(
            nn.Linear(self.num_features_in, self.h_size * 8),
            nn.ReLU(True),
            nn.Linear(self.h_size * 8, self.h_size * 4),
            nn.ReLU(True),
            nn.Linear(self.h_size * 4, self.h_size * 2),
            nn.ReLU(True),
            nn.Linear(self.h_size * 2, self.num_features_out))

        self.decoder = nn.Sequential(
            nn.Linear(self.num_features_out, self.h_size * 2),
            nn.ReLU(True),
            nn.Linear(self.h_size * 2, self.h_size * 4),
            nn.ReLU(True),
            nn.Linear(self.h_size * 4, self.h_size * 8),
            nn.ReLU(True),
            nn.Linear(self.h_size * 8, self.num_features_in),
            nn.Tanh())

        if self.show_flg:
            print_net(self.encoder, describe_str='Encoder')
            print_net(self.decoder, describe_str='Decoder')

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train(self):

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.loss = []
        for epoch in range(self.epochs):
            for iter, (batch_X, batch_Y) in enumerate(dataloader):
                # img = img.view(img.size(0), -1)
                input_data_X = Variable(batch_X)
                # ===================forward=====================
                # output = model(input_data_X)
                output = self.forward(input_data_X)
                loss = self.criterion(output, input_data_X)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss.append(loss.data)

            # ===================log========================
            print('epoch [{:d}/{:d}], loss:{:.4f}'
                  .format(epoch + 1, self.epochs, loss.data))
            # if epoch % 10 == 0:
            #     # pic = to_img(output.cpu().data)
            #     # save_image(pic, './mlp_img/image_{}.png'.format(epoch))

        if self.show_flg:
            plt.figure()

            plt.plot(self.loss, 'r', alpha=0.5, label='loss')
            # plt.plot(G_loss, 'g', alpha=0.5, label='G_loss')
            plt.legend(loc='upper right')
            plt.show()


def select_features_from_file(input_file, selected_features_list=[], output_file='', invalid_file=''):
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(invalid_file):
        os.remove(invalid_file)

    X = []
    Y = []
    with open(output_file, 'w') as out_file:
        with open(invalid_file, 'w') as out_invalid_file:
            with open(input_file, 'r') as in_file:
                line = in_file.readline()
                all_data_count = 0
                while line:
                    line_arr = line.split(',')
                    if line.startswith('Flow'):
                        out_invalid_file.write(line)
                        index_arr = []
                        header_tmp = ''
                        for i in range(len(selected_features_list)):
                            index_arr.append(line_arr.index(selected_features_list[i]))
                            header_tmp += selected_features_list[i] + ','

                        out_file.write(header_tmp + ' Label\n')
                        line = in_file.readline()
                        continue

                    all_data_count += 1
                    line_arr[0] = str(
                        all_data_count)  # instead of the first column(192.168.10.14-209.48.71.168-49459) with int_value
                    line = ''
                    for t in range(len(line_arr)):
                        line += line_arr[t] + ','

                    if '-' in line or 'NaN' in line or 'Infinity' in line:  # ignore negative values
                        # if len(invalid_data) % 1000 == 0:
                        #     print(line_arr)
                        out_invalid_file.write(line)
                        line = in_file.readline()
                        continue

                    # only select the features in select_features_list
                    line_tmp = ''
                    for i in range(len(index_arr)):
                        line_tmp += line_arr[index_arr[i]] + ','
                    # if line_arr[-1] =='0':
                    #     line_arr[-1]=='1'
                    # elif line_arr[-1] =='1':
                    #     line_arr[-1] == '0'
                    # else:
                    #     print('unknow label:',line)
                    line_tmp += line_arr[-1]
                    out_file.write(line_tmp)
                    out_invalid_file.write(line_tmp)

                    X.append(line_arr)
                    Y.append(line_arr[-1])

                    line = in_file.readline()

    return X, Y, output_file


def load_data_from_file(input_file):
    X = []
    Y = []
    with open(input_file, 'r') as in_file:
        line = in_file.readline()
        while line:
            line_arr = line.split(',')
            if line.startswith(' Sou'):
                print('Num. of features is ', len(line.split(',')), ', header:', line)
                line = in_file.readline()
                continue

            X.append(line_arr[:-1])
            Y.append(line_arr[-1])

            line = in_file.readline()

    return X, Y


def change_labels(Y, labels=[1, 0]):
    new_Y = []
    for i in range(len(Y)):
        if Y[i] == 'BENIGN' or Y[i] == 'BENIGN\n':
            new_label = labels[0]
        else:
            new_label = labels[1]
        new_Y.append(new_label)

    return new_Y


def save_data_in_autoencoder(X, Y, output_file='./gen_data.csv'):
    # print(X)
    # print('Y:',Y)
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as out_file:
        for i in range(X.shape[0]):
            line_str = ''
            for j in range(len(X[i].tolist())):
                line_str += str(X[i].tolist()[j]) + ','
            out_file.write(line_str + str(Y[i]) + '\n')
            out_file.flush()


if __name__ == '__main__':
    torch.manual_seed(1)
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

    input_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX.csv'
    output_file = '../original_data_no_sample/features_selected_Wednesday-workingHours.pcap_ISCX.csv'
    invalid_file = '../original_data_no_sample/invalid_data_Wednesday-workingHours.pcap_ISCX.csv'
    selected_features_list = selected_features.split(',')
    # _, _, output_file= select_features_from_file(input_file, selected_features_list, output_file, invalid_file)
    X, Y = load_data_from_file(output_file)
    new_X = normalize_data(X, axis=0, low=-1, high=1, eps=1e-5)
    new_Y = change_labels(Y, labels=[1, 0])  # 'BENIGN=1, others=0'
    output_file = '../original_data_no_sample/features_selected_Normalized_Wednesday-workingHours.pcap_ISCX.csv'
    save_data(new_X, new_Y, output_file)

    model = AutoEncoder(new_X, new_Y, epochs=1)
    # 1. train model
    model.train()
    # torch.save(model.state_dict(), './sim_autoencoder.pth')

    # 2. encoding data and save the encoding data
    output_file = '../original_data_no_sample/features_selected_Normalized_Reduced_data_Wednesday-workingHours.pcap_ISCX.csv'
    reduced_features_data = model.encoder(torch.Tensor(new_X))
    save_data_in_autoencoder(reduced_features_data, new_Y, output_file)

    end_time = time.strftime('%Y-%h-%d %H:%M:%S', time.localtime())
    print('It ends at ', end_time)
    print('All takes %.4f s' % (time.time() - st))
