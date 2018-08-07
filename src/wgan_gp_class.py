# -*- coding: utf-8 -*-
"""
 @ wgan_gp_class

 created on 20180615
"""
__author__ = 'Learn-Live'

import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable, grad

from utilties.showcase import show_figures_2, show_figures


def print_network(describe_str, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


class WGAN_GP(nn.Module):

    def __init__(self, *args, **kwargs):
        # super(GAN,self).__init__() # python 2.x
        super().__init__()  # python 3.x

        # divide data
        (X, Y) = args[0]
        self.training_set = (X, Y)
        # get nn size
        self.nn_size_lst = args[1]
        self.in_size = self.nn_size_lst[0]
        self.h_size = self.nn_size_lst[1]
        self.out_size = self.nn_size_lst[2]
        self.g_in_size = self.nn_size_lst[3]

        self.batch_size = args[2]
        self.epochs = args[3]

        self.show_flg = args[4]
        self.data_flg = args[5]
        self.n_critic = args[6]
        self.difference_value = args[7]  # distance_threshold  # if wgan_distance < 0.1 in 10 times, then break 'while'.

        self.D = nn.Sequential(nn.Linear(self.in_size, self.h_size * 2), nn.Tanh(),
                               nn.Linear(self.h_size * 2, self.h_size), nn.Tanh(),
                               nn.Linear(self.h_size, self.out_size)
                               )

        self.G = nn.Sequential(nn.Linear(self.g_in_size, self.h_size), nn.Tanh(),
                               nn.Linear(self.h_size, self.h_size * 2), nn.Tanh(),
                               nn.Linear(self.h_size * 2, self.in_size), nn.Tanh()
                               )
        print('---------- Networks architecture -------------')
        print_network('D:', self.D)
        print_network('G:', self.G)
        print('-----------------------------------------------')

        # self.criterion = nn.MSELoss(size_average=False)
        self.d_learning_rate = 1e-4
        self.g_learning_rate = 1e-4
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_learning_rate, betas=(0.5, 0.9))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=(0.5, 0.9))

    def train_gp(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['D_decision'] = []

        self.gpu_mode = False
        self.lambda_ = 8.0
        self.n_critic = 1

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(
                torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        # self.D.train()  # only useful for dropout, batch_normalizati
        print('training start!!')
        training_x_tmp = []
        training_y_tmp = []
        for i in range(len(self.training_set[1].tolist())):
            if self.data_flg == 'BENIGN':
                if self.training_set[1].tolist()[i] == 1:  # benign_data
                    training_x_tmp.append(self.training_set[0].tolist()[i])
                    training_y_tmp.append(self.training_set[1].tolist()[i])
            elif self.data_flg == 'ATTACK':
                if self.training_set[1].tolist()[i] == 0:  # attack_data
                    training_x_tmp.append(self.training_set[0].tolist()[i])
                    training_y_tmp.append(self.training_set[1].tolist()[i])
            else:
                print('self.data_flg ', self.data_flg)
                pass
        print(self.data_flg + ' training set is', len(training_y_tmp), ', counter y:', Counter(training_y_tmp))
        dataset = Data.TensorDataset(torch.Tensor(training_x_tmp), torch.Tensor(training_y_tmp))  # X, Y

        start_time = time.time()
        # self.D_loss_lst=[]
        # self.G_loss_lst=[]
        epoch = 0
        # self.difference_value = 0.05  # if wgan_distance < 0.1 in 10 times, then break 'while'.
        while True:
            epoch += 1
            # for epoch in range(self.epochs):
            # self.G.train()
            epoch_start_time = time.time()

            ### re divide dataset
            self.training_set_data_loader = Data.DataLoader(
                dataset=dataset,  # torch TensorDataset format
                batch_size=self.batch_size,  # mini batch size
                shuffle=True,
                num_workers=2,
            )
            for iter, (x_, y_) in enumerate(self.training_set_data_loader):
                if iter == self.training_set_data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.randn((self.batch_size, self.g_in_size))  # random normal 0-1

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                # update D network
                self.d_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = torch.mean(D_real)

                G_ = self.G(z_)  # detach to avoid training G on these labels
                # G_=self.G(z_)
                D_fake = self.D(G_.detach())
                D_fake_loss = torch.mean(D_fake)

                # gradient penalty
                if self.gpu_mode:
                    alpha = torch.rand(x_.size()).cuda()
                else:
                    alpha = torch.rand(x_.size())

                x_hat = Variable(alpha * x_.data + (1 - alpha) * G_.data, requires_grad=True)

                pred_hat = self.D(x_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = -(D_real_loss - D_fake_loss) + gradient_penalty
                wgan_distance = (D_real_loss - D_fake_loss)

                D_loss.backward()
                self.d_optimizer.step()

                if ((iter + 1) % self.n_critic) == 0:
                    # update G network
                    self.g_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(
                        D_fake)  # L = E[D(real)] - E[D(fake)], the previous one (E[D(real)]) has nothing to do with the generator. So when updating generator, we only need to consider the -E[D(fake)].
                    # self.train_hist['G_loss'].append(-G_loss.data[0])
                    self.train_hist['G_loss'].append(D_real_loss.data - (-G_loss.data))

                    G_loss.backward()
                    self.g_optimizer.step()

                    # self.train_hist['D_loss'].append(D_loss.data[0])
                    # self.train_hist['D_loss'].append(wgan_distance.data[0])
                    self.train_hist['D_loss'].append(wgan_distance.data)
                    self.train_hist['D_decision'].append([D_real_loss.data, D_fake_loss.data, -G_loss.data])

                # if ((iter + 1) % self.batch_size) == 0:
                #     print("Epoch: [%2d] [%4d/%4d] wgan_distance: %.8f real:%.8f/fake:%.8f, -G_loss: %.8f" %
                #           ((epoch + 1), (iter + 1), self.training_set_data_loader.dataset.__len__() // self.batch_size,
                #            wgan_distance.data[0], D_real_loss.data[0], D_fake_loss.data[0], -G_loss.data[0]))
            len_tmp = len(self.train_hist['D_loss'])
            if len_tmp > 100 and len_tmp % 10 == 0:
                accumulated_value = 0.0
                j = len_tmp - 10
                for i in range(10):
                    accumulated_value += abs(self.train_hist['D_loss'][j + i].data.tolist())
                # print(accumulated_value)
                if abs(
                        accumulated_value) / 10 < self.difference_value and epoch > 10:  # last 10 results's mean < 0.01, then break.
                    print('training finish, it takes epochs =', epoch)
                    break
            if ((epoch) % self.batch_size) == 0:
                print("Epoch: [%2d] [%4d/%4d] wgan_distance: %.8f real:%.8f/fake:%.8f, -G_loss: %.8f" %
                      ((epoch + 1), (iter + 1), self.training_set_data_loader.dataset.__len__() // self.batch_size,
                       wgan_distance.data, D_real_loss.data, D_fake_loss.data, -G_loss.data))
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            # self.visualize_results((epoch+1))
        show_figures(self.train_hist['D_loss'], self.train_hist['G_loss'])
        show_figures_2(self.train_hist['D_decision'])

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible

    input_file = '../data/ids2017_sampled.csv'
    output_file = '../data/ids_selected_features_data.csv'
    #
    features_selected = [' Source Port', ' Destination Port', ' Flow Duration', 'Total Length of Fwd Packets',
                         ' Total Length of Bwd Packets', 'Bwd Packet Length Max', ' Bwd Packet Length Min',
                         'Flow Bytes/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                         'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Min', 'Bwd IAT Total',
                         ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
                         ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Packets/s', ' Packet Length Mean',
                         ' ACK Flag Count', ' Down/Up Ratio', ' Avg Fwd Segment Size', ' Fwd Header Length.1',
                         'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Bwd Avg Bytes/Bulk', 'Bwd Avg Bulk Rate',
                         'Subflow Fwd Packets', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward', ' act_data_pkt_fwd',
                         ' Active Std', ' Active Min', ' Idle Max']
