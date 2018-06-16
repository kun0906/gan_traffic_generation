# -*- coding: utf-8 -*-
import time
from collections import Counter
import os

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable, grad
from torchvision import datasets, transforms

from utils.show_save import *

from utils.data_loader import load_data, one_hot, normalize_data, load_data_from_files


def print_network(describe_str,net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str,net)
    print('Total number of parameters: %d' % num_params)

class WGAN(nn.Module):

    def __init__(self, *args, **kwargs):
        # super(WGAN,self).__init__() # python 2.x
        super().__init__()  # python 3.x

        # divide data
        (X, Y) = args[0]
        self.training_set=(X, Y)
        # get nn size
        self.nn_size_lst = args[1]
        self.in_size = self.nn_size_lst[0]
        self.h_size = self.nn_size_lst[1]
        self.out_size = self.nn_size_lst[2]
        self.g_in_size= self.in_size//10

        self.batch_size = args[2]
        self.epochs = args[3]

        self.show_flg = args[4]
        self.data_flg=args[5]
        self.n_critic=args[6]
        self.difference_value =args[7]  # distance_threshold  # if wgan_distance < 0.1 in 10 times, then break 'while'.

        # self.map1 = nn.Linear(self.in_size, self.h_size * 2)
        # self.map2 = nn.Linear(self.h_size * 2, self.h_size)
        # self.map3 = nn.Linear(self.h_size, self.out_size)
        #
        # #
        #
        # self.model = nn.Sequential(self.map1, nn.Tanh(),
        #                            self.map2,nn.Tanh(),
        #                            self.map3
        #                            )

        self.D = nn.Sequential(nn.Linear(self.in_size, self.h_size * 2), nn.Tanh(),
                               nn.Linear(self.h_size * 2, self.h_size), nn.Tanh(),
                               nn.Linear(self.h_size, self.out_size)
                               )

        self.G = nn.Sequential(nn.Linear(self.g_in_size, self.h_size ), nn.Tanh(),
                               nn.Linear(self.h_size, self.h_size*2), nn.Tanh(),
                               nn.Linear(self.h_size*2, self.in_size), nn.Tanh()
                               )
        print('---------- Networks architecture -------------')
        print_network('D:',self.D)
        print_network('G:',self.G)
        print('-----------------------------------------------')

        # self.criterion = nn.MSELoss(size_average=False)
        self.d_learning_rate = 1e-4
        self.g_learning_rate = 1e-4
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_learning_rate, betas=(0.5,0.9))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=(0.5,0.9))

    # def forward(self, x, mini_batch_size=10):
    #     y_preds = self.model(x)
    #
    #     return y_preds
    #
    #     # x = F.elu(self.map1(x))
    #     # x = F.elu(self.map2(x))
    #     # x = F.elu(self.map2(x))
    #     # # x = F.tanh(self.map1(x))
    #     # # x = F.tanh(self.map2(x))
    #     # # x = F.tanh(self.map2(x))
    #     # return F.tanh(self.map3(x))
    #     # # return self.map3(x)

    # def train(self):
    #
    #     for epoch in range(self.epochs):
    #
    #         dataset = Data.TensorDataset(self.training_set[0], self.training_set[1])  # X, Y
    #         training_set_data_loader = Data.DataLoader(
    #             dataset=dataset,  # torch TensorDataset format
    #             batch_size=1,  # mini batch size
    #             shuffle=True,
    #             num_workers=2,
    #         )
    #
    #         for mini_batch_index, (batch_x, batch_y) in enumerate(training_set_data_loader):
    #             y_preds = self.model(batch_x)
    #             # print(y_preds)
    #             # y_preds=self.model.forward(batch_x)
    #             batch_y_one_hot = one_hot(np.reshape(batch_y.numpy(), [len(batch_y.numpy()), 1]),
    #                                       out_tensor=torch.FloatTensor(y_preds.shape[0], y_preds.shape[1]))
    #             loss = self.criterion(y_preds, batch_y_one_hot)
    #             print(mini_batch_index, 'loss:', loss)
    #
    #             self.model.zero_grad()
    #             loss.backward()
    #
    #             # # Update the weights using gradient descent. Each parameter is a Tensor, so
    #             # # we can access and gradients like we did before.
    #             # with torch.no_grad():
    #             #     for param in self.model.parameters():
    #             #         param -= self.learning_rate * param.grad
    #
    #             self.optimizer.step()
    #
    #             # Weight Clipping
    #             for p in self.model.parameters():
    #                 p.data.clamp_(-0.01, 0.01)

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
        training_x_tmp=[]
        training_y_tmp=[]
        for i in range(len(self.training_set[1].tolist())):
            if self.data_flg=='benign_data':
                if self.training_set[1].tolist()[i]==1:    # benign_data
                    training_x_tmp.append(self.training_set[0].tolist()[i])
                    training_y_tmp.append(self.training_set[1].tolist()[i])
            elif self.data_flg=='attack_data':
                if self.training_set[1].tolist()[i]==0:   # attack_data
                    training_x_tmp.append(self.training_set[0].tolist()[i])
                    training_y_tmp.append(self.training_set[1].tolist()[i])
            else:
                print('self.data_flg ',self.data_flg)
                pass
        print(self.data_flg+' training set is', len(training_y_tmp),', counter y:',Counter(training_y_tmp))
        dataset = Data.TensorDataset(torch.Tensor(training_x_tmp), torch.Tensor(training_y_tmp)) # X, Y

        start_time = time.time()
        # self.D_loss_lst=[]
        # self.G_loss_lst=[]
        epoch=0
        # self.difference_value = 0.05  # if wgan_distance < 0.1 in 10 times, then break 'while'.
        while True:
            epoch +=1
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

                z_ = torch.randn((self.batch_size, self.g_in_size))   # random normal 0-1

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
                wgan_distance=(D_real_loss - D_fake_loss)

                D_loss.backward()
                self.d_optimizer.step()

                if ((iter + 1) % self.n_critic) == 0:
                    # update G network
                    self.g_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)   # L = E[D(real)] - E[D(fake)], the previous one (E[D(real)]) has nothing to do with the generator. So when updating generator, we only need to consider the -E[D(fake)].
                    # self.train_hist['G_loss'].append(-G_loss.data[0])
                    self.train_hist['G_loss'].append(D_real_loss.data-(-G_loss.data))

                    G_loss.backward()
                    self.g_optimizer.step()

                    # self.train_hist['D_loss'].append(D_loss.data[0])
                    # self.train_hist['D_loss'].append(wgan_distance.data[0])
                    self.train_hist['D_loss'].append(wgan_distance.data)
                    self.train_hist['D_decision'].append([D_real_loss.data,D_fake_loss.data, -G_loss.data])

                # if ((iter + 1) % self.batch_size) == 0:
                #     print("Epoch: [%2d] [%4d/%4d] wgan_distance: %.8f real:%.8f/fake:%.8f, -G_loss: %.8f" %
                #           ((epoch + 1), (iter + 1), self.training_set_data_loader.dataset.__len__() // self.batch_size,
                #            wgan_distance.data[0], D_real_loss.data[0], D_fake_loss.data[0], -G_loss.data[0]))
            len_tmp=len(self.train_hist['D_loss'])
            if len_tmp > 100 and len_tmp % 10 == 0:
                accumulated_value=0.0
                j = len_tmp - 10
                for i in range(10):
                    accumulated_value += abs(self.train_hist['D_loss'][j+i].data.tolist())
                #print(accumulated_value)
                if abs(accumulated_value) / 10 < self.difference_value and epoch > 1000:  # last 10 results's mean < 0.01, then break.
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

        # self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

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
    #
    X, Y = load_data(input_file, features_selected, output_file)
    # output_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX_feature_selected.csv'
    # X, Y= load_data_from_files(output_file, sample_ratio=0.05, preprocess=True)
    # print('X.shape:', X.shape, ' Y.shape:', np.asarray(Y).shape)
    print('X[0]:', X[0])
    X = normalize_data(X, axis=0, low=-1, high=1, eps=1e-5)
    print('Normalized X[0]:', X[0])
    print('X.shape:', X.shape, ' Y.shape:', np.asarray(Y).shape)
    print('label:', Counter(Y))

    show_flg = True
    save_flg=True
    in_size = 41
    h_size = 64
    out_size = 1
    dtype = torch.float
    percent=0.05
    root_dir='./sample_wgan_data_'+str(percent)+'percent_' + time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    training_set, testing_set=split_data(X, Y, percent)
    print('training percent',percent,':training_set (Benigin and Attack)', training_set[0].shape,', testing_set (Benigin and Attack)', testing_set[0].shape)

    if save_flg:
        test_percent='%.2f'%(1-percent)
        save_tensor_data(training_set, output_file=os.path.join(root_dir,str(percent)+'_training_set.csv'))
        save_tensor_data(testing_set, output_file=os.path.join(root_dir,str(test_percent)+'_testing_set.csv'))

        training_file_lst = [os.path.join(root_dir,str(percent)+'_origin_training_set.arff'), os.path.join(root_dir,str(percent)+'_training_set.csv')]
        merge_files(training_file_lst, header='header.txt', feature_lst=features_selected)  # add_arff_header
        testing_file_lst = [os.path.join(root_dir,str(test_percent)+'_origin_testing_set.arff'), os.path.join(root_dir,str(test_percent)+'_testing_set.csv')]
        merge_files(testing_file_lst, header='header.txt', feature_lst=features_selected)  # add_arff_header


    # N = 100

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    # N, D_in, H, D_out = 64, 1000, 100, 10

    mini_batch_size = 20
    epochs = 10000
    nn_size_lst = [in_size, h_size, out_size]

    for i in range(2):
        if i ==0:
            data_flg = 'benign_data'
        else:
            data_flg='attack_data'
        wgan = WGAN((training_set[0], training_set[1]), nn_size_lst, mini_batch_size, epochs, show_flg, data_flg,critic=3, distance_threshold=0.05)

        # wgan.train()
        wgan.train_gp()

        input = torch.randn((1000, in_size//10))
        gen_data=wgan.G(input)

        output_file = os.path.join(root_dir,str(percent)+'_gen_' + data_flg + '.csv')
        if data_flg=='benign_data':
            data_type = '1'
            save_data(gen_data,data_type,output_file)
        else:
            data_type='0'
            save_data(gen_data, data_type, output_file)

        ### test
        normal_data=[]
        # achieve mini_batch_size normal data to compute the average of D
        for j in range(len(training_set[1])):
            if int(training_set[1][j].tolist())==1: # normal data
                normal_data.append(training_set[0][j])
            if len(normal_data) > mini_batch_size:
                break
        normal_data= torch.stack(normal_data)
        threshold = wgan.D(normal_data).mean()  # get threshold on average

        #test on training set
        preds_res = wgan.D(training_set[0])
        preds_res = list(map(lambda x: x[0].tolist(), preds_res))
        # evaluate(preds_res, training_set[1].tolist(), threshold[0].tolist())
        evaluate(preds_res, training_set[1].tolist(), threshold.tolist())
        # test on testing set
        preds_res = wgan.D(testing_set[0])
        preds_res= list(map(lambda x:x[0].tolist(),preds_res))
        evaluate(preds_res,testing_set[1].tolist(),threshold.tolist())

    ## merge files
    all_file_lst=[os.path.join(root_dir,str(percent)+'_all_in_one_file.csv.arff'), os.path.join(root_dir,str(percent)+'_training_set.csv'),
               os.path.join(root_dir,str(percent)+'_gen_benign_data.csv'),os.path.join(root_dir,str(percent)+'_gen_attack_data.csv')]
    merge_files(all_file_lst, header='header.txt', feature_lst=features_selected)



