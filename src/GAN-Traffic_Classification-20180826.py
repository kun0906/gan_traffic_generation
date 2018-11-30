# -*- coding: utf-8 -*-
"""
    using D in GAN to profile normal traffic.

    created at 20180716
"""

from collections import Counter
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import optim
from torch.autograd import Variable, grad

from utilties.TrafficDataset import TrafficDataset, split_train_test
from utilties.data_preprocess import normalize_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def print_network(describe_str, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


# GAN neural network
class GAN(nn.Module):
    def __init__(self, num_features=41, batch_size=30):
        """

        :param num_classes: normal and attack, normal=0, attack=1
        :param num_features: 41
        """
        super(GAN, self).__init__()
        self.in_size = num_features
        self.h_size = 100
        self.out_size = num_classes
        self.g_in_size = 4
        self.batch_size = batch_size

        self.D = nn.Sequential(nn.Linear(self.in_size, self.h_size * 2), nn.Tanh(),
                               nn.Linear(self.h_size * 2, self.h_size), nn.Tanh(),
                               nn.Linear(self.h_size, self.out_size), nn.Sigmoid()
                               )

        self.G = nn.Sequential(nn.Linear(self.g_in_size, self.h_size), nn.Tanh(),
                               nn.Linear(self.h_size, self.h_size * 2), nn.Tanh(),
                               nn.Linear(self.h_size * 2, self.in_size), nn.Tanh()
                               )
        print('---------- Networks architecture -------------')
        print_network('D:', self.D)
        print_network('G:', self.G)
        print('-----------------------------------------------')

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.d_learning_rate = 1e-4
        self.g_learning_rate = 1e-4
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_learning_rate, betas=(0.9, 0.99))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=(0.9, 0.99))

    def run_train(self, train_loader):
        # Train the model
        self.results = {}
        self.results['train_acc'] = []
        self.results['train_loss'] = []

        self.results['test_acc'] = []
        self.results['test_loss'] = []

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['D_decision'] = []

        self.gpu_mode = False
        self.lambda_ = 8.0
        self.n_critic = 2

        for epoch in range(num_epochs):
            for i, (b_x, b_y) in enumerate(train_loader):
                # Adversarial ground truths
                valid = torch.Tensor([0.0 for _ in range(b_x.shape[0])]).view(-1, 1)
                fake = torch.Tensor([1.0 for _ in range(b_x.shape[0])]).view(-1, 1)

                b_x = b_x.to(device)
                b_y = b_y.to(device)

                b_x = b_x.view([b_x.shape[0], -1])
                b_x = Variable(b_x).float()
                b_y = Variable(b_y).type(torch.LongTensor)

                # tmp_list = b_y.data.tolist()
                # index_tmp = [step for step, i in enumerate(tmp_list) if i == 0]  # benigin_data
                # index_attack_tmp = [step for step, i in enumerate(tmp_list) if i == 1]  # attack_data
                # index_tmp_len = min(len(index_tmp), len(index_attack_tmp))

                #
                # b_x_benign = b_x[index_tmp[:index_tmp_len]]
                # b_x_attack = b_x[index_attack_tmp[:index_tmp_len]]

                # if random_flg:
                #     indexs_random = np.random.randint(0, 41, random_choose_n_features)  # select random indexs
                # else:
                #     indexs_random = [r_i for r_i in range(random_choose_n_features)]
                z_ = torch.randn((b_x.shape[0], self.g_in_size))  # random normal 0-1
                # z_ = z_.view([len(index_attack_tmp), -1])

                # update D network
                self.d_optimizer.zero_grad()
                D_real = self.D(b_x)
                D_real_loss = self.criterion(D_real, valid)
                G_ = self.G(z_)  # detach to avoid training G on these labels
                # for g_i in range(len(G_)):
                #     for g_step, g_j in enumerate(indexs_random):
                #         b_x_attack[g_i][g_j] = G_[g_i][g_step]
                #
                # G_ = b_x_attack
                D_fake = self.D(G_.detach())
                D_fake_loss = self.criterion(torch.sigmoid(D_fake-0.1), fake)

                D_loss = (D_real_loss + D_fake_loss)
                D_loss.backward()
                self.d_optimizer.step()

                if ((epoch) % self.batch_size) == 0 :
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f real:%.8f/fake:%.8f, G_loss" %
                          ((epoch), (i),
                           len(train_loader.sampler) // self.batch_size,
                           D_loss.data, D_real_loss.data, D_fake_loss.data))
                    if i % 20 == 0:
                        print('D_fake', D_fake.data.tolist())  # fake = 1.
                        print('D_real', D_real.data.tolist())  # real = 0

                if (i % self.n_critic) == 0:
                    # update G network
                    self.g_optimizer.zero_grad()
                    G_ = self.G(z_)
                    # for g_i in range(len(G_)):
                    #     for g_step, g_j in enumerate(indexs_random):
                    #         b_x_attack[g_i][g_j] = G_[g_i][g_step]
                    #
                    # G_ = b_x_attack
                    D_fake = self.D(G_)
                    G_loss = self.criterion(torch.sigmoid(D_fake-0.1), valid)
                    # self.train_hist['G_loss'].append(-G_loss.data[0])
                    # self.train_hist['G_loss'].append(D_real_loss.data - (-G_loss.data))
                    G_loss.backward()
                    self.g_optimizer.step()
                    # self.train_hist['D_loss'].append(D_loss.data[0])
                    # self.train_hist['D_loss'].append(wgan_distance.data[0])
                    # self.train_hist['D_loss'].append(wgan_distance.data)
                    # self.train_hist['D_decision'].append([D_real_loss.data, D_fake_loss.data, -G_loss.data])



            #     # self.visualize_results((epoch+1))
            self.train_hist['G_loss'].append(D_fake_loss.data)
            self.train_hist['D_loss'].append(D_real_loss.data)
            self.train_hist['D_decision'].append([D_real_loss.data, D_fake_loss.data, G_loss.data])
        show_figures(self.train_hist['D_loss'], self.train_hist['G_loss'])
        show_figures_2(self.train_hist['D_decision'])

        print("Training finish!... save training results")
        #
        # self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

        # Save the model checkpoint
        # torch.save(model.state_dict(), 'model.ckpt')

    def run_test(self, test_loader):
        # Test the model

        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        r"""model.eval(): Sets the module in evaluation mode.

               This has any effect only on certain modules. See documentations of
               particular modules for details of their behaviors in training/evaluation
               mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
               etc.
               """

        with torch.no_grad():
            correct = 0.0
            loss = 0.0
            total = 0
            cm = []
            for step, (b_x, b_y) in enumerate(test_loader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                b_x = b_x.view([b_x.shape[0], 1, -1, 1])  # (nSamples, nChannels, x_Height, x_Width)
                b_x = Variable(b_x).float()
                b_y = Variable(b_y).type(torch.LongTensor)
                # b_y_preds = model(b_x)
                b_y_preds, _, _ = self.D(b_x)
                loss += self.criterion(b_y_preds, b_y)
                _, predicted = torch.max(b_y_preds.data, 1)
                total += b_y.size(0)
                correct += (predicted == b_y).sum().item()

                if step == 0:
                    cm = confusion_matrix(b_y, predicted, labels=[0, 1, 2, 3])
                    sk_accuracy = accuracy_score(b_y, predicted) * len(b_y)
                else:
                    cm += confusion_matrix(b_y, predicted, labels=[0, 1, 2, 3])
                    sk_accuracy += accuracy_score(b_y, predicted) * len(b_y)

            print(cm, sk_accuracy / total)
            # print('Evaluation Accuracy of the model on the {} samples: {} %'.format(total, 100 * correct / total))

        acc = correct / total
        return acc, loss.data.tolist()


def show_results(data_dict, i=1):
    """

    :param data_dict:
    :param i: first_n_pkts
    :return:
    """
    import matplotlib.pyplot as plt
    # plt.subplots(1,2)

    plt.subplot(1, 2, 1)
    length = len(data_dict['train_acc'])
    plt.plot(range(length), data_dict['train_acc'], 'g-', label='train_acc')
    plt.plot(range(length), data_dict['test_acc'], 'r-', label='test_acc')
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of the first %d pkts' % i)

    plt.subplot(1, 2, 2)
    length = len(data_dict['train_loss'])
    plt.plot(range(length), data_dict['train_loss'], 'g-', label='train_loss')
    plt.plot(range(length), data_dict['test_loss'], 'r-', label='test_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of the first %d pkts' % i)

    plt.show()


def show_figures(D_loss, G_loss):
    import matplotlib.pyplot as plt
    plt.figure()

    plt.plot(D_loss, 'r', alpha=0.5, label='D_loss')
    plt.plot(G_loss, 'g', alpha=0.5, label='G_loss')
    plt.legend(loc='upper right')
    plt.show()


def show_figures_2(decision_data):
    import matplotlib.pyplot as plt
    plt.figure()
    new_decision_data = np.copy(decision_data)
    plt.plot(new_decision_data[:, 0], 'r', alpha=0.5, label='D_real')
    plt.plot(new_decision_data[:,1], 'b', alpha =0.5,label='D_fake')
    # plt.plot(new_decision_data[:, 2], 'g', alpha=0.5, label='D_G_fake')
    plt.legend(loc='upper right')
    plt.show()


def get_loader_iterators_contents(train_loader):
    X = []
    y = []
    for step, (b_x, b_y) in enumerate(train_loader):
        X.extend(b_x.data.tolist())
        y.extend(b_y.data.tolist())

    return X, y


def run_main(input_file, num_features):
    # # input_file = '../data/data_split_train_v2_711/train_%dpkt_images_merged.csv' % i
    # input_file = '../data/attack_normal_data/benign_data.csv'
    print(input_file)
    dataset = TrafficDataset(input_file, transform=None, normalization_flg=True)

    train_sampler, test_sampler = split_train_test(dataset, split_percent=0.7, shuffle=True)
    cntr = Counter(dataset.y)
    print('dataset: ', len(dataset), ' y:', sorted(cntr.items()))
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4)  # use all dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=4, sampler=train_sampler)
    X, y = get_loader_iterators_contents(train_loader)
    cntr = Counter(y)
    print('train_loader: ', len(train_loader.sampler), ' y:', sorted(cntr.items()))
    global test_loader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                              sampler=test_sampler)
    X, y = get_loader_iterators_contents(test_loader)
    cntr = Counter(y)
    print('test_loader: ', len(test_loader.sampler), ' y:', sorted(cntr.items()))

    model = GAN(num_features, batch_size=batch_size).to(device)
    model.run_train(train_loader)
    # show_results(model.results, i)

    # model.run_test(test_loader)

    # Save the model checkpoint
    # torch.save(model.state_dict(), 'wgan_model_%d.ckpt' % i)

    return model, test_loader


def two_stages_online_evaluation(benign_model, attack_model, input_file):
    """

    :param benign_model: 0
    :param attack_model: 1
    :param input_file: mix 'benign_data' and 'attack_data'
    :return:
    """
    data = []
    y_s = []
    with open(input_file, 'r') as fid_in:
        line = fid_in.readline()
        while line:
            tmp_data=line.strip().split(',')
            data.append(tmp_data[:-1])
            if tmp_data[-1] == '0':
                y = '1'
            elif tmp_data[-1] == '1':
                y = '0'
            else:
                print('y:', y)
                y = '-100'
            y_s.append(y)
            line = fid_in.readline()

    data = np.asarray(data, dtype='float')
    data=normalize_data(data,range_value=[-1, 1])
    unknow_cnt = 0
    y_preds = []

    for i in range(len(data)):
        # x = list(map(lambda t: float(t), data[i][:-1]))
        # y = data[i][-1]
        x = data[i]
        y = y_s[i]
        x = torch.from_numpy(np.asarray(x)).double()
        x = x.view([1, -1])  # (nSamples, nChannels, x_Height, x_Width)
        x = Variable(x).float()
        # # b_y = Variable(b_y).type(torch.LongTensor)
        # y_s.append(y)

        threshold1 = benign_model.D(x)
        if (threshold1 > 0.1) and (threshold1 < 0.9):
        # if threshold1 < 0.3 :
            y_pred = '0'
            print('benign_data', y, y_pred, threshold1.data.tolist())  # attack =1, benign = 0
        else:
            threshold2 = attack_model.D(x)
            if (threshold2 > 0.1) and (threshold2 < 0.9):
            # if threshold2 < 0.3:
                y_pred = '1'
                print('attack_data', y, y_pred, threshold2.data.tolist())
            else:
                y_pred = '-10'
                print('unknow_data', y, y_pred, threshold1.data.tolist(), threshold2.data.tolist())
                unknow_cnt += 1
        y_preds.append(y_pred)

    cm = confusion_matrix(y_s, y_preds, labels=['0', '1', 'unknow'])
    sk_accuracy = accuracy_score(y_s, y_preds) * len(data)
    cntr = Counter(y_s)
    print('y_s =', sorted(cntr.items()))
    cntr = Counter(y_preds)
    print('y_preds =', sorted(cntr.items()))
    print(cm, ', acc =', sk_accuracy / len(data))
    print('unknow_cnt', unknow_cnt)


def save_data(output_file, data):
    with open(output_file, 'w') as fid_out:
        for i in data:
            print('i', i.data.tolist())
            tmp = list(map(lambda x: str(x), i.data.tolist()))
            fid_out.write(','.join(tmp) + ',label=0\n')


def generated_data(benign_model, attack_model,output_file='./generated_data.csv',num=1000):

    with open(output_file,'w') as fid_out:

        z_ = torch.randn(num, benign_model.g_in_size)  # random normal 0-1
        data=benign_model.G(z_)

        for i in range(num):
            fid_out.write(','.join([str(i) for i in data[i].data.tolist()])+',1\n')

        data=attack_model.G(z_)
        for i in range(num):
            fid_out.write(','.join([str(i) for i in data[i].data.tolist()])+',0\n')

    return output_file


def merge_data(file_lst, output_file='all_in_one.csv'):
    """

    :param file_lst: [benign_normalized_file, attack_normalized_file, generated_normalized_file]
    :param output_file:
    :return:
    """
    with open(output_file,'w') as fid_out:

        for i in file_lst:
            with open(i,'r') as fid_in:
                line = fid_in.readline()
                while line:
                    fid_out.write(line)
                    line = fid_in.readline()

    return output_file



def save_to_arff(input_file, output_file, features_num=41, labels=[0, 1, 2, 3]):
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

    return output_file


if __name__ == '__main__':
    torch.manual_seed(1)
    # Hyper parameters
    global num_epochs, num_classes, batch_size
    num_epochs = 1000
    num_classes = 1
    batch_size = 64

    cross_validation_flg = False
    benign_file = '../data/attack_normal_data/benign_data.csv'
    benign_model, benign_test_loader = run_main(benign_file, num_features=41)
    # Save the model checkpoint
    torch.save(benign_model.state_dict(), 'benign_model_epoches%d.ckpt'%num_epochs)

    attack_file = '../data/attack_normal_data/attack_data.csv'
    attack_model, attack_test_loader = run_main(attack_file, num_features=41)
    # Save the model checkpoint
    torch.save(attack_model.state_dict(), 'attack_model_epochs%d.ckpt' % num_epochs)

    input_file = '../data/attack_normal_data/test_mixed.csv'
    two_stages_online_evaluation(benign_model, attack_model, input_file)

    generated_normalized_file=generated_data(benign_model,attack_model,output_file='./generated_data.csv',num=1000)
    benign_normalized_file= benign_file + '_normalized.csv'
    attack_normalized_file = attack_file + '_normalized.csv'
    file_lst = [benign_normalized_file, attack_normalized_file]
    output_file = merge_data(file_lst,output_file='benign_data+attack_data_normalized.csv')
    input_file = output_file
    save_to_arff(input_file, output_file=str(num_epochs)+'_two_in_one.arff', features_num=41, labels=[0, 1])  # two_in_one: benign_data+attack_data_normalized

    input_file = generated_normalized_file
    save_to_arff(input_file, output_file=str(num_epochs) + '_generated_in_one.arff', features_num=41,
                 labels=[0, 1])  # two_in_one: benign_data+attack_data+ generated_normalized

    file_lst = [benign_normalized_file, attack_normalized_file, generated_normalized_file]
    output_file=merge_data(file_lst, output_file='all_in_one.csv')
    input_file=output_file
    save_to_arff(input_file,output_file=str(num_epochs)+'_three_in_one.arff',features_num=41, labels=[0,1])  # two_in_one: benign_data+attack_data+ generated_normalized

