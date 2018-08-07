# -*- coding: utf-8 -*-
"""
    using WGAN to classification in two stages.

    created at 20180714
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def print_network(describe_str, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


# WGAN neural network (two convolutional layers)
class WGAN(nn.Module):
    def __init__(self, num_classes=2, num_features=60, batch_size=30):
        """

        :param num_classes: normal and attack, normal=0, attack=1
        :param num_features:
        """

        self.in_size = num_features
        self.h_size = 50
        self.out_size = num_classes
        self.g_in_size = 4

        self.batch_size = batch_size

        super(WGAN, self).__init__()
        # 1 input image channel, 6 output channels, 5x1 square convolution
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

        # Loss and optimizer
        # self.criterion = nn.MSELoss(size_average=False)
        self.criterion = nn.CrossEntropyLoss()
        self.d_learning_rate = 1e-4
        self.g_learning_rate = 1e-4
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_learning_rate, betas=(0.5, 0.9))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=(0.5, 0.9))

    # def forward(self, x):
    #     layer1_out = self.layer1(x)
    #     layer2_out = self.layer2(layer1_out)
    #     out = layer2_out.reshape(layer2_out.size(0), -1)
    #     out = self.fc(out)
    #     # out= nn.Softmax(out)
    #     return out, layer2_out, layer1_out
    #
    # def l1_penalty(self, var):
    #     return torch.abs(var).sum()
    #
    # def l2_penalty(self, var):
    #     return torch.sqrt(torch.pow(var, 2).sum())

    def run_train(self, train_loader, mode=True):
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
        self.n_critic = 5

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            G_loss = torch.Tensor([0.0])
            for i, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                b_x = b_x.view([b_x.shape[0], -1])
                b_x = Variable(b_x).float()
                b_y = Variable(b_y).type(torch.LongTensor)
                # Forward pass
                # b_y_preds = model(b_x)

                z_ = torch.randn((len(b_y), self.g_in_size))  # random normal 0-1
                z_ = z_.view([len(b_y), -1])
                # update D network
                self.d_optimizer.zero_grad()

                D_real = self.D(b_x)
                D_real_loss = torch.mean(D_real, dim=0)

                G_ = self.G(z_)  # detach to avoid training G on these labels
                # G_=self.G(z_)
                D_fake = self.D(G_.detach())
                D_fake_loss = torch.mean(D_fake, dim=0)

                # gradient penalty
                if self.gpu_mode:
                    alpha = torch.rand(b_x.size()).cuda()
                else:
                    alpha = torch.rand(b_x.size())

                b_xhat = Variable(alpha * b_x.data + (1 - alpha) * G_.data, requires_grad=True)

                pred_hat = self.D(b_xhat)
                if self.gpu_mode:
                    gradients = \
                        grad(outputs=pred_hat, inputs=b_xhat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=b_xhat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * (
                        (gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = -(D_real_loss - D_fake_loss) + gradient_penalty
                wgan_distance = (D_real_loss - D_fake_loss)
                #
                # l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
                # for W in self.parameters():
                #     # l2_reg = l2_reg+ W.norm(1)
                #     l2_reg = l2_reg + W.norm(2) ** 2
                #     # l2_reg +=  torch.pow(W, 2).sum()
                #
                # #
                # loss = self.criterion(b_y_preds, b_y) + 1e-3 * l2_reg

                D_loss.backward()
                self.d_optimizer.step()


                if ((i + 1) % self.n_critic) == 0:
                    # update G network
                    self.g_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(
                        D_fake)  # L = E[D(real)] - E[D(fake)], the previous one (E[D(real)]) has nothing to do with the generator. So when updating generator, we only need to consider the -E[D(fake)].
                    # self.train_hist['G_loss'].append(-G_loss.data[0])
                    # self.train_hist['G_loss'].append(D_real_loss.data - (-G_loss.data))

                    G_loss.backward()
                    self.g_optimizer.step()
                    # self.train_hist['D_loss'].append(D_loss.data[0])
                    # self.train_hist['D_loss'].append(wgan_distance.data[0])
                    # self.train_hist['D_loss'].append(wgan_distance.data)
                    # self.train_hist['D_decision'].append([D_real_loss.data, D_fake_loss.data, -G_loss.data])

                # if ((iter + 1) % self.batch_size) == 0:
                #     print("Epoch: [%2d] [%4d/%4d] wgan_distance: %.8f real:%.8f/fake:%.8f, -G_loss: %.8f" %
                #           ((epoch + 1), (iter + 1), self.training_set_data_loader.dataset.__len__() // self.batch_size,
                #            wgan_distance.data[0], D_real_loss.data[0], D_fake_loss.data[0], -G_loss.data[0]))
                # len_tmp = len(self.train_hist['D_loss'])
                # if len_tmp > 1 and len_tmp % 10 == 0:
                #     accumulated_value = 0.0
                #     j = len_tmp - 10
                #     for i in range(10):
                #         accumulated_value += abs(self.train_hist['D_loss'][j + i].data.tolist())
                #     # print(accumulated_value)
                #     if abs(
                #             accumulated_value) / 10 < self.difference_value and epoch > 1000:  # last 10 results's mean < 0.01, then break.
                #         print('training finish, it takes epochs =', epoch)
                #         break
                if ((epoch) % self.batch_size) == 0 and ((i + 1) % 20 == 0) and ((i + 1) % self.n_critic == 0):
                    print("Epoch: [%2d] [%4d/%4d] wgan_distance: %.8f real:%.8f/fake:%.8f, -G_loss: %.8f" %
                          ((epoch + 1), (i + 1),
                           len(train_loader.sampler) // self.batch_size,
                           wgan_distance.data, D_real_loss.data, D_fake_loss.data, -G_loss.data))
                    if (i + 1) % 20 == 0:
                        print('D_fake', D_fake.data.tolist())
                        print('D_real', D_real.data.tolist())

            #     # self.visualize_results((epoch+1))
            self.train_hist['G_loss'].append(D_real_loss.data - (-G_loss.data))
            self.train_hist['D_loss'].append(wgan_distance.data)
            self.train_hist['D_decision'].append([D_real_loss.data, D_fake_loss.data, -G_loss.data])
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

    # def run_test(self, test_loader):
    #     # Test the model
    #
    #     self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    #     r"""model.eval(): Sets the module in evaluation mode.
    #
    #            This has any effect only on certain modules. See documentations of
    #            particular modules for details of their behaviors in training/evaluation
    #            mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    #            etc.
    #            """
    #
    #     with torch.no_grad():
    #         correct = 0.0
    #         loss = 0.0
    #         total = 0
    #         cm = []
    #         for step, (b_x, b_y) in enumerate(test_loader):
    #             b_x = b_x.to(device)
    #             b_y = b_y.to(device)
    #             b_x = b_x.view([b_x.shape[0], -1])  # (nSamples, nChannels, x_Height, x_Width)
    #             b_x = Variable(b_x).float()
    #             b_y = Variable(b_y).type(torch.LongTensor)
    #             # b_y_preds = model(b_x)
    #             b_y_preds = self.D(b_x)
    #             print('b_y_preds_real', b_y_preds.data.tolist())
    #             _, predicted = torch.max(b_y_preds.data, 1)
    #             total += b_y.size(0)
    #             correct += (predicted == b_y).sum().item()
    #
    #             z_ = torch.randn((len(b_y), self.in_size))  # random normal 0-1
    #             z_ = z_.view([len(b_y), -1])
    #             b_y_preds = self.D(z_)
    #             print('b_y_preds_fake', b_y_preds.data.tolist())
    #
    #             # if step == 0:
    #             #     cm = confusion_matrix(b_y, predicted, labels=[0, 1, 2, 3])
    #             #     sk_accuracy = accuracy_score(b_y, predicted) * len(b_y)
    #             # else:
    #             #     cm += confusion_matrix(b_y, predicted, labels=[0, 1, 2, 3])
    #             #     sk_accuracy += accuracy_score(b_y, predicted) * len(b_y)
    #
    #         # print(cm, sk_accuracy / total)
    #         # print('Evaluation Accuracy of the model on the {} samples: {} %'.format(total, 100 * correct / total))
    #
    #     acc = correct / total
    #     # return acc, loss.data.tolist()


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

    plt.plot(D_loss, 'r', alpha=0.5, label='Wgan_distance(D_loss-G_loss)')
    # plt.plot(G_loss, 'g', alpha=0.5, label='G_loss')
    plt.legend(loc='upper right')
    plt.show()


def show_figures_2(decision_data):
    import matplotlib.pyplot as plt
    plt.figure()
    new_decision_data = np.copy(decision_data)
    plt.plot(new_decision_data[:, 0], 'r', alpha=0.5, label='D_real')
    # plt.plot(new_decision_data[:,1], 'b', alpha =0.5,label='D_fake')
    plt.plot(new_decision_data[:, 2], 'g', alpha=0.5, label='D_G_fake')
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

    model = WGAN(num_classes, num_features, batch_size=batch_size).to(device)
    model.run_train(train_loader)
    # show_results(model.results, i)

    # model.run_test(test_loader)

    # Save the model checkpoint
    # torch.save(model.state_dict(), 'wgan_model_%d.ckpt' % i)

    return model, test_loader


def run_main_cross_validation(i):
    input_file = '../data/data_split_train_v2_711/train_%dpkt_images_merged.csv' % i
    print(input_file)
    # dataset = TrafficDataset(input_file, transform=None, normalization_flg=True)
    #
    # acc_sum = 0.0
    #
    # # k_fold = KFold(n_splits=10)
    # k_fold = StratifiedKFold(n_splits=10)
    # for k, (train_idxs_k, test_idxs_k) in enumerate(k_fold.split(dataset)):
    #     print('--------------------- k = %d -------------------' % (k + 1))
    #     cntr = Counter(dataset.y)
    #     print('dataset: ', len(dataset), ' y:', sorted(cntr.items()))
    #     train_sampler = SubsetRandomSampler(train_idxs_k)
    #     test_sampler = SubsetRandomSampler(test_idxs_k)
    #     # train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4)  # use all dataset
    #     train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=4,
    #                                                sampler=train_sampler)
    #     X, y = get_loader_iterators_contents(train_loader)
    #     cntr = Counter(y)
    #     print('train_loader: ', len(train_idxs_k), ' y:', sorted(cntr.items()))
    #     global test_loader
    #     test_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=4,
    #                                               sampler=test_sampler)
    #     X, y = get_loader_iterators_contents(test_loader)
    #     cntr = Counter(y)
    #     print('test_loader: ', len(test_idxs_k), ' y:', sorted(cntr.items()))
    #
    #     model = ConvNet(num_classes, num_features=i * 60 + i - 1).to(device)
    #     model.run_train(train_loader)
    #     show_results(model.results, i)
    #
    #     # model.run_test(test_loader)
    #     acc_sum_tmp = np.sum(model.results['test_acc'])
    #     if acc_sum < acc_sum_tmp:
    #         print('***acc_sum:', acc_sum, ' < acc_sum_tmp:', acc_sum_tmp)
    #         acc_sum = acc_sum_tmp
    #         # Save the model checkpoint
    #         torch.save(model.state_dict(), 'model_%d.ckpt' % i)


def two_stages_online_evaluation(benign_model, attack_model, input_file):
    """

    :param benign_model:
    :param attack_model:
    :param input_file: mix 'benign_data' and 'attack_data'
    :return:
    """
    data = []
    with open(input_file, 'r') as fid_in:
        line = fid_in.readline()
        while line:
            data.append(line.strip().split(','))
            line = fid_in.readline()

    shuffle(data)
    unknow_cnt = 0
    y_preds = []
    y_s = []
    for i in range(len(data)):
        x = list(map(lambda t: float(t), data[i][:-1]))
        y = data[i][-1]
        x = torch.from_numpy(np.asarray(x)).double()
        x = x.view([1, -1])  # (nSamples, nChannels, x_Height, x_Width)
        x = Variable(x).float()
        # b_y = Variable(b_y).type(torch.LongTensor)
        y_s.append(y)

        threshold1 = benign_model.D(x)
        if (threshold1 > 0.3) and (threshold1 < 0.7):
            y_pred = '1'
            print('benign_data', y, y_pred, threshold1.data.tolist())  # attack =0, benign = 1
        else:
            threshold2 = attack_model.D(x)
            if (threshold2 > 0.3) and (threshold2 < 0.7):
                y_pred = '0'
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


if __name__ == '__main__':
    torch.manual_seed(1)
    # Hyper parameters
    num_epochs = 10000
    num_classes = 1
    global batch_size
    batch_size = 64
    learning_rate = 0.001

    cross_validation_flg = False
    input_file = '../data/attack_normal_data/benign_data.csv'
    benign_model, benign_test_loader = run_main(input_file, num_features=41)

    input_file = '../data/attack_normal_data/attack_data.csv'
    attack_model, attack_test_loader = run_main(input_file, num_features=41)

    input_file = '../data/attack_normal_data/test_mixed.csv'
    two_stages_online_evaluation(benign_model, attack_model, input_file)
