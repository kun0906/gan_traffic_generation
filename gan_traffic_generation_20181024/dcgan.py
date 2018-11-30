# -*- coding: utf-8 -*-
"""
DCGAN Tutorial
==============

**Author**: `Nathan Inkawhich <https://github.com/inkawhich>`__

"""
#%matplotlib inline
import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML

# Set random seem for reproducibility
from naive_gan import show_figures_2, show_figures, NaiveGAN

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


######################################################################
# Inputs
# ------
# 
# Let’s define some inputs for the run:
# 
# -  **dataroot** - the path to the root of the dataset folder. We will
#    talk more about the dataset in the next section
# -  **workers** - the number of worker threads for loading the data with
#    the DataLoader
# -  **batch_size** - the batch size used in training. The DCGAN paper
#    uses a batch size of 128
# -  **image_size** - the spatial size of the images used for training.
#    This implementation defaults to 64x64. If another size is desired,
#    the structures of D and G must be changed. See
#    `here <https://github.com/pytorch/examples/issues/70>`__ for more
#    details
# -  **nc** - number of color channels in the input images. For color
#    images this is 3
# -  **nz** - length of latent vector
# -  **ngf** - relates to the depth of feature maps carried through the
#    generator
# -  **ndf** - sets the depth of feature maps propagated through the
#    discriminator
# -  **num_epochs** - number of training epochs to run. Training for
#    longer will probably lead to better results but will also take much
#    longer
# -  **lr** - learning rate for training. As described in the DCGAN paper,
#    this number should be 0.0002
# -  **beta1** - beta1 hyperparameter for Adam optimizers. As described in
#    paper, this number should be 0.5
# -  **ngpu** - number of GPUs available. If this is 0, code will run in
#    CPU mode. If this number is greater than 0 it will run on that number
#    of GPUs
# 
def print_network(describe_str, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)

# Generator Code
class Generator(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, ngpu=''):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # Number of channels in the training images. For color images this is 3
        nc = 1
        # Size (channel) of z latent vector (i.e. size of generator input)
        nz = 1
        # Size of feature maps in generator
        ngf = 5
        self.main = nn.Sequential(
            # # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, kernel_size=(2,1), stride=(2,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(ngf * 8),  #  # includes 2 * (ndf*2) parameters, (gamma, beta)
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, 4 ,kernel_size=(2,1), stride=(3,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(4), #  # includes 2 * (ndf*2) parameters, (gamma, beta)
            nn.ReLU(True),
            # # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( 4, nc, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

######################################################################
# Discriminator
# ~~~~~~~~~~~~~
#
# As mentioned, the discriminator, :math:`D`, is a binary classification
# network that takes an image as input and outputs a scalar probability
# that the input image is real (as opposed to fake). Here, :math:`D` takes
# a 3x64x64 input image, processes it through a series of Conv2d,
# BatchNorm2d, and LeakyReLU layers, and outputs the final probability
# through a Sigmoid activation function. This architecture can be extended
# with more layers if necessary for the problem, but there is significance
# to the use of the strided convolution, BatchNorm, and LeakyReLUs. The
# DCGAN paper mentions it is a good practice to use strided convolution
# rather than pooling to downsample because it lets the network learn its
# own pooling function. Also batch norm and leaky relu functions promote
# healthy gradient flow which is critical for the learning process of both
# :math:`G` and :math:`D`.
#

#########################################################################
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, ngpu=''):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # Number of channels in the training images. For color images this is 3
        nc = 1
        self.nc = nc
        # Size of feature maps in discriminator
        ndf = 5
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=(2,1), stride=(1,1), padding=(0,0), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=(2,1), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(ndf * 2),   # includes 2 * (ndf*2) parameters, (gamma, beta)
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*2) x 16 x 16
            # nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(1,2), stride=(0,1), padding=(0,1),bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(1,2),stride=(0,1), padding=(0,1), bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 2, 1, kernel_size=(1,1),  stride=(1,1), padding=(0,0),bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class DCGAN(NaiveGAN):
    def __init__(self, num_epochs=10, num_features=41, batch_size=30, show_flg=False, output_dir='log',
                 GAN_name='normal_GAN', time_str=time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())):
        """

        :param num_classes: normal and attack, normal=0, attack=1
        :param num_features: 41
        """
        super(DCGAN, self).__init__()
        self.in_channel = 1
        self.out_channel = 5
        self.in_size = num_features
        self.h_size = 10
        self.out_size = 1
        self.g_in_size = 2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.show_flg = show_flg
        self.output_dir = output_dir
        self.gan_name = GAN_name
        # self.time_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        self.time_str = time_str
        print(self.time_str)

        self.D = Discriminator(self.in_size,self.h_size, self.out_size).main   # returen Sequential object
        # self.D= nn.Sequential(nn.Linear(self.in_size, self.h_size * 2), nn.ReLU(),
        #                    nn.Linear(self.h_size * 2, self.h_size), nn.ReLU(),
        #                    nn.Linear(self.h_size, self.out_size), nn.Sigmoid()
        #                    )

        self.G = Generator(self.g_in_size, self.h_size, self.in_size).main
        # self.G = nn.Sequential(nn.Linear(self.g_in_size, self.h_size), nn.ReLU(),
        #                        nn.Linear(self.h_size, self.h_size * 2), nn.ReLU(),
        #                        nn.Linear(self.h_size * 2, self.in_size), nn.Sigmoid()
        #                        )
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

    def train(self, train_set):
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

        self.n_critic = 2
        train_loader = torch.utils.data.DataLoader(train_set, self.batch_size, shuffle=True, num_workers=4)
        for epoch in range(self.num_epochs):
            for i, (b_x, b_y) in enumerate(train_loader):
                # print('i = ', i)
                # Adversarial ground truths
                valid = torch.Tensor([0.0 for _ in range(b_x.shape[0])]).view(-1, 1)  # real 0:
                fake = torch.Tensor([1.0 for _ in range(b_x.shape[0])]).view(-1, 1)  # fake : 1

                b_x = b_x.view([b_x.shape[0], 1,3, 1]).float()  #  in Pytorch, your data should be (batch x channel x height x width), this may be the problem.
                # b_x = Variable(b_x).float()
                # b_y = Variable(b_y).type(torch.LongTensor)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                # update D network
                self.d_optimizer.zero_grad()
                D_real = self.D(b_x)
                D_real_loss = self.criterion(D_real, valid)

                z_ = torch.randn((b_x.shape[0], 1, self.g_in_size, 1))  # random normal 0-1  # (bacth_size, input_channel, height, width)
                # z_ = z_.view([len(index_attack_tmp), -1])
                G_ = self.G(z_)  # detach to avoid training G on these labels
                D_fake = self.D(G_.detach())
                D_fake_loss = self.criterion(D_fake, fake)

                D_loss = (D_real_loss + D_fake_loss)
                D_loss.backward()
                self.d_optimizer.step()

                if ((epoch) % self.batch_size) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f real:%.8f/fake:%.8f, G_loss" %
                          ((epoch), (i),
                           len(train_loader.sampler) // self.batch_size,
                           D_loss.data, D_real_loss.data, D_fake_loss.data))
                    if i % 20 == 0:
                        print('D_fake', D_fake.data.tolist())  # fake = 1.
                        print('D_real', D_real.data.tolist())  # real = 0

                if (i % self.n_critic) == 0:
                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    # update G network
                    self.g_optimizer.zero_grad()

                    # z_ = torch.randn((b_x.shape[0], self.g_in_size))  # random normal 0-1
                    z_ = torch.randn((b_x.shape[0], 1, self.g_in_size,
                                      1))  # random normal 0-1  # (bacth_size, input_channel, height, width)
                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = self.criterion(D_fake, valid)

                    G_loss.backward()
                    self.g_optimizer.step()
                    # self.train_hist['D_loss'].append(D_loss.data[0])
                    # self.train_hist['D_loss'].append(wgan_distance.data[0])
                    # self.train_hist['D_loss'].append(wgan_distance.data)
                    # self.train_hist['D_decision'].append([D_real_loss.data, D_fake_loss.data, -G_loss.data])

            #     # self.visualize_results((epoch+1))
            self.train_hist['G_loss'].append(G_loss.data)
            self.train_hist['D_loss'].append(D_loss.data)
            self.train_hist['D_decision'].append(
                [sum(D_real.data) / D_real.shape[0], sum(D_fake.data) / D_fake.shape[0]])

        if self.show_flg:
            show_figures(self.train_hist['D_loss'], self.train_hist['G_loss'])
            show_figures_2(self.train_hist['D_decision'])

        print("Training finish!... save training results")
        # save_data(output_file='loss.txt', self)
        self.gan_loss_file = os.path.join(self.output_dir, self.gan_name + '_D_loss+G_loss_%s.txt'%self.time_str)
        self.save_data(output_file=self.gan_loss_file, data1=self.train_hist['D_loss'],
                       data2=self.train_hist['G_loss'])
        self.gan_decision_file = os.path.join(self.output_dir, self.gan_name + '_D_decision_%s.txt'%self.time_str)
        self.save_data_2(output_file=self.gan_decision_file, data=self.train_hist['D_decision'])
        #
        # self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

        # Save the model checkpoint
        # torch.save(model.state_dict(), 'model.ckpt')

    def generate_data(self, num):
        # gen_data = []
        # for i in range(num):
        #     z_ = torch.randn((num, self.g_in_size))  # random normal 0-1
        #     gen_data.append(self.G(z_))
        # z_ = torch.randn((num, self.g_in_size))
        z_ = torch.randn((num, 1, self.g_in_size,
                          1))  # random normal 0-1  # (bacth_size, input_channel, height, width)
        gen_data = self.G(z_)

        return gen_data.data.view(num, self.in_size)
