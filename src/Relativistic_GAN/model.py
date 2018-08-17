from torch.autograd import Variable
from torch.optim import Adam
from loss import GANLoss 
from module import G, D
import torch.nn.functional as F
import torch.nn as nn
import torch

class SGAN(nn.Module):
    def __init__(self, device = 'cpu', last = nn.Sigmoid):
        super(SGAN, self).__init__()
        self.device = device
        self.net_g = G()
        self.net_d = D(last = last)
        self.criterion = GANLoss(relativistic = False)  
        self.optim_G = Adam(self.net_g.parameters())
        self.optim_D = Adam(self.net_d.parameters())

    def forward(self, batch_size = 64):
        # forward generator
        z = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
        z = Variable(z).to(self.device)
        self.fake_img = self.net_g(z)

        # forward discriminator
        self.true_pred = self.net_d(self.true_img)
        self.fake_pred = self.net_d(self.fake_img)

    def optimize(self, true_img):
        self.true_img = Variable(true_img).to(self.device)
        batch_size = true_img.size(0)

        # optimize discriminator
        self.forward(batch_size)
        self.optim_D.zero_grad()
        self.loss_D = self.criterion(input = self.true_pred, target_is_real = True) + \
                        self.criterion(input = self.fake_pred, target_is_real = False)
        self.loss_D.backward()
        self.optim_D.step()

        # Optimize generator
        self.forward(batch_size)
        self.optim_G.zero_grad()
        self.loss_G = self.criterion(input = self.fake_pred, target_is_real = True)
        self.loss_G.backward()
        self.optim_G.step()

    def getInfo(self):
        return {
            'loss_D': self.loss_D.item(),
            'loss_G': self.loss_G.item()
        }

class RSGAN(SGAN):
    def __init__(self, device = 'cpu'):
        super(RSGAN, self).__init__(device = device, last = None)
        self.criterion = GANLoss(relativistic = True, average = False)

    def optimize(self, true_img, n_D = 1):
        self.true_img = Variable(true_img).to(self.device)
        batch_size = true_img.size(0)

        # Optimize discriminator
        for i in range(n_D):
            self.forward(batch_size)
            self.optim_D.zero_grad()
            self.loss_D = self.criterion(input = self.true_pred, opposite = self.fake_pred, target_is_real = True)
            self.loss_D.backward()
            self.optim_D.step()

        # Optimize generator
        self.forward(batch_size)
        self.optim_G.zero_grad()
        self.loss_G = self.criterion(input = self.fake_pred, opposite = self.true_pred, target_is_real = True)
        self.loss_G.backward()
        self.optim_G.step()

class RaSGAN(RSGAN):
    def __init__(self, device = 'cpu'):
        super(RaSGAN, self).__init__(device = device)
        self.criterion = GANLoss(relativistic = True, average = True)

    def optimize(self, true_img, n_D = 1):
        self.true_img = Variable(true_img).to(self.device)
        batch_size = true_img.size(0)

        # Optimize discriminator
        for i in range(n_D):
            self.forward(batch_size)
            self.optim_D.zero_grad()
            self.loss_D = (
                self.criterion(input = self.true_pred, opposite = self.fake_pred, target_is_real = True) + \
                self.criterion(input = self.fake_pred, opposite = self.true_pred, target_is_real = False)
            )
            self.loss_D.backward()
            self.optim_D.step()

        # Optimize generator
        self.forward(batch_size)
        self.optim_G.zero_grad()
        self.loss_G = (
            self.criterion(input = self.fake_pred, opposite = self.true_pred, target_is_real = True) + \
            self.criterion(input = self.true_pred, opposite = self.fake_pred, target_is_real = False)
        )
        self.loss_G.backward()
        self.optim_G.step()

class RaLSGAN(RaSGAN):
    def __init__(self, device = 'cpu'):
        super(RaLSGAN, self).__init__(device = device)
        self.criterion = GANLoss(use_lsgan = True, relativistic = True, average = True)