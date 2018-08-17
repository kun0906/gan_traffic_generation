import torch.nn.functional as F
import torch.nn as nn

"""
    ref: https://github.com/prcastro/pytorch-gan/blob/master/MNIST%20GAN.ipynb
"""

class G(nn.Module):
    def __init__(self, d = 32):
        super(G, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x

class D(nn.Module):
    def __init__(self, d = 32, last = None):
        super(D, self).__init__()
        model = [   
            nn.Conv2d(1, d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 4, d * 8, 4, 1),
            nn.BatchNorm2d(d * 8), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 8, 1, 4, 1)
        ]
        if last is not None:
            model += [last()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)