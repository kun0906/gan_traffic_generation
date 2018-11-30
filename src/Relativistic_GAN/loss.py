import torch.nn as nn
import torch

"""
    The relativistic version of GAN loss
    The definition of this GAN loss is 'absolutely' compatible to the old version
    After you substitute the loss with this script define, the old code can still run normally
    However, if you change some code in your optimization part, you can adopt relativistic idea directly!
    The detail can be referred to the 'model.py'
"""

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0, relativistic = True, average = True):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.use_lsgan = use_lsgan
        self.relativistic = relativistic
        self.average = average
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            if relativistic:
                self.loss = nn.BCEWithLogitsLoss()
            else:
                self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real = True, opposite = None):
        if self.relativistic and opposite is None:
            raise Exception("You should assign opposite parameter in relativistic GAN!")
        target_tensor = self.get_target_tensor(input, target_is_real)
        if not self.relativistic:
            return self.loss(input, target_tensor)
        else:
            if self.average:
                return self.loss(input - torch.mean(opposite), target_tensor)
            else:
                return self.loss(input - opposite, target_tensor)