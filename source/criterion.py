import torch
import torch.nn as nn

## Feature Matching Loss
class LossFM(nn.Module):
    def __init__(self, num_D=3, n_layers=4):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.num_D = num_D
        self.n_layers= n_layers
        
    def forward(self, x_list, y_list):
        total_loss = 0
        for k in range(self.num_D):
            for i in range(self.n_layers):
                total_loss += self.l1_loss(x_list[k][i], y_list[k][i])
        return total_loss

## GAN Loss
class LossGAN(nn.Module):
    def __init__(self, ls = True):
        super(LossGAN, self).__init__()
        self.loss_func = nn.MSELoss() if ls == True else nn.BCELoss()
    
    def forward(self, y, real_label = False):
        label = torch.ones_like(y) if real_label else torch.zeros_like(y)
        loss = self.loss_func(y, label)
        return loss

## learning rate scheduler    
def lr_lambda(epoch, n_epochs, decay_epoch):
    return 1. if epoch < decay_epoch else 1 - float(epoch - decay_epoch) / (n_epochs - decay_epoch)
