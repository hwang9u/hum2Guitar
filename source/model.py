## Modified Pix2PixHD architecture

import torch
import torch.nn as nn

######### Discriminator #########
class ConvBlock4x4(nn.Module):
    '''
    Sub-block of SingleDiscriminator
    '''
    def __init__(self, in_dim, out_dim):
        super(ConvBlock4x4, self).__init__()
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2)
        self.norm2d = nn.InstanceNorm2d(out_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        z = self.conv2d(x)
        z = self.norm2d(z)
        y = self.activation(z)
        return y

## - Single Discriminator
class SingleDiscriminator(nn.Module):
    '''
    C64-C128-C256-C512
    '''
    def __init__(self, input_dim=1, n_layers=4, return_inter_features=False, use_sigmoid=False):
        super(SingleDiscriminator, self).__init__()
        self.return_inter_features = return_inter_features
        self.n_layers = n_layers
        self.in_dim = input_dim *2
        self.out_dim=64
        disc_sequence = []
        for i in range(self.n_layers):
            disc_sequence.append([ConvBlock4x4(in_dim=self.in_dim, out_dim=self.out_dim)])
            self.in_dim = self.out_dim
            self.out_dim = self.in_dim*2
        
        disc_sequence.append([nn.Conv2d(self.in_dim, 1, kernel_size=4, stride=1, padding=2)])
        
        if use_sigmoid:
            disc_sequence.append([nn.Sigmoid()])
            
        if return_inter_features:
            for i in range(len(disc_sequence)):
                setattr(self, f'd_{i}', nn.Sequential(*disc_sequence[i])) # ith sub-layer -> d_i
        else:
            disc_sequence_stream = []
            for i in range(len(disc_sequence)):
                disc_sequence_stream += disc_sequence[i]
            self.model = nn.Sequential(*disc_sequence_stream)
    
    def forward(self, x, z):
        if self.return_inter_features:
            x = torch.concat((x, z), dim=1 )
            y = [x] 
            for i in range(self.n_layers):
                model = getattr(self, f'd_{i}') # sublayer
                y.append(model(y[-1])) # sequential
            return y[1:]
        else: 
            return self.model(x)

## - Multi-scale Discriminator
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_dim=2, n_layers=4, use_sigmoid=False, num_D=3, return_inter_features=False):
        super(MultiscaleDiscriminator, self).__init__()                
        self.num_D=num_D
        self.n_layers = n_layers # the number of layers in single D
        self.return_inter_features=return_inter_features
        
        for n in range(num_D):
            netD = SingleDiscriminator(input_dim, n_layers=n_layers, use_sigmoid=use_sigmoid, return_inter_features=return_inter_features)
            if return_inter_features:
                for i in range(n_layers+1): 
                    setattr(self, f'scale_{n}_{i}', getattr(netD, f'd_{i}'))
            else:
                setattr(self, f'D_{n}', netD.model)
        self.downsample = nn.AvgPool2d(kernel_size=(3,1), stride=(2,1), padding=[1, 0], count_include_pad=False)
    
    def singleD_forward(self, model, x, z):
        '''
        SingleDiscriminator forward 
        만약 return_inter_features = True 이면, scale_n_i가 num_D * n_layers만큼 생성
        아니라면 D_n이 num_D만큼 생성되어 있음.
        '''
        if self.return_inter_features:
            y = [torch.concat((x,z),dim=1) ]
            for i in range(len(model)):
                y.append(model[i](y[-1]) )
            return y[1:]
        else:
            return [model(x)] 
    
    def forward(self, x, z):
        y = []
        x_down = x
        z_down = z
        for n in range(self.num_D):
            if self.return_inter_features:
                model = [getattr(self, f'scale_{n}_{i}') for i in range(self.n_layers+1)]
            else:
                model = getattr(self, f'D_{n}')
            
            y.append(self.singleD_forward(model, x_down, z_down))
            if n < (self.num_D-1):
                x_down = self.downsample(x_down) # multi-scale
                z_down = self.downsample(z_down)
        return y

######### Generator #########
class ConvDown(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0):
        super(ConvDown, self).__init__()
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect')
        self.norm2d = nn.InstanceNorm2d(out_dim)
        self.activation = nn.ReLU(True)
    def forward(self, x):
        z = self.conv2d(x)
        z = self.norm2d(z)
        y = self.activation(z)
        return y

class ConvUp(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, output_padding=0):
        super(ConvUp, self).__init__()
        self.convT2d = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', output_padding=output_padding)
        self.norm2d = nn.InstanceNorm2d(out_dim)
        self.activation = nn.ReLU(True)
    def forward(self, x):
        z = self.convT2d(x)
        z = self.norm2d(z)
        y = self.activation(z)
        return y

class ResidualBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, padding_mode='reflect'), 
                     nn.InstanceNorm2d(in_dim),
                    nn.ReLU(True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, padding_mode='reflect'), 
                     nn.InstanceNorm2d(in_dim)
                                 )
    
    def forward(self, x):
        return self.conv(x) + x

## - Global Generator
class GlobalGenerator(nn.Module):
    def __init__(self, gin_dim):
        super(GlobalGenerator, self).__init__()
        self.down = nn.Sequential(
                                nn.Conv2d(gin_dim, 64, kernel_size=3, stride=1, padding=1),
                                nn.InstanceNorm2d(64),
                                nn.ReLU(True),
                                ConvDown(64, 128),
                                ConvDown(128, 256),
                                ConvDown(256, 512)
                                )
        
        self.res = nn.Sequential(
                                ResidualBlock(512),
                                ResidualBlock(512),
                                ResidualBlock(512),
                                )
        
        self.up = nn.Sequential(
            ConvUp(in_dim=512, out_dim=256),
            ConvUp(in_dim=256, out_dim=128),
            ConvUp(in_dim=128, out_dim=64))
        
        self.final = nn.Sequential(
            nn.Conv2d(64, gin_dim, kernel_size=3, padding=(1,1) ),
            nn.Tanh()
        )
    def forward(self, x):
        z_down = self.down(x)
        z_res = self.res(z_down)
        z_up = self.up(z_res)
        z_final = self.final(z_up)
        return z_final
        
       
## - Local Enhancer
class LocalEnhancer(nn.Module):
    def __init__(self, input_dim=1, global_generator=None):
        super(LocalEnhancer,self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=5, stride=(2,1), padding=(2,2), padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            ConvDown(32, 64,kernel_size=3, stride=1, padding=1)
        )
        self.res = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        self.up = nn.Sequential(
            ConvUp(64, 32, output_padding=(1,0), stride=(2,1), padding=0),
            nn.Conv2d(32, input_dim, kernel_size=5, stride=1, padding=(1,1), padding_mode='reflect'),
            nn.Tanh()
        )
        self.downsample = nn.AvgPool2d(kernel_size=(3,1), stride=(2,1), padding=[1, 0], count_include_pad=False) # down-sampling
        self.global_generator = global_generator
        
    def forward(self, x):
        x_downs = self.downsample(x)
        global_out = self.global_generator(x_downs)
        local_down_out = self.down(x)
        local_res_out = self.res(local_down_out + global_out)
        local_up_out = self.up(local_res_out)
        return local_up_out

    
def trainable_global(model, grad = True):
    for name, p in model.named_parameters():
        if "global_generator" in name:
            p.requires_grad = grad

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
