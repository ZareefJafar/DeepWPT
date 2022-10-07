
#import libraries

import torchvision.models.vgg as vgg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models.vgg as vgg
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.utils.data import Dataset
from torch.utils import data



import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from PIL import Image
import pdb
import pickle
# import moxing as mox
# import os
# mox.file.shift('os','mox')
#img1=img1.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6)))
#img2=img2.crop((int(w/6),int(h/6),int(w*5/6),int(h*5/6)))
import matplotlib 
matplotlib.rcParams['backend'] = "Agg" 
import random




#from model_dense import *
#from dataset import * 


















#l1 loss between the image and its ground-truth in the RGBdomain,
def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input-output))
 






"""
loss_Textures: 
Determines Wavelet Reconstruction Loss

ABOUT:
Minimizing MSE loss can hardly capture high-frequency
texture details to produce satisfactory perceptual results.
As texture details can be depicted by high-frequency wave-
let coefficients, we transform the super-resolution problem
from the original image pixel domain to the wavelet domain
and introduce wavelet-domain loss functions to help texture
reconstruction.


SOURCE:
From 
Huang, Huaibo, et al. “Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination - 
International Journal of Computer Vision.” SpringerLink, Springer US, 12 Feb. 2019, https://link.springer.com/article/10.1007/s11263-019-01154-8. 

CODE:
git: https://github.com/hhb072/WaveletSRNet 
"""        
def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
  xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
  yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
  xi2 = torch.sum(xi * xi, dim=2)
  yi2 = torch.sum(yi * yi, dim=2)
  #pdb.set_trace()    #15*32*32
  out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
  return torch.mean(out)














'''
LossNetwork:

For perceptual loss
We define two perceptual loss functions that measure high-level perceptual and
semantic differences between images. They make use of a loss network φ pre-
trained for image classification, meaning that these perceptual loss functions are
themselves deep convolutional neural networks. In all our experiments φ is the
16-layer VGG network pretrained on the ImageNet dataset.

Perceptual loss functions are used when comparing two different images that look similar, 
like the same photo but shifted by one pixel. The function is used to compare high level differences, like content and style discrepancies, between images

SOURCE:
Johnson, J., Alahi, A., &amp; Fei-Fei, L. (1970, January 1). Perceptual losses for real-time style transfer and Super-Resolution. SpringerLink. 
Retrieved September 29, 2022, from https://link.springer.com/chapter/10.1007/978-3-319-46475-6_43 


CODE:
https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer


'''
class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",        #1_2 to 5_2
        }
        
    def forward(self, x):
        output = {}
        #import pdb
        #pdb.set_trace()
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        
        return output

        
 











parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="facades3", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')   
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')      
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=40, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')    # image put in the network
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')

parser.add_argument('--data_url', type=str, default="", help='name of the dataset')
parser.add_argument('--init_method', type=str, default="", help='name of the dataset')
parser.add_argument('--train_url', type=str, default="", help='name of the dataset')

opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True) #saves the trained model here

cuda = True if torch.cuda.is_available() else False #check if cuda toolkit is available or not














# Loss functions
'''
Mean Absolute Error(MAE) measures the numerical distance between predicted and true value by 
subtracting and then dividing it by the total number of data points.
'''
criterion_pixelwise = torch.nn.L1Loss() #Mean Absolute Error 
 











'''
The dense branch = detects close-range patterns
The dilation branches = far-range patterns.


The reason of using 7 pairs is to balance the computation cost and the model's accuracy. Using more pairs will surely improve the model a bit, 
but will also lead to a more complex model.
'''


# Initialize wdnet or the DeepWPD 
class WDNet(nn.Module): # Here nn.Module is the super class. nn.Module is the base class for all neural network modules.https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    def __init__(self,in_channel=3):
        super(WDNet,self).__init__() #This line imports all the methods of nn.Module as a super class

        self.cascade1=nn.Sequential(
            Conv2d(48, 64 , 1 , stride=1, padding=0), #Increasing Feature Map size from 48 to 64 using 1x1 convolutions
           
            nn.LeakyReLU(0.2, inplace=True), #LeakyReLU: It is a beneficial function if the input is negative the derivative of the function is not zero and the learning rate of the neuron does not stop. This function is used to solve the problem of dying neurons.
            
            Conv2d(64, 64 , 3 , stride=1), #stride=1 means the kernel/filter will move one pixel  at a time.
            
            nn.LeakyReLU(0.2, inplace=True), # 0.2 is the value for "negative_slope" parameter. It is used to control the angle of the negative slope.
        )

        self.cascade2=nn.Sequential( #gc= 32. Growth channel, i.e. intermediate channels.Growth rate represents the dimension of output feature mapping Defined and tested by Residual Dense Network for Image Super-Resolution, CVPR 18
                                     #delia=dilation rate. Values from Wang, P., Chen, P., Yuan, Y., Liu, D., Huang, Z., Hou, X., Cottrell, G.: Understanding convolution for semantic segmentation. In: WACV, 2018
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=1),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=2),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=5),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=7),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=12),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=19),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=31)
        )
        
        self.final=nn.Sequential(
            conv_block(64,48, kernel_size=1, norm_type=None, act_type=None) #Decreasing Feature Map size from 64 to 48 using 1x1 convolutions
        )
        
        
    def forward(self, x):
        x1 = self.cascade1(x)
        #pdb.set_trace()
        
        x1 = self.cascade2(x1)

        x = self.final(x1)
        
        return x


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd): 

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

      
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)





def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
     
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer



class DMDB2(nn.Module):
    """
    DeMoireing  Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=1):
        super(DMDB2, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

        self.deli = nn.Sequential(
            Conv2d(64, 64 , 3, stride=1, dilation=delia),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deli2 = nn.Sequential(
            Conv2d(64, 64 , 3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #self.sam1 = SAM(64,64,1)
        #self.sam2 = SAM(64,64,1)
    def forward(self, x):
        #att1 = self.sam1(x)
        #att2 = self.sam2(x)

        out = self.RDB1(x)
        out = out+x
        out2 = self.RDB2(out)
        
        out3 = self.deli(x)+0.2*self.deli2(self.deli(x))
        return out2.mul(0.2)+ out3


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)


    git: https://github.com/yjn870/RDN-pytorch
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))  #torch.cat = Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        x3 = self.conv3(torch.cat((x, x1, x2), 1)) # 1 means in dimension 1
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2)  #.......................!!!!!!!!!!!!! 


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:                                           #In our case it is 'CNA', norm_type=None, 
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


generator = WDNet()





































"""
From Huang, Huaibo, et al. “Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination - 
International Journal of Computer Vision.” SpringerLink, Springer US, 12 Feb. 2019, https://link.springer.com/article/10.1007/s11263-019-01154-8. 


git: https://github.com/hhb072/WaveletSRNet 
"""
class WaveletTransform(nn.Module): 
    def __init__(self, scale=1, dec=True, params_path='/home/zareef/projects/ibrahim_thesis/WDNet_demoire/wavelet_weights_c2.pkl', transpose=True):
        super(WaveletTransform, self).__init__()
        
        self.scale = scale #2   #object parameters
        self.dec = dec #True
        self.transpose = transpose #True
        
        ks = int(math.pow(2, self.scale)  ) #ks=2**2=4 pow means power of something 
        nc = 3 * ks * ks #3*4*4 = 48 nc means number of channels
        
        if dec: #is decomposition is true
          self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        else: #if dec = False or decomposition is False apply inverse transpose and convert to RGB image from wavelet sub-bands
          self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        
        for m in self.modules(): #self.modules() method returns an iterable to the many layers or “modules” defined in the model class
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path,'rb') #open the wavelet parameter file. rb: read-only in binary format
                u = pickle._Unpickler(f) # Unpickler Class. 
                u.encoding = 'latin1' #Using encoding='latin1' is required for unpickling NumPy arrays
                dct = u.load() #using load() method of Unpickler Class.
                #dct = pickle.load(f)
                f.close() #close the file 
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False  
                           
    def forward(self, x): 
        if self.dec:
          #pdb.set_trace()
          output = self.conv(x)          
          if self.transpose:
            osz = output.size()
            #print(osz)
            output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)            
        else:
          if self.transpose:
            xx = x
            xsz = xx.size()
            xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)         
          output = self.conv(xx)        
        return output 






wavelet_dec = WaveletTransform(scale=2, dec=True) #level1=8, level2=16, level3=32
wavelet_rec = WaveletTransform(scale=2, dec=False) #if dec = False or decomposition is False apply inverse transpose and convert to RGB image from wavelet sub-bands         
















if cuda:
    generator = generator.cuda()
    

    criterion_pixelwise.cuda()
    lossnet = LossNetwork().float().cuda()
    wavelet_dec = wavelet_dec.cuda()
    wavelet_rec = wavelet_rec.cuda()
    #generator=nn.DataParallel(generator,device_ids=[0,1])
    #discriminator=nn.DataParallel(discriminator,device_ids=[0,1])












def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.constant_(m.weight.data, 0.0)
        b,c,w,h = m.weight.shape
        cx, cy = w//2, h//2
        torch.nn.init.eye_(m.weight.data[:,:,cx,cy])
        #torch.nn.init.normal_(m.weight.data, 0.0, 0.02)                 # no use the kaming ini
        #pdb.set_trace()
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if opt.epoch != 0:
    generator.load_state_dict(torch.load('./saved_models/facades3/lastest.pth' ))#%  opt.epoch))
   

else:
    # Initialize weights
    generator.apply(weights_init_normal)
    
device = torch.device("cuda:0")










# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2)) #From https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

 








  

'''
DATASET AND DATAS PREPROSESSING

Source: 
Sun, Y., Yu, Y., &amp; Wang, W. (2018, May 8). 
Moiré photo restoration using multiresolution convolutional neural networks. arXiv.org. Retrieved October 1, 2022, from https://arxiv.org/abs/1805.02996 

CODE:
https://github.com/ZhengJun-AI/MoirePhotoRestoration-MCNN

'''
# change the root to your own data path

def default_loader(path1,path2): #source and target
    #pdb.set_trace()
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    w,h=img1.size #width and hight
    '''
    x=random.randint(0,w-256)
    y=random.randint(0,h-256)
    #iConv')!=-1:
    img1=img1.crop((x,y,x+256,y+256))
    img2=img2.crop((x,y,x+256,y+256))
    '''
    #print(path1,path2)
    
    #assert(path1.split('_')[-2]==path2.split('_')[-2])
    # demoire photo dataset
    i = random.randint(-6,6)
    j = random.randint(-6,6)
    img1=img1.crop((int(w/6)+i,int(h/6)+j,int(w*5/6)+i,int(h*5/6)+j)) #im.crop((left, upper, right, lower))
    img2=img2.crop((int(w/6)+i,int(h/6)+j,int(w*5/6)+i,int(h*5/6)+j))
    
    #img1 = img1[int(w/4):int(3*w/4),int(h/4):int(h*3/4)]
    #img2 = img2[int(w/4):int(3*w/4),int(h/4):int(h*3/4)]
    img1 = img1.resize((256,256),Image.BILINEAR )
    img2 = img2.resize((256,256),Image.BILINEAR )
    
    r= random.randint(0,1)
    if r==1:
        img1=img1.transpose(Image.FLIP_LEFT_RIGHT) #Image is a function of python pillow module 
        img2=img2.transpose(Image.FLIP_LEFT_RIGHT)
        
    t = random.randint(0,2)
    if t==0:
        img1=img1.transpose(Image.ROTATE_90)
        img2=img2.transpose(Image.ROTATE_90)
    elif t==1:
        img1=img1.transpose(Image.ROTATE_180)
        img2=img2.transpose(Image.ROTATE_180)
    elif t==2:
        img1=img1.transpose(Image.ROTATE_270)
        img2=img2.transpose(Image.ROTATE_270)
    '''
    x=random.randint(0,w-256)
    y=random.randint(0,h-256)
    #iConv')!=-1:
    img1=img1.crop((x,y,x+256,y+256))
    img2=img2.crop((x,y,x+256,y+256))
    '''
    
    '''
    minnum= min(w,h);
    if minnum>256:
        k=random.randint(0,minnum-256)
        img1=img1.crop((k,k,k+256,k+256))
        img2=img2.crop((k,k,k+256,k+256))
    else:
      img1=img1.resize((256,256));
      img2=img1.resize((256,256));
    #print(img1.size)
    '''
    return img1 ,img2

class myImageFloder(data.Dataset):
    def __init__(self,root,transform = None,target_transform = None,loader = default_loader):


        #c = 0
        imgin = []
        imgout = []
        imgin_names = []
        imgout_names = []

        for img_name in os.listdir(os.path.join(root,'source')):
            if img_name !='.' and img_name !='..':
                imgin_names.append(os.path.join(root,'source',img_name))
                
        for img_name in os.listdir(os.path.join(root,'target')):
            if img_name !='.' and img_name !='..':
                imgout_names.append(os.path.join(root,'target',img_name))
        imgin_names.sort()
        imgout_names.sort()
        #imgin_names = imgin_names[0:104814]
        #imgout_names = imgout_names[0:104814]

        print(len(imgin_names),len(imgout_names))

        assert len(imgin_names)==len(imgout_names)
        self.root = root
        self.imgin_names = imgin_names
        self.imgout_names = imgout_names
        self.transform = transform
        #self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self,index):
        imgin = self.imgin_names[index]
        imgout = self.imgout_names[index]
        
        img1,img2 = self.loader(imgin,imgout)
        
        if self.transform is not None:
            #pdb.set_trace()
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1,img2

    def __len__(self):
        return len(self.imgin_names)



"""
#This just converts your input image to PyTorch tensor. 
Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor  

source: https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose
"""        
mytransform = transforms.Compose([    
     transforms.ToTensor(),   
    ])

myfolder = myImageFloder(root = '/home/zareef/projects/datasets/trainData',  transform = mytransform)
dataloader = DataLoader(myfolder, num_workers=opt.n_cpu, batch_size=opt.batch_size, shuffle=True)
print('data loader finish！')






    























# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



def sample_images(epoch , i ,real_A,real_B,fake_B):
    data,pred,label = real_A *255 , fake_B *255, real_B *255
    data = data.cpu()
    pred = pred.cpu()
    label = label.cpu()
    #pdb.set_trace()
    pred = torch.clamp(pred.detach(),0,255)
    data,pred,label = data.int(),pred.int(),label.int()
    h,w = pred.shape[-2],pred.shape[-1]
    img = np.zeros((h,1*3*w,3))
    #pdb.set_trace()
    for idx in range(0,1):
        row = idx*h
        tmplist = [data[idx],pred[idx],label[idx]]
        for k in range(3):
            col = k*w
            tmp = np.transpose(tmplist[k],(1,2,0))
            img[row:row+h,col:col+w]=np.array(tmp)
    #pdb.set_trace()
    img = img.astype(np.uint8)
    img= Image.fromarray(img)
    img.save("./train_result/%03d_%06d.png"%(epoch,i))
















"""
From Huang, Huaibo, et al. “Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination - 
International Journal of Computer Vision.” SpringerLink, Springer US, 12 Feb. 2019, https://link.springer.com/article/10.1007/s11263-019-01154-8. 


git: https://github.com/hhb072/WaveletSRNet 
"""  
# ----------
#  Training
# ----------

prev_time = time.time()
step = 0
for epoch in range(opt.epoch, opt.n_epochs): #epoch means number of time the image dataset will be trained. 0-50 loops.
    for i, batch in enumerate(dataloader): #batch contains input(img_train[0]) and target(img_train[1]) image set.
        step = step+1
        
        # set lr rate
        current_lr = 0.0002*(1/2)**(step/100000) #learning rate
        for param_group in optimizer_G.param_groups: #Adam optimizer
            param_group["lr"] = current_lr

        
        '''
        data,pred,label
        real_A = data or moire image
        real_B =  label or ground truth
        fake_B = demoired image with DeepWPD
        '''

        # Model inputs
        img_train = batch
        real_A, real_B = Variable(img_train[0].cuda()), Variable(img_train[1].cuda()) #computes the forward pass using operations on PyTorch Variables
        #pdb.set_trace() 
        x_r = (real_A[:,0,:,:]*255-105.648186)/255.+0.5 #red channel, 
        x_g = (real_A[:,1,:,:]*255-95.4836)/255.+0.5 #green channel, 
        x_b = (real_A[:,2,:,:]*255-86.4105)/255.+0.5 # blue channel, 
        real_A = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1) #Cat() in PyTorch is used for concatenating two or more tensors in the same dimension.
                                                                                        #unsqueeze operation increases the dimension of the output tensor
  
        y_r = ((real_A[:,0,:,:]-0.5)*255+121.2556)/255.
        y_g = ((real_A[:,1,:,:]-0.5)*255+114.89969)/255.
        y_b = ((real_A[:,2,:,:]-0.5)*255+102.02478)/255.
        real_A = torch.cat([ y_r.unsqueeze(1) , y_g.unsqueeze(1) , y_b.unsqueeze(1)  ],1)
        
        #121.2556, 114.89969, 102.02478
        target_wavelets = wavelet_dec(real_B) #dec means decomposition
        batch_size = real_B.size(0)
        wavelets_lr_b = target_wavelets[:,0:3,:,:] # SR stands for Short Reach or high-frequency components
        wavelets_sr_b = target_wavelets[:,3:,:,:] # LR stands for Long Reach or low-frequency components
        
        source_wavelets = wavelet_dec(real_A)
        
        if epoch >-1 :
            
        
            optimizer_G.zero_grad()

        
            tensor_c = torch.from_numpy(np.array([123.6800, 116.7790, 103.9390]).astype(np.float32).reshape((1,3,1,1))).cuda()
           
            
            
            wavelets_fake_B_re = generator(source_wavelets)
            #wavelets_lr_fake_B = wavelets_fake_B[:,0:3,:,:]
            #wavelets_sr_fake_B = wavelets_fake_B[:,3:,:,:]
            
            fake_B = wavelet_rec(wavelets_fake_B_re) +  real_A       
            
            wavelets_fake_B    = wavelet_dec(fake_B)
            wavelets_lr_fake_B = wavelets_fake_B[:,0:3,:,:]# LR stands for Long Reach or low-frequency components
            wavelets_sr_fake_B = wavelets_fake_B[:,3:,:,:]# SR stands for Short Reach or high-frequency components
            
       
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)   #.................................


            # preceptual loss
            loss_fake_B = lossnet(fake_B*255-tensor_c)
            loss_real_B = lossnet(real_B*255-tensor_c)
            p0=compute_l1_loss(fake_B*255-tensor_c,real_B*255-tensor_c)*2
            p1=compute_l1_loss(loss_fake_B['relu1'],loss_real_B['relu1'])/2.6
            p2=compute_l1_loss(loss_fake_B['relu2'],loss_real_B['relu2'])/4.8
            #p3=compute_l1_loss(loss_fake_B['relu3'],loss_real_B['relu3'])/3.7
            #p4=compute_l1_loss(loss_fake_B['relu4'],loss_real_B['relu4'])/5.6
            #p5=compute_l1_loss(loss_fake_B['relu5'],loss_real_B['relu5'])/5.6     #   *10/1.5  
            loss_p = p0+p1+p2   #+p3+p4+p5
            
           
            loss_lr = compute_l1_loss(wavelets_lr_fake_B[:,0:3,:,:],  wavelets_lr_b ) # loss MAE for SR or Short Reach or high-frequency components.
            loss_sr = compute_l1_loss(wavelets_sr_fake_B,  wavelets_sr_b ) # loss MAE for LR or Long Reach or low-frequency components.
            loss_textures = loss_Textures(wavelets_sr_fake_B, wavelets_sr_b)
            
            





            '''
            LOSS FUNCTION

            loss_lr, loss_sr, loss_textures 
            From Huang, Huaibo, et al. “Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination - 
            International Journal of Computer Vision.” SpringerLink, Springer US, 12 Feb. 2019, https://link.springer.com/article/10.1007/s11263-019-01154-8. 
            git: https://github.com/hhb072/WaveletSRNet 
            '''
            loss_G = (1*loss_p) + loss_sr.mul(100) + loss_lr.mul(10) + loss_textures.mul(5)  # +  loss_tv  loss_pixel



            #1, 5, 10, 200, 0.01 and 1.1
            '''
            loss_p= preceptual loss
            loss_sr= mean absolute error (MAE) for Short Reach.
            loss_lr=  mean absolute error (MAE) for Long Reach.
            loss_textures = Wavelet Reconstruction Loss
           
            We did not use Attention Loss because there is no IRNN is used.

            '''






            '''
             do gradient of all parameters for which we set required_grad= True. parameters could be any variable defined in code,
            '''
            loss_G.backward() #The backward() method is used to compute the gradient during the backward pass in a neural network.


            '''
            According to the optimizer function (defined previously in our code), we update those parameters to finally get the minimum loss(error).
            When loss.backward() is called
            all it does is compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True and store them in 
            parameter.grad attribute for every parameter.
            '''
            optimizer_G.step()
           
            
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            if i%100==0:
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                         loss_G.item(),
                                                        loss_pixel.item(),
                                                        time_left)) 
            
            if i % 1000==0:
                sample_images(epoch , i ,real_A,real_B,fake_B);
                
                
        else:
            pass;
            
            
    torch.save(generator.state_dict(),'./saved_models/%s/lastest.pth'%opt.dataset_name)
    
    

    if epoch==11 or epoch==30 or epoch==39 or epoch==49 or epoch==59:
      torch.save(generator.state_dict(), './saved_models/%s/generator_%d.pth' % (opt.dataset_name, epoch))
      
      

