# DeepWPT

## Hardware
GPU: ZOTAC GAMING GeForce RTX 3060 Ti Twin Edge\
CPU: AMD Ryzen 9 3900X desktop processor\
RAM: 16GB 3200MHz DDR4

## Environment setup
Operating system: Tested on Ubuntu 20.04.5 LTS \
Package management system:  conda\
Deep Learning framework: Pytorch\
GPU Driver Version:  [515.76](https://www.nvidia.com/en-us/drivers/results/193095/)\
Python version: Python 3.9.13\
cudatoolkit version: 11.6.



## Code explanation

### Some terminologies:
**gc**: Growth channel or intermediate channels. Growth rate represents the dimension of output feature mapping Defined and tested by Residual Dense Network for Image Super-Resolution, CVPR 18

**1x1 convolutions**: Used to Increase or decrease  Feature Map size. (e.g from 48 to 64 channels and from 64 channels to 48)

**stride=1**: means the kernel/filter will move one pixel  at a time.

**Parameters vs hyperparameters**: see this [video](https://www.youtube.com/watch?v=V4AcLJ2cgmU)


**VGG19**: what is [vgg19](https://deepchecks.com/glossary/vggnet/)

**Epoch**: epochs is a hyperparameter that defines the number times that the learning 
algorithm will work through the entire training dataset.(We set it to 50)




### Wavelet packet transform:
The original Implementation of "Wavelet packet transform" of our paper is from "[Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination](https://link.springer.com/article/10.1007/s11263-019-01154-8)",  [code](https://github.com/hhb072/WaveletSRNet/blob/f0219900056c505143d9831b44a112453784b2a7/networks.py)


Some resource to understand wavelet and it's different implementation:

1. [Wavelets: a mathematical microscope](https://www.youtube.com/watch?v=jnxqHcObNK4&t=1405s)

2. [Discrete Wavelet Transform of Images (Haar and Hadamard)](https://www.youtube.com/watch?v=1BTyUIPMMbw&t=1655s)


### Loss Function:

loss_G = (1*loss_p) + loss_sr.mul(100) + loss_lr.mul(10) + loss_textures.mul(5)

We did not use Attention Loss because there is no IRNN is used.


1. loss_p= preceptual loss

    ABOUT:

    Perceptual loss functions are used when comparing two different images that look similar, 
    like the same photo but shifted by one pixel. The function is used to compare high level differences.
    
    In instances where we want to know if two images look like each-other, we could use a mathematical equation to compare the images but this is             unlikely to produce good results. Two images can look the same to humans but be very different mathematically (i.e. if there is a picture of a man vs     the same picture of the man but the man is shifted one pixel to the left). Using a perceptual loss function solves this issue by taking a neural         network that recognizes features of the image; these can include autoencoders, image classifiers, etc.
    
    They make use of a loss network φ pre-
    trained for image classification, meaning that these perceptual loss functions are
    themselves deep convolutional neural networks. In all our experiments φ is the
    16-layer VGG network pretrained on the ImageNet dataset.

    [SOURCE](https://link.springer.com/article/10.1007/s10845-022-02003-1)

    [CODE](https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer)
  
  
  
2. loss_sr= loss MAE(mean absolute error) for SR or Short Reach or high-frequency components.

   loss_lr=  loss MAE(mean absolute error) for LR or Long Reach or low-frequency components.
   
    [SOURCE](https://link.springer.com/article/10.1007/s11263-019-01154-8)

    [CODE](https://github.com/hhb072/WaveletSRNet )



3. loss_textures = Wavelet Reconstruction Loss


    ABOUT:

    Minimizing MSE loss can hardly capture high-frequency
    texture details to produce satisfactory perceptual results.
    As texture details can be depicted by high-frequency wave-
    let coefficients, we transform the super-resolution problem
    from the original image pixel domain to the wavelet domain
    and introduce wavelet-domain loss functions to help texture
    reconstruction.

    [SOURCE](https://link.springer.com/article/10.1007/s11263-019-01154-8)

    [CODE](https://github.com/hhb072/WaveletSRNet )





### Dataset:
Download dataset from [here](https://drive.google.com/drive/folders/109cAIZ0ffKLt34P7hOMKUO14j3gww2UC)

paper: "[Moiré Photo Restoration Using Multiresolution
Convolutional Neural Networks](https://arxiv.org/abs/1805.02996)"

[Code](https://github.com/ZhengJun-AI/MoirePhotoRestoration-MCNN)


images = 130,307 pair (90% for training and 10% testing) of RGB images.

Type: PNG

Resolution: average 850x850. Converted to 256x256 for training and testing.  

Created from: [ImageNet ISVRC 2012 dataset](https://image-net.org/download.php)


### Directional Residual Dense Network:

Used the Residual Dense Block (RDB) from “[Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)”

Code of [RDB](https://github.com/yjn870/RDN-pytorch)
