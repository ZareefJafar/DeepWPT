# DeepWPT

## Hardware
GPU: ZOTAC GAMING GeForce RTX 3060 Ti Twin Edge\
CPU: AMD Ryzen 9 3900X desktop processor\
RAM: 16GB 3200MHz DDR4

## Environment setup
Operating system: Tested on Ubuntu 20.04.5 LTS (Ubuntu is a popular free and open-source Linux-based operating system)\
Package management system:  conda (click [here](https://cloudsmith.com/blog/what-is-conda/) to more about conda )\
Deep Learning framework:    Pytorch (What is [Pytorch?](https://www.javatpoint.com/pytorch-introduction))\
GPU Driver Version:  [515.76](https://www.nvidia.com/en-us/drivers/results/193095/)\
Python version: Python 3.9.13\
cudatoolkit version: 11.6.   [what is cudatoolkit?](https://anaconda.org/nvidia/cudatoolkit)

**1. Install Anaconda distribution.**

Follow the [Anaconda Installation page](https://docs.anaconda.com/anaconda/install/linux/) for installation.

**2. Installing pytorch with conda.**

Create a conda environment with ```conda create -n pytorch```

Activate the new environment with ```conda activate pytorch```

Go to the pytorch official [website](https://pytorch.org/) and select the following settings shown in the image.


![INSTALL PYTORCH](https://github.com/ZareefJafar/DeepWPT/blob/main/pytorch.png)

Copy the ```conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge``` in the terminal and run.

## Code explanation
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
  
  We define two perceptual loss functions that measure high-level perceptual and
  semantic differences between images. They make use of a loss network φ pre-
  trained for image classification, meaning that these perceptual loss functions are
  themselves deep convolutional neural networks. In all our experiments φ is the
  16-layer VGG network pretrained on the ImageNet dataset.

  Perceptual loss functions are used when comparing two different images that look similar, 
  like the same photo but shifted by one pixel. The function is used to compare high level differences, like content and style discrepancies, between       images

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
