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

[Wavelets: a mathematical microscope](https://www.youtube.com/watch?v=jnxqHcObNK4&t=1405s)



### Loss Function:
The original Implementation of loss function we used : “[Wavelet Domain Generative Adversarial Network for Multi-Scale Face Hallucination](https://link.springer.com/article/10.1007/s11263-019-01154-8)", [code](https://github.com/hhb072/WaveletSRNet/blob/f0219900056c505143d9831b44a112453784b2a7/main.py)


### Dataset:
Download dataset from [here](https://drive.google.com/drive/folders/109cAIZ0ffKLt34P7hOMKUO14j3gww2UC)

paper: "[Moiré Photo Restoration Using Multiresolution
Convolutional Neural Networks](https://arxiv.org/abs/1805.02996)"

[Code](https://github.com/ZhengJun-AI/MoirePhotoRestoration-MCNN)


images = 130,307 pair (90% for training and 10% testing) of RGB images.

Type: PNG

Resolution: 256 * 256

Created from: [ImageNet ISVRC 2012 dataset](https://image-net.org/download.php)


### Directional Residual Dense Network:

Used the Residual Dense Block (RDB) from “[Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)”

Code of [RDB](https://github.com/yjn870/RDN-pytorch)
