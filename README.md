# DeepWPT

## Hardware
GPU: ZOTAC GAMING GeForce RTX 3060 Ti Twin Edge\
CPU: AMD Ryzen 9 3900X desktop processor\
RAM: 16GB 3200MHz DDR4

## Environment setup
Operating system: Tested on Ubuntu 20.04.5 LTS (Ubuntu is a popular free and open-source Linux-based operating system)\
Package management system:  conda (click [here](https://cloudsmith.com/blog/what-is-conda/) to more about conda )\
Deep Learning framework:    Pytorch (What is [Pytorch?](https://www.javatpoint.com/pytorch-introduction))\
GPU Driver Version:  [515.76](https://www.nvidia.com/en-us/drivers/results/193095/)


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
The original Implementation of "Wavelet packet transform" of our paper is available [here](https://github.com/hhb072/WaveletSRNet)
