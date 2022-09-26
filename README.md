# DeepWPT

## Environment setup

Operating system: &nbsp;&nbsp;&nbsp; Tested on Ubuntu 20.04.5 LTS (Ubuntu is a popular free and open-source Linux-based operating system)\
Package management system:  conda (click [here](https://cloudsmith.com/blog/what-is-conda/) to more about conda )\
Deep Learning framework:    Pytorch (What is [Pytorch?](https://www.javatpoint.com/pytorch-introduction))\

**1. Install Anaconda distribution.**

Follow the [Anaconda Installation page](https://docs.anaconda.com/anaconda/install/linux/) for installation.

**2. Installing CIAO with conda.**

Run the following command in the terminal to install ciao, caldb and some associated software in a conda environment named “ciao-4.14” or anything you like.
```
conda create -n ciao-4.14 -c https://cxc.cfa.harvard.edu/conda/ciao -c conda-forge ciao sherpa ds9 ciao-contrib caldb marx jupyter jupyterlab numpy matplotlib astropy scipy scikit-learn pandas seaborn
```
CALDB, acis_bkgrnd and hrc_bkgrnd file download might fail because of  ```CondaHTTPError: HTTP 000 CONNECTION FAILED for url``` error or slow internet connection. If this happens remove caldb from CIAO installation command and follow the [Alternative download instructions](https://cxc.cfa.harvard.edu/ciao/threads/ciao_install_conda/index.html#alt_download)

See the [Installing CIAO with conda page](https://cxc.cfa.harvard.edu/ciao/threads/ciao_install_conda/) to know more.
