# VAT and WRM for Pytorch

## Introduction

This is the code for a group project for the course "21-701 Intro to Machine Learning" at CMU.

We implement the Virtual Adversarial Training (VAT) and Wassertein Robustness Method (WRM)implementation for Pytorch

## Reference
### VAT
Code based on original repo of this: https://github.com/lyakaap/VAT-pytorch.git.
* Distributional Smoothing with Virtual Adversarial Training - https://arxiv.org/abs/1507.00677
* Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning - 
https://arxiv.org/abs/1704.03976

### WRM
Code based on original implementation of WRM in Tensorflow: https://github.com/duchi-lab/certifiable-distributional-robustness.git
* Certifying Some Distributional Robustness with Principled Adversarial Training https://arxiv.org/abs/1710.10571

## Requirements
* Python 3.6
* Pytorch 0.4.0
* Torchvision 0.2.1
* Numpy 1.14.3
* Matplotlib 2.2.2
* Scipy 1.1.0
* Scikit-learn 0.19.1
* TensorboardX 1.0
* Tensorflow 1.8.0 (for tensorboard)
* tqdm 4.23.4

## Usage
### Train
All models in our writeup:
```
sh run.sh
```

VAT model: 
```
ython3 train.py  --method vat --dataset MNIST --iters 2000  --log-interval 50
```
WRM model: 
```
python3 train.py  --method wrm --dataset MNIST --iters 2000  --log-interval 50
```
### Dataset
* MNIST
* CIFAR10
* CIFAR100
* SVHN
* FashionMNIST

