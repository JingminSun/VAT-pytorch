#
#python3 train.py --exp-id all --vat True --dataset MNIST &&
#
#python3 train.py --exp-id all --vat True --dataset CIFAR10 &&
#
#python3 train.py --exp-id all --vat True --dataset CIFAR100 &&

python3 train.py --exp-id all --vat False --dataset MNIST &&

python3 train.py --exp-id all --vat False --dataset CIFAR10 &&

python3 train.py --exp-id all --vat False --dataset CIFAR100