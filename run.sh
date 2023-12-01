
#python3 train.py --exp-id plot --method vat --dataset moon --plot True &&

python3 train.py --exp-id all --method vat  --dataset MNIST &&

python3 train.py --exp-id all --method wrm  --dataset MNIST &&

python3 train.py --exp-id all --method reg  --dataset MNIST &&

python3 train.py --exp-id all --method ci  --dataset CIFAR10 &&

python3 train.py --exp-id all --method wrm  --dataset CIFAR10 &&

python3 train.py --exp-id all --method reg  --dataset CIFAR10 &&

python3 train.py --exp-id all --method vat  --dataset CIFAR100 &&

python3 train.py --exp-id all --method wrm  --dataset CIFAR100 &&

python3 train.py --exp-id all --method reg  --dataset CIFAR100 &&