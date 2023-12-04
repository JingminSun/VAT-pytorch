
#python3 train.py --exp-id plot --method vat --dataset moon --plot True &&

exp_id=all1
#python3 train.py --exp-id $exp_id --method vat  --dataset MNIST --iters 2000  --log-interval 500&&
#
#python3 train.py --exp-id $exp_id --method wrm  --dataset MNIST --iters 2000  --log-interval 500&&
#
#python3 train.py --exp-id $exp_id --method reg  --dataset MNIST --iters 2000 --log-interval 500&&

python3 train.py --exp-id $exp_id --method vat  --dataset CIFAR10 &&

python3 train.py --exp-id $exp_id --method wrm  --dataset CIFAR10 &&

python3 train.py --exp-id $exp_id --method reg  --dataset CIFAR10 &&

python3 train.py --exp-id $exp_id --method vat  --dataset SVHN &&

python3 train.py --exp-id $exp_id --method wrm  --dataset SVHN &&

python3 train.py --exp-id $exp_id --method reg  --dataset SVHN