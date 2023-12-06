
#python3 train.py --exp-id plot --method vat --dataset moon --plot True &&

exp_id=allfinal
python3 train.py --exp-id $exp_id --method vat  --dataset MNIST --iters 2000  --log-interval 50&&

python3 train.py --exp-id $exp_id --method wrm  --dataset MNIST --iters 2000  --log-interval 50&&

python3 train.py --exp-id $exp_id --method reg  --dataset MNIST --iters 2000 --log-interval 50&&

python3 train.py --exp-id $exp_id --method vat  --dataset FashionMNIST --iters 2000  --log-interval 50&&

python3 train.py --exp-id $exp_id --method wrm  --dataset FashionMNIST --iters 2000  --log-interval 50&&

python3 train.py --exp-id $exp_id --method reg  --dataset FashionMNIST --iters 2000 --log-interval 50&&

# python3 train.py --exp-id $exp_id --method vat  --dataset CIFAR10 --iters 200000 &&

# python3 train.py --exp-id $exp_id --method wrm  --dataset CIFAR10 --iters 200000 &&

# python3 train.py --exp-id $exp_id --method reg  --dataset CIFAR10 --iters 200000&&

python3 train.py --exp-id $exp_id --method vat  --dataset SVHN --iters 2000 --log-interval 50&&

python3 train.py --exp-id $exp_id --method wrm  --dataset SVHN --iters 2000 --log-interval 50&&

python3 train.py --exp-id $exp_id --method reg  --dataset SVHN --iters 2000 --log-interval 50