#! /bin/bash

dir=$HOME/rnnlm

python $dir/rnnlm.py --word_dim 10000 --hidden_dim 50 --batch_size 15 --epoch 10 --data_size 5000 --sort 1 --network RNN

#python $dir/rnnlm.py --word_dim 10000 --hidden_dim 50 --batch_size 15 --epoch 10 --data_size 5000 --sort 1 --network GRU
