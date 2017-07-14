#! /bin/bash

dir=$HOME/rnnlm

python $dir/rnnlm.py --word_dim 10000 --hidden_dim 50 --batch_size 15 --epoch 1 --data_size 1000 --sort 1 --network RNN

#python $dir/rnnlm.py --word_dim 10000 --hidden_dim 50 --batch_size 15 --epoch 1 --data_size 5000 --sort 1 --network RNN --training_data data/ntcir7-98571

#python $dir/rnnlm.py --word_dim 10000 --hidden_dim 50 --batch_size 15 --epoch 10 --data_size 5000 --sort 1 --network GRU
