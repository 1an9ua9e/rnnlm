#! /bin/bash

dir=$HOME/rnnlm
pyenv global 3.6.0
python $dir/rnnlm.py --word_dim 10000 --hidden_dim 50 --class_dim 30 --class_type 0 --batch_size 15 --epoch 1 --sort 1 --data_size 1000
