#! /bin/bash

dir=$HOME/rnnlm
src=$dir/src
#pyenv global 3.6.0
#python $dir/rnnlm.py --word_dim 5000 --hidden_dim 30 --class_dim 30 --class_type 0 --batch_size 15 --epoch 10 --sort 1 --data_size 1000
python $src/rnnlm.py --word_dim 10000 --hidden_dim 50 --class_dim 100 --class_type 0 --batch_size 15 --epoch 5 --sort 1 --data_size 10000
