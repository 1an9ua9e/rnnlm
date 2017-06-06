#! /bin/bash

source $HOME/venv35/bin/activate
dir=$HOME/rnnlm

python $dir/rnnlm.py --word_dim 10000 --hidden_dim 50 --class_dim -1 --class_type 0 --batch_size 2 --epoch 1
