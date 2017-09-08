#! /bin/bash

dir=$HOME/rnnlm
src=$dir/src
data=$HOME/nlp-research/corpus/one-billion-word-benchmark/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled

#python $src/rnnlm.py --word_dim 10000 --hidden_dim 100 --batch_size 15 --epoch 5 --data_size 50000 --sort 1 --network RNN --test_data_size 10000 --training_data $data/news.en-00001-of-00100 

#python $src/rnnlm.py --word_dim 10000 --hidden_dim 50 --batch_size 15 --epoch 5 --data_size 10000 --sort 1 --network RNN_BlackOut --test_data_size 3000 --training_data $data/news.en-00001-of-00100 


python $src/rnnlm.py --word_dim 10000 --hidden_dim 50 --class_dim 30 --class_type 0 --batch_size 15 --epoch 5 --sort 1 --network EFRNN --data_size 10000 --test_data_size 3000 --training_data $data/news.en-00001-of-00100 --alpha 1.0 --interval 200 --class_change_interval 50

#python $src/rnnlm.py --word_dim 10000 --hidden_dim 50 --class_dim 30 --class_type 0 --batch_size 15 --epoch 5 --sort 1 --network EFRNN --data_size 10000 --test_data_size 3000 --training_data $data/news.en-00001-of-00100 --alpha 0.5 --interval 1

#python $src/rnnlm.py --word_dim 10000 --hidden_dim 50 --batch_size 15 --epoch 5 --data_size 10000 --sort 1 --network RNNwithNCE --test_data_size 3000 --training_data $data/news.en-00001-of-00100
