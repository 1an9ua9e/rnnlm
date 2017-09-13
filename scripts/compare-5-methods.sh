#! /bin/bash

dir=$HOME/rnnlm
src=$dir/src
data=$HOME/nlp-research/corpus/one-billion-word-benchmark/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled
word_dim=10000
hidden_dim=50
class_dim=30
batch_size=15
epoch=5
sort=1
data_size=30000
test_data_size=10000
eos=1


#echo "-----EFRNN,alpha=1.0,eval=500,class=1000-----"
#python $src/rnnlm.py --word_dim $word_dim --hidden_dim $hidden_dim --class_dim $class_dim --class_type 0 --batch_size $batch_size --epoch $epoch --sort $sort --network EFRNN --data_size $data_size --test_data_size $test_data_size --training_data $data/news.en-00001-of-00100 --alpha 1.0 --interval 500 --class_change_interval 1000 --eos $eos

#echo "-----EFRNN,alpha=0.5,eval=500,class=100-----"
#python $src/rnnlm.py --word_dim $word_dim --hidden_dim $hidden_dim --class_dim $class_dim --class_type 0 --batch_size $batch_size --epoch $epoch --sort $sort --network EFRNN --data_size $data_size --test_data_size $test_data_size --training_data $data/news.en-00001-of-00100 --alpha 0.5 --interval 500 --class_change_interval 100 --eos $eos

#echo "-----EFRNN,alpha=1.0,eval=100,class=500-----"
#python $src/rnnlm.py --word_dim $word_dim --hidden_dim $hidden_dim --class_dim $class_dim --class_type 0 --batch_size $batch_size --epoch $epoch --sort $sort --network EFRNN --data_size $data_size --test_data_size $test_data_size --training_data $data/news.en-00001-of-00100 --alpha 1.0 --interval 100 --class_change_interval 500 --eos $eos

#echo "-----EFRNN,alpha=0.5,eval=100,class=500-----"
#python $src/rnnlm.py --word_dim $word_dim --hidden_dim $hidden_dim --class_dim $class_dim --class_type 0 --batch_size $batch_size --epoch $epoch --sort $sort --network EFRNN --data_size $data_size --test_data_size $test_data_size --training_data $data/news.en-00001-of-00100 --alpha 0.5 --interval 100 --class_change_interval 500 --eos $eos

echo "-----ClassRNN"
python $src/rnnlm.py --word_dim $word_dim --hidden_dim $hidden_dim --class_dim $class_dim --class_type 0 --batch_size $batch_size --epoch $epoch --sort $sort  --data_size $data_size --test_data_size $test_data_size --training_data $data/news.en-00001-of-00100 --eos $eos --network classRNN

echo "-----NCE-----"
python $src/rnnlm.py --word_dim $word_dim --hidden_dim $hidden_dim --batch_size $batch_size --epoch $epoch --data_size $data_size --sort $sort --network RNNwithNCE --test_data_size $test_data_size --training_data $data/news.en-00001-of-00100 --eos $eos

#echo "-----BlackOut-----"
#python $src/rnnlm.py --word_dim $word_dim --hidden_dim $hidden_dim --batch_size $batch_size --epoch $epoch --data_size $data_size --sort $sort --network RNN_BlackOut --test_data_size $test_data_size --training_data $data/news.en-00001-of-00100 --eos $eos

echo "-----RNN-----"
python $src/rnnlm.py --word_dim $word_dim --hidden_dim $hidden_dim --batch_size $batch_size --epoch $epoch --data_size $data_size --sort $sort --network RNN --test_data_size $test_data_size --training_data $data/news.en-00001-of-00100  --eos $eos
