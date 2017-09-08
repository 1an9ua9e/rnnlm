import os
import numpy as np
import time
import datetime
from preprocessing import getSentenceData
from rnn import Model
from class_rnn import ClassModel
from gru import GRUModel
from rnn_with_nce import RNN_NCE
from rnn_with_is import IS_Model
from rnn_with_ns import NS_Model
from rnn_with_truncation import TRC_Model
from linear_two_input_rnn import LinTwoInputModel
from rnn_with_blackout import RNN_BlackOut
from efrnn import EFRNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--word_dim", type=int, default=5000)
parser.add_argument("--hidden_dim", type=int, default=30)
parser.add_argument("--class_dim", type=int, default=-1)
parser.add_argument("--class_type", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1, help="minibatch size")
parser.add_argument("--epoch", type=int, default=1, help="epoch")
parser.add_argument("--sort", type=bool, default=False, help="sort senteces")
parser.add_argument("--shuffle", type=bool, default=False, help="data shuffle per epoch")
parser.add_argument("--data_size", type=int, default=1000, help="Number of senteces for training.")
parser.add_argument("--network", "-n", default="RNN", help="Network Architecture")
parser.add_argument("--training_data", type=str, default="data/reddit-comments-2015-08.csv")
parser.add_argument("--test_data_size", type=int, default=1000)
parser.add_argument("--record", type=bool, default=False)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--class_change_interval", type=int, default=1)

args = parser.parse_args()
fname= "network" + args.network + "V-" + str(args.word_dim) + "-hidden-" + str(args.hidden_dim) + "-class-" + str(args.class_dim) + "-batch-" + str(args.batch_size) + "-epoch-" + str(args.epoch) + "-sort-" + str(args.sort) + "-shuffle-" + str(args.shuffle) + "-data_size-" + str(args.data_size) +   "test_size" +  str(args.test_data_size)
'''
word_dim = args.word_dim
hidden_dim = args.hidden_dim
class_dim = args.class_dim
class_type = args.class_type # frequency binning
batch_size = 
'''

if args.class_dim > 0:
    # ソフトクラスタリングの場合
    '''
    X_train, y_train, index_to_class_dist, class_to_word_list = getSentenceData(
        args.training_data, args.word_dim, args.class_dim,sort=args.sort)
    '''
    # ハードクラスタリングの場合
    X_train, y_train, index_to_class, class_to_word_list = getSentenceData(
        args.training_data, args.word_dim, args.class_dim,sort=args.sort)
else:
    X_train, y_train, unigram = getSentenceData(args.training_data, args.word_dim, args.class_dim, sort=args.sort)

np.random.seed(10)

start = time.time()
rnn = 0

#if args.class_dim > 0:
if args.network == "classRNN":
    # ソフトクラスタリングの場合
    '''
    rnn = ClassModel(args.word_dim, args.hidden_dim, class_dim=args.class_dim,
                           index_to_class_dist=index_to_class_dist, class_to_word_list=class_to_word_list)
    '''
    # ハードクラスタリングの場合
    rnn = ClassModel(args.word_dim, args.hidden_dim, class_dim=args.class_dim,
                           index_to_class=index_to_class, class_to_word_list=class_to_word_list)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size,
                       X_test=X_train[args.data_size:args.data_size + args.test_data_size - 1],
                       Y_test=y_train[args.data_size:args.data_size + args.test_data_size - 1])
    
elif args.network == "EFRNN":
    rnn = EFRNN(args.word_dim, args.hidden_dim, class_dim=args.class_dim,
                           index_to_class=index_to_class, class_to_word_list=class_to_word_list, alpha=args.alpha, interval=args.interval, class_change_interval=args.class_change_interval)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size,
                       X_test=X_train[args.data_size:args.data_size + args.test_data_size - 1],
                       Y_test=y_train[args.data_size:args.data_size + args.test_data_size - 1])

elif args.network == "RNN":
    rnn = Model(args.word_dim, args.hidden_dim)
    losses, test_losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size,
                       X_test=X_train[args.data_size:args.data_size + args.test_data_size - 1],
                       Y_test=y_train[args.data_size:args.data_size + args.test_data_size - 1])

elif args.network == "GRU":
    rnn = GRUModel(args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size)

elif args.network == "RNNwithNCE":
    rnn = RNN_NCE(unigram, args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size,record=args.record,
                       X_test=X_train[args.data_size:args.data_size + args.test_data_size - 1],
                       Y_test=y_train[args.data_size:args.data_size + args.test_data_size - 1])

elif args.network == "LinTwoInputRNN":
    rnn = LinTwoInputModel(args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size,
                       X_test=X_train[args.data_size:args.data_size + args.test_data_size - 1],
                       Y_test=y_train[args.data_size:args.data_size + args.test_data_size - 1])

elif args.network == "RNN_BlackOut":
    rnn = RNN_BlackOut(unigram, args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size,
                       X_test=X_train[args.data_size:args.data_size + args.test_data_size - 1],
                       Y_test=y_train[args.data_size:args.data_size + args.test_data_size - 1])

elif args.network == "RNNwithIS":
    rnn = IS_Model(unigram, args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size)

elif args.network == "RNNwithNS":
    rnn = NS_Model(unigram, args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size)

elif args.network == "RNNwithTRC":
    rnn = TRC_Model(unigram, args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size)

print("training time : %.2f[s]"%(time.time() - start))

fname = args.network + str(datetime.datetime.today()).split()[0]
if os.path.isfile("../result/" + fname + ".csv"):
    os.remove("../result/" + fname + ".csv")
f = open("../result/" + fname + ".csv", "a")
f.write("epoch,PPL\n")
for i in range(args.epoch):
    f.write("%.2f,%.2f\n"%(losses[i], test_losses[i]))
f.close()


#rnn.test(X_train[args.data_size:args.data_size + args.test_data_size - 1], y_train[args.data_size:args.data_size + args.test_data_size - 1])
