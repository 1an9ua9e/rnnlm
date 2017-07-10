import numpy as np
import time
from preprocessing import getSentenceData
from rnn import Model
from class_rnn import ClassModel
from gru import GRUModel
from rnn_with_nce import RNN_NCE
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--word_dim", type=int, default=5000)
parser.add_argument("--hidden_dim", type=int, default=30)
parser.add_argument("--class_dim", type=int, default=-1)
parser.add_argument("--class_type", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1, help="minibatch size")
parser.add_argument("--epoch", type=int, default=1, help="epoch")
parser.add_argument("--sort", type=bool, default=False, help="sort senteces")
parser.add_argument("--data_size", type=int, default=1000, help="Number of senteces for training.")
parser.add_argument("--network", "-n", default="RNN", help="Network Architecture")
parser.add_argument("--training_data", type=str, default="data/reddit-comments-2015-08.csv")

args = parser.parse_args()
'''
word_dim = args.word_dim
hidden_dim = args.hidden_dim
class_dim = args.class_dim
class_type = args.class_type # frequency binning
batch_size = 
'''

if args.class_dim > 0:
    X_train, y_train, index_to_class_dist, class_to_word_list = getSentenceData(
        args.training_data, args.word_dim, args.class_dim,sort=args.sort)
else:
    X_train, y_train, unigram = getSentenceData(args.training_data, args.word_dim, args.class_dim, sort=args.sort)

np.random.seed(10)

start = time.time()

if args.class_dim > 0:
    class_rnn = ClassModel(args.word_dim, args.hidden_dim, class_dim=args.class_dim,
                           index_to_class_dist=index_to_class_dist, class_to_word_list=class_to_word_list)
    losses = class_rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                             learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size)
elif args.network == "RNN":
    rnn = Model(args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size)

elif args.network == "GRU":
    rnn = GRUModel(args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size)
elif args.network == "RNNwithNCE":
    rnn = RNN_NCE(unigram, args.word_dim, args.hidden_dim)
    losses = rnn.train(X_train[:args.data_size], y_train[:args.data_size],
                       learning_rate=0.005, nepoch=args.epoch, evaluate_loss_after=1,batch_size=args.batch_size)

print("training time : %.2f[s]"%(time.time() - start))

