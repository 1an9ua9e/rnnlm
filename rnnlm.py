import numpy as np
import time
from preprocessing import getSentenceData
from rnn import Model
from class_rnn import ClassModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--word_dim")
parser.add_argument("--hidden_dim")
parser.add_argument("--class_dim")
parser.add_argument("--class_type")
parser.add_argument("--batch_size")
parser.add_argument("--epoch")
args = parser.parse_args()
'''
word_dim = args.word_dim
hidden_dim = args.hidden_dim
class_dim = args.class_dim
class_type = args.class_type # frequency binning
batch_size = 
'''

if int(args.class_dim) > 0:
    X_train, y_train, index_to_class_dist, class_to_word_list = getSentenceData('data/reddit-comments-2015-08.csv', int(args.word_dim), int(args.class_dim))
else:
    X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv', int(args.word_dim), int(args.class_dim))

np.random.seed(10)
start = time.time()

if int(args.class_dim) > 0:
    class_rnn = ClassModel(int(args.word_dim), int(args.hidden_dim), int(args.class_dim), index_to_class_dist, class_to_word_list)
    losses = class_rnn.train(X_train[:10000], y_train[:10000], learning_rate=0.005, nepoch=int(args.epoch), evaluate_loss_after=1,batch_size=int(args.batch_size))
else:
    rnn = Model(int(args.word_dim), int(args.hidden_dim))
    losses = rnn.train(X_train[:10000], y_train[:10000], learning_rate=0.005, nepoch=int(args.epoch), evaluate_loss_after=1,batch_size=int(args.batch_size))

print("training time : %.2f[s]"%(time.time() - start))

