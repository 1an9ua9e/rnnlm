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
X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv', args.word_dim)

np.random.seed(10)
rnn = Model(args.word_dim, args.hidden_dim)
#class_rnn = ClassModel(word_dim, hidden_dim, class_dim)
start = time.time()
losses = rnn.train(X_train[:10000], y_train[:10000], learning_rate=0.005, nepoch=1, evaluate_loss_after=1,batch_size=1)
#losses = class_rnn.train(X_train[:10000], y_train[:10000], learning_rate=0.005, nepoch=1, evaluate_loss_after=1,batch_size=1)
print("training time : %.2f[s]"%(time.time() - start))

