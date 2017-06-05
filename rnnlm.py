import numpy as np
import time
from preprocessing import getSentenceData
from rnn import Model

word_dim = 10000
hidden_dim = 50
X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv', word_dim)

np.random.seed(10)
rnn = Model(word_dim, hidden_dim)
start = time.time()
losses = rnn.train(X_train[:10000], y_train[:10000], learning_rate=0.005, nepoch=1, evaluate_loss_after=1,batch_size=1)
print("training time : %.2f[s]"%(time.time() - start))

