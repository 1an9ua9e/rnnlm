from datetime import datetime
import os
import math
import numpy as np
import sys
from layer import RNNLayer
from output import Softmax
import multiprocessing as mp
import itertools as itr
import utils
#from numpy.random import *
import random
'''
Negative SamplingによるRNN言語モデル
'''
class NS_Model:
    def __init__(self, unigram, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.unigram = unigram
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.k = 20 # 損失関数の勾配の近似に用いるサンプルの個数
        
        self.total_sample_size = 0
        self.negative_samples = 0
        '''
        self.samples = []
        for i in range(self.T):
            self.samples.append(self.generate_from_q())
        '''
        
    '''
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    '''
    # 単語のunigram確率を計算する。
    def q(self, x):
        #return 1.0 / self.word_dim
        #return (1.0 / self.word_dim) ** 0.75
        return self.unigram[x]

    # コーパスから構築したunigramの情報に基づき、単語を１つサンプルする。
    def generate_from_q(self):
        '''
        return int(random.uniform(0, self.word_dim))
        '''
        r = random.random()
        threshold = 0.0
        for (i,p) in enumerate(self.unigram):
            #p = 1.0 / self.word_dim
            #threshold += p ** (1.33)
            threshold += p
            if r <= threshold:
                return i
        print("WARNING generate error")
        return 0
        

                                                                                                        
    def forward_propagation(self, x, sample_list=[]):
        # The total number of time steps
        T = len(x)
        layers = []
        prev_s = np.zeros(self.hidden_dim)
        # For each time step...
        for t in range(T):
            layer = RNNLayer()
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            if sample_list != []:
                layer.forward(input, prev_s, self.U, self.W, self.V, sample_list[t*self.k:(t+1)*self.k])
            else:
                layer.forward(input, prev_s, self.U, self.W, self.V)
            prev_s = layer.s
            layers.append(layer)
        return layers


    def predict(self, x):
        output = Softmax()
        layers = self.forward_propagation(x)
        return [np.argmax(output.predict(layer.mulv)) for layer in layers]

    def calculate_loss(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.mulv, y[i])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def bptt(self, x, y, sentenceID):
        assert len(x) == len(y)
        T = len(x)
        y_sample_list = []
        for i in range(T):
            for j in range(self.k):
                a = self.generate_from_q()
                while a == y[i]:
                    a = self.generate_from_q()
                if(a == y[i]):
                    print("----- ERROR -----")
                else:
                    y_sample_list.append(a)
            
        output = Softmax()
        #layers = self.forward_propagation(x)
        layers = self.forward_propagation(x, y_sample_list)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)

        for t in range(0, T):
            
            dmulv = np.zeros(self.word_dim)
            '''
            if sentenceID >= 0:
                y_sample_list = self.negative_samples[sentenceID * ]
            else:
                y_sample_list = []
                for i in range(self.k):
                    a = self.generate_from_q()
                    y_sample_list.append(a)
            '''     
            dmulv[y[t]] = -1.0 / (np.exp(layers[t].mulv[y[t]]) + 1.0)
            for v in y_sample_list[t * self.k:(t + 1) * self.k]:
                if v != y[t]:
                    a = np.exp(layers[t].mulv[v])
                    dmulv[v] += a / (a + 1.0)
                else:
                    print("collision !!")
            ''' 
            dmulv = output.diff(layers[t].mulv, y[t])
            '''
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)
            '''
            逆伝搬計算を効率化する？
            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(
                input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv, y_sample_list[t * self.k:(t + 1) * self.k])
            '''
            prev_s_t = layers[t].s
            dmulv = np.zeros(self.word_dim)
            for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                input = np.zeros(self.word_dim)
                input[x[i]] = 1
                prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i-1].s
                dprev_s, dU_i, dW_i, dV_i = layers[i].backward(input, prev_s_i, self.U, self.W, self.V, dprev_s, dmulv)
                dU_t += dU_i
                dW_t += dW_i
            dV += dV_t
            dU += dU_t
            dW += dW_t
        return (dU, dW, dV)

    def sgd_step(self, data, sentenceID=-1):
        x = data[0]
        y = data[1]
        learning_rate = data[2]
        dU, dW, dV = self.bptt(x, y, sentenceID)
        #print(os.getpid())
        #self.U -= learning_rate * dU
        #self.V -= learning_rate * dV
        #self.W -= learning_rate * dW
        return np.array([dU,dW,dV])

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5,batch_size=1):
        num_examples_seen = 0
        losses = []
        '''
        for x in X:
            self.total_sample_size += len(x)
        self.total_sample *= self.k
        self.negative_samples = np.zeros(self.total_sample_size)
        print("size of negative samples = %d"%self.total_sample_size)
        for i in range(self.total_sample_size):
            self.negative_samples[i] = self.generate_from_q()
            sys.stdout.write("\r%d / %d"%(i+1,self.total_sample_size))
            sys.stdout.flush()
        '''
        for epoch in range(nepoch):
            # For each training example...
            data_size = len(Y)
            max_batch_loop = math.floor(data_size / batch_size)
            if(batch_size == 1):
                print("training mode : online learning")
            else:
                print("training mode : minibatch learning (batch size %d)"%batch_size)
            for i in range(max_batch_loop):
                # online learning
                num_examples_seen += batch_size
                sys.stdout.write("\r%s / %s"%(num_examples_seen,data_size))
                sys.stdout.flush()

                if batch_size <= 1:
                    sentenceID = i
                    dU,dW,dV = self.sgd_step((X[i],Y[i],learning_rate, sentenceID))
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                # minibatch learning
                else:
                    data_list = []
                    for j in range(batch_size):
                        index = i * batch_size + j
                        sentenceID = index
                        data_list.append([X[index],Y[index],learning_rate, sentenceID])
                    pool = mp.Pool(batch_size)
                    args = zip(itr.repeat(self),itr.repeat('sgd_step'),data_list)
                    dU,dW,dV = np.sum(np.array(pool.map(utils.tomap,args)),axis=0)
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                    pool.close()
                if (i+1)%20==0:
                    los = self.calculate_total_loss(X, Y)
                    print("\nloss : %.4f"%los)
                    
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss(X, Y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
                print("Training Perplexity : %.2f"%2**loss)

        return losses

    def test(self, X, Y):
        loss = self.calculate_total_loss(X, Y)
        print("Test Perplexity : %.2f" % 2**loss)
