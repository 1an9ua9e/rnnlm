from datetime import datetime
import time
import os
import math
import numpy as np
import sys
from layer import RNNLayer
from layer import RNN_NCE_Layer
from output import Softmax
import multiprocessing as mp
import itertools as itr
import utils
from numpy.random import *
import random
from record import Record

class RNN_BlackOut:
    def __init__(self, unigram, word_dim, hidden_dim=100, bptt_truncate=4):
        self.unigram = unigram
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.k = 10 # ニセの分布から生成する単語の個数
        self.max_len = 0 # 学習データの文の最大の長さ。これに基づいて予めノイズを生成する
        self.noise = 0
        self.rcd = 0
        self.prob_q = 1.0 / self.word_dim
    '''
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    '''
    # 単語のunigram確率を計算する。
    def q(self, x):
        #return self.prob_q
        #return 1 / self.word_dim
        if self.unigram[x] <= 0.0:
            return 0.000001
        return self.unigram[x]
    
    # コーパスから構築したunigramの情報に基づき、単語を１つサンプルする。
    def generate_from_q(self):
        r = random.random()
        threshold = 0.0
        for (i,p) in enumerate(self.unigram):
            #p = 1.0 / self.word_dim
            #threshold += p ** (1.33)
            #threshold += p ** 2.0
            threshold += p
            if r <= threshold:
                return i
        return 0
        #return int(random.uniform(0, self.word_dim))
    
    def forward_propagation(self, x, y=[], forward_list=[]):
        # The total number of time steps
        T = len(x)
        layers = []
        prev_s = np.zeros(self.hidden_dim)
        # For each time step...
        for t in range(T):
            layer = RNNLayer()
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
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
    
    def p(self, i, o_i, S):
        if o_i > 50.0:
            a = np.exp(50.0) / self.q(i)
        else:
            a = np.exp(o_i) / self.q(i)
        #a = np.exp(o_i)
        return a / (a + S)
        
    def bptt(self, x, y, n):
        assert len(x) == len(y)
        output = Softmax()
        T = len(x)
        '''
        samples = []
        for i in range(T * self.k):
            samples.append(self.generate_from_q())
        '''
        
        layers = self.forward_propagation(x)#, y, samples)
        
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        for t in range(0, T):
            # 学習時のみNCEを用いるプログラムではdmulvの計算を書き換えるだけで良い。
            dmulv = np.zeros(self.word_dim)
            S = 0.0
            sample = []
            sample_prob = []
            for i in range(self.k):
                qx = y[t]
                while qx == y[t] or qx in sample:
                    qx = self.generate_from_q()
                sample.append(qx)
                if layers[t].mulv[qx] > 50.0:
                    S += np.exp(50.0) / self.q(qx)
                else:
                    S += np.exp(layers[t].mulv[qx]) / self.q(qx)
                # qはどの単語に対しても同じ値になるよう設定している
                # したがって、pの項からqが消える
                #S += np.exp(layers[t].mulv[qx])
                
            dmulv[y[t]] = self.p(y[t], layers[t].mulv[y[t]], S) - 1.0
            
            for j in sample:
                sample_prob.append(self.p(j, layers[t].mulv[j], S))
                
            for j in sample:
                a = 0.0
                if layers[t].mulv[j] - layers[t].mulv[y[t]] > 50.0:
                    a = self.p(y[t], layers[t].mulv[y[t]], S) * np.exp(50.0)
                else:
                    a = self.p(y[t], layers[t].mulv[y[t]], S) * np.exp(layers[t].mulv[j] - layers[t].mulv[y[t]])
                b = 0.0
                for (ind,k) in enumerate(sample):
                    if k != j:
                        p_k = sample_prob[ind]#self.p(layers[t].mulv[k], S)
                        if layers[t].mulv[j] - layers[t].mulv[k] > 50.0:
                            b += np.exp(50.0) * p_k / (1.0 - p_k)
                        else:
                            b += np.exp(layers[t].mulv[j] - layers[t].mulv[k]) * p_k / (1.0 - p_k)
                p_j = self.p(j, layers[t].mulv[j], S)
                c = (1.0 - 2.0 * p_j) * p_j / (1.0 - p_j)
                dmulv[j] = a - b + c
            

            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(
                input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)#, y[t], samples[t * self.k:t * self.k + self.k])
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

    def sgd_step(self, data):
        x = data[0]
        y = data[1]
        learning_rate = data[2]
        n = data[3] # dataの番号
        dU, dW, dV = self.bptt(x, y, n)
        #print(os.getpid())
        #self.U -= learning_rate * dU
        #self.V -= learning_rate * dV
        #self.W -= learning_rate * dW
        return np.array([dU,dW,dV])

    def test(self, X, Y):
        loss = self.calculate_total_loss(X, Y)
        print("Test Perplexity : %.2f" % 2**loss)

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5, batch_size=1, record=False, X_test=[], Y_test=[]):
        data_size = len(Y)
        #self.rcd = Record("nce", self.hidden_dim, self.word_dim, data_size, batch_size)
        record = False
        if record:
            self.rcd.create()
        num_examples_seen = 0
        losses = []
        max_batch_loop = math.floor(data_size / batch_size)
        number = [i for i in range(max_batch_loop)] # データの処理の順番
        self.max_len = len(X[-1]) # 文の最大の長さ
        print("max length of sentence is %d"%self.max_len)
        print()
        for epoch in range(nepoch):
            print("----- Training epoch %d -----"%epoch)
            # epochごとにノイズデータを生成する
            '''
            self.noise = np.zeros(self.k * self.max_len * len(X))
            length = len(self.noise)
            for (i,v) in enumerate(self.noise):
                self.noise[i] = self.generate_from_q()
                if i % 100 == 0:
                    sys.stdout.write("\r%d / %d"%(i, length))
                    sys.stdout.flush()
            print("generate noise data : %d"%len(self.noise))
            '''
            #self.noise = [0] * data_size
            # For each training example...
            start = time.time()
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
                    dU,dW,dV = self.sgd_step((X[i],Y[i],learning_rate, i))
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                # minibatch learning
                else:
                    data_list = []
                    for j in range(batch_size):
                        index = number[i] * batch_size + j
                        data_list.append([X[index],Y[index],learning_rate, index])
                    pool = mp.Pool(batch_size)
                    args = zip(itr.repeat(self),itr.repeat('sgd_step'),data_list)
                    dU,dW,dV = np.sum(np.array(pool.map(utils.tomap,args)),axis=0)
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                    pool.close()
                
                if (i+1) % 20 == 0:
                    loss = self.calculate_total_loss(X, Y)
                    print("\Train Perplexity : %.2f"%2.0 ** loss)
            # データシャッフル
            #np.random.shuffle(number)
            print("training time %d[s]"%(time.time() - start))
            print("partition function Z = %.2f"%self.Z)
            loss = self.calculate_total_loss(X, Y,"softmax")
            losses.append((num_examples_seen, loss))
            dtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (dtime, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            ppl = 2.0 ** loss
            print("Train Perplexity : %.2f"%ppl)
            if X_test != []:
                self.test(X_test, Y_test)

        return losses

                        
