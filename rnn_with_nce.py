from datetime import datetime
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

class RNN_NCE:
    def __init__(self, unigram, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.unigram = unigram
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.Z = 10.0 # NCEで推定すべき分配関数
        self.k = 10 # ニセの分布から生成する単語の個数
    '''
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    '''
    # 単語のunigram確率を計算する。
    def q(self, x):
        #return 1 / self.word_dim
        return self.unigram[x]
    
    # コーパスから構築したunigramの情報に基づき、単語を１つサンプルする。
    def generate_from_q(self):
        r = rand()
        threshold = 0.0
        for (i,p) in enumerate(self.unigram):
            threshold += p
            if r <= threshold:
                return i
                
        return 0
    
    def forward_propagation(self, x):
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
            loss += np.log(abs(self.Z)) - layer.mulv[y[i]]
            #loss -= layer.mulv[y[i]]
            #loss += output.loss(layer.mulv, y[i])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))
    
    # データが真の分布から生成されるときの確率
    def true_prob(self, x_t, o_j, D):
        a = 30.0
        if D == 1:
            if o_j > a:
                return 1.0
            elif o_j < -a:
                return 0.0
            else:
                return np.exp(o_j) / (np.exp(o_j) + self.k * self.q(x_t)) * self.Z
        else:
            if o_j > a:
                return 0.0
            if o_j < -a:
                return 1.0
            else:
                return self.k * self.q(x_t) * self.Z / (np.exp(o_j) + self.k * self.q(x_t) * self.Z)

    def bptt(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        dZ = 0.0

        T = len(layers)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        #print("\nself.Z : %.4f"%self.Z)
        for t in range(0, T):
            # 学習時のみNCEを用いるプログラムではdmulvの計算を書き換えるだけで良い。
            dmulv = np.zeros(self.word_dim)
            #pp = np.exp(layers[t].mulv[y[t]]) / self.Z
            pp = self.Z
            #print("\npp = %f"%pp)
            dmulv[y[t]] = -self.true_prob(y[t], layers[t].mulv[y[t]], 0) / pp
            dZ += self.true_prob(y[t], layers[t].mulv[y[t]], 0) / self.Z
            for i in range(self.k):
                #qlayer = RNN_NCE_Layer()
                qx = self.generate_from_q()
                dmulv[qx] += self.true_prob(qx, layers[t].mulv[qx], 1)
                dZ -= self.true_prob(qx, layers[t].mulv[qx], 1) / self.Z
                '''
                input_word = np.zeros(self.word_dim)
                input_word[qx] = 1
                ps = np.zeros(self.hidden_dim) if t==0 else layers[t-1].s
                qlayer.forward(input_word, ps, self.U, self.W, self.V, y[t])
                qo_y_t = qlayer.mulv_y_t
                dmulv[y[t]] -= self.true_prob(qx, qo_y_t) / (self.k * self.q(qx))
                '''
            #print("dZ : %.4f"%dZ)
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)
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
        return (dU, dW, dV, dZ)

    def sgd_step(self, data):
        x = data[0]
        y = data[1]
        learning_rate = data[2]
        dU, dW, dV, dZ = self.bptt(x, y)
        #print(os.getpid())
        #self.U -= learning_rate * dU
        #self.V -= learning_rate * dV
        #self.W -= learning_rate * dW
        return np.array([dU,dW,dV,dZ])

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5, batch_size=1):
        num_examples_seen = 0
        losses = []
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
                    dU,dW,dV,dZ = self.sgd_step((X[i],Y[i],learning_rate))
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                    self.Z -= learning_rate * dZ
                # minibatch learning
                else:
                    data_list = []
                    for j in range(batch_size):
                        index = i * batch_size + j
                        data_list.append([X[index],Y[index],learning_rate])
                    pool = mp.Pool(batch_size)
                    args = zip(itr.repeat(self),itr.repeat('sgd_step'),data_list)
                    dU,dW,dV,dZ = np.sum(np.array(pool.map(utils.tomap,args)),axis=0)
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                    self.Z -= learning_rate * dZ
                    pool.close()
                #loss = self.calculate_total_loss(X, Y)
                #print("\nloss : %.4f"%loss)
                #print("\nself.Z = %.4f"%self.Z)

                    
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
                print("Perplexity : %.2f"%2.0 ** loss)

        return losses
