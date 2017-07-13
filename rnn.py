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

class Model:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))

    '''
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    '''
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
            loss += output.loss(layer.mulv, y[i])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def bptt(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        T = len(layers)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        for t in range(0, T):
            dmulv = output.diff(layers[t].mulv, y[t])
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
        return (dU, dW, dV)

    def sgd_step(self, data):
        x = data[0]
        y = data[1]
        learning_rate = data[2]
        dU, dW, dV = self.bptt(x, y)
        #print(os.getpid())
        #self.U -= learning_rate * dU
        #self.V -= learning_rate * dV
        #self.W -= learning_rate * dW
        return np.array([dU,dW,dV])

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5,batch_size=1):
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
                    dU,dW,dV = self.sgd_step((X[i],Y[i],learning_rate))
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                # minibatch learning
                else:
                    data_list = []
                    for j in range(batch_size):
                        index = i * batch_size + j
                        data_list.append([X[index],Y[index],learning_rate])
                    pool = mp.Pool(batch_size)
                    args = zip(itr.repeat(self),itr.repeat('sgd_step'),data_list)
                    dU,dW,dV = np.sum(np.array(pool.map(utils.tomap,args)),axis=0)
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                    pool.close()
                    
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
                print("Perplexity : %.2f"%2**loss)

        return losses
