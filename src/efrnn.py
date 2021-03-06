# -*- coding:utf-8 -*-
from datetime import datetime
import time
import os
import math
import numpy as np
import sys
from layer import RNNLayer
from layer import ClassRNNLayer
from output import Softmax
from output import ClassSoftmax
import multiprocessing as mp
import itertools as itr
import utils

class EFRNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4,class_dim=0,index_to_class=[],class_to_word_list=[], alpha=1.0, interval=1, class_change_interval=1):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.class_dim = class_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.Q = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (class_dim, hidden_dim))

        #self.class_dist = np.array(index_to_class_dist)
        self.word2class = np.array(index_to_class)
        self.word_list = np.array(class_to_word_list)
        
        # 評価関数にかける係数
        self.alpha = alpha
        # データ(interval)個につき１回クラスター指標を考慮したbackwardを行う
        self.interval = interval
        # 現時点でのクラス割り当てから計算したセントロイドに基づき、クラス割り当てを変更する操作の間隔
        self.class_change_interval = class_change_interval
    '''
        forward propagation (predicting word probabilities)
        x is one single data, and a batch of data
        for example x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    '''
    def forward_propagation(self, x, y=[]):
        # The total number of time steps
        T = len(x)
        layers = []
        prev_s = np.zeros(self.hidden_dim)
        # For each time step...
        for t in range(T):
            layer = ClassRNNLayer()
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            
            if y==[]:
                # 学習やテストに関係ない順伝搬計算の場合
                layer.forward(input, prev_s, self.U, self.W, self.V, self.Q)
            else:
                # time step tにおける望ましい出力y[t]と同じクラスに属する単語
                # について、順伝搬の計算を行う必要がある。したがって
                # forward計算に教師データを渡さなくてはならない。
                #layer.forward(input, prev_s, self.U, self.W, self.V, self.Q, self.class_dist[y[t]])
                layer.forward(input, prev_s, self.U, self.W, self.V, self.Q, self.word_list[self.word2class[y[t]]], y[t])
            prev_s = layer.s
            layers.append(layer)
        return layers


    def predict(self, x):
        output = Softmax()
        layers = self.forward_propagation(x)

        # クラスの予測分布を各ステップごとに作り、最大のものを予測クラスとする
        class_pred = [np.argmax(output.predict(layer.mulq)) for layer in layers]

        # あるクラスに対し、クラスに属する単語をリストで返す
        index_lists = [class_to_index_list[c] for c in class_pred]

        layers_mulv = []
        for i in range(len(layers)):
            layer_mulv= []
            for j in range(len(index_lists[i])):
                layer_mulv.append(layers[i].mulv[index_list[i][j]])
            layers_mulv.append(layer_mulv)
            
        return [np.argmax(output.predict(layer_mulv)) for layer_mulv in layers_mulv]

    def calculate_loss(self, x, y, eos):
        assert len(x) == len(y)
        output = Softmax()
        class_output = ClassSoftmax()
        layers = self.forward_propagation(x, y)
        loss = 0.0
        for t, layer in enumerate(layers):
            if eos and t == len(y) - 1:
                return loss / float(len(y) - 1)
            '''
            if t == len(y) - 1:            
                loss += output.loss(layers[t].mulv, y[t])
            '''
            # 単語の出力の損失だけを計算する場合
            # update_loss = - log{ p(w_j|c(w_j),s) * p(c(w_j)|s) }
            '''
            c_j = np.argmax(self.class_dist[y[t]])
            c_j = self.word2class[y[t]]
            if y[t] in self.word_list[c_j]:
                loss += output.loss(layer.mulq, c_j)
                loss += output.sub_loss(layer.mulv, y[t], self.word_list[c_j], c_j)
            else:
                loss += class_output.uni_loss(layer.mulq, self.word_dim, self.word_list[c_j], c_j)
            '''
            c_t = self.word2class[y[t]]
            loss += class_output.loss(layers[t].mulq, layers[t].mulv, c_t, y[t], self.word_list[c_t])
            # 単語とクラス、両方の出力層で損失を計算する場合
            '''
            loss += output.loss(layer.mulv, y[i])
            loss += class_output.loss(layer.mulq, self.class_dist[y[i]])
            '''
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y, eos=False):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i], eos)
        return loss / float(len(Y))
    
    def d(self, a, b):
        return math.sqrt(np.dot(a-b, a-b))
    
    def calculate_centroids(self):
        centroids = []
        for c in range(self.class_dim):
            class_size = len(self.word_list[c])
            centroid = np.zeros(self.hidden_dim)
            for w in self.word_list[c]:
                centroid += self.V[w]
            centroids.append(centroid / class_size)
        return centroids
        
    def calculate_ef_loss(self):
        # 各クラスのセントロイドを計算する
        centroids = self.calculate_centroids()
        
        # 現時点での評価関数の値を計算する
        S = 0.0
        for c in range(self.class_dim):
            for w in self.word_list[c]:
                S += self.d(self.V[w], centroids[c])
        return S
                
    def calculate_word2class(self, centroids):
        word2class = [0] * self.word_dim
        for w in range(self.word_dim):
            min = 1000000000.0
            c = self.word2class[w]
            for i,r in enumerate(centroids):
                d = self.d(self.V[w], r)
                if d < min:
                    min = d
                    c = i
            word2class[w] = c
        return word2class
        
    def bptt(self, x, y, data_id):
        if data_id % self.class_change_interval == 0:
            centroids = self.calculate_centroids()
            self.word2class = self.calculate_word2class(centroids)
        assert len(x) == len(y)
        output = Softmax()
        class_output = ClassSoftmax()
        layers = self.forward_propagation(x, y)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        dQ = np.zeros(self.Q.shape)

        T = len(layers)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        for t in range(0, T):
            # dmulvの計算はこれでいいのか
            # 該当するノードのみ誤差を伝搬させる必要がある
            
            #dmulv = output.diff(layers[t].mulv, y[t])
            #word_list = self.word_list[np.argmax(self.class_dist[y[t]])] # y[t]と同じクラスの単語リスト
            
            class_y_t = self.word2class[y[t]]
            word_list = self.word_list[class_y_t]
            dmulv = output.hard_class_diff(layers[t].mulv, y[t], word_list)
            dmulq = class_output.diff(layers[t].mulq, class_y_t)
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            
            # データ50個に１個のペースで、単語ベクトルのクラスター指標を考慮したbackwardを行う
            if data_id%self.interval == 0:
                dprev_s, dU_t, dW_t, dV_t, dQ_t = layers[t].backward(
                    input, prev_s_t, self.U, self.W, self.V, self.Q, diff_s, dmulv, dmulq,
                    word_list, y[t], class_y_t, self.calculate_centroids(), self.word2class, self.alpha)
            else:
                dprev_s, dU_t, dW_t, dV_t, dQ_t = layers[t].backward(
                    input, prev_s_t, self.U, self.W, self.V, self.Q, diff_s, dmulv, dmulq, [], -1, -1)
                
            prev_s_t = layers[t].s
            dmulv = np.zeros(self.word_dim)
            dmulq = np.zeros(self.class_dim)
            for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                input = np.zeros(self.word_dim)
                input[x[i]] = 1
                prev_s_i = np.zeros(self.hidden_dim) if i == 0 else layers[i-1].s
                dprev_s, dU_i, dW_i, dV_i, dQ_i = layers[i].backward(input, prev_s_i, self.U, self.W, self.V, self.Q, dprev_s, dmulv, dmulq, [], -1, -1)
                dU_t += dU_i
                dW_t += dW_i
            dV += dV_t
            dU += dU_t
            dW += dW_t
            dQ += dQ_t
        return (dU, dW, dV, dQ)

    def sgd_step(self, data):
        x = data[0]
        y = data[1]
        learning_rate = data[2]
        data_id = data[3]
        dU, dW, dV, dQ = self.bptt(x, y, data_id)
        #print(os.getpid())
        #self.U -= learning_rate * dU
        #self.V -= learning_rate * dV
        #self.W -= learning_rate * dW
        return np.array([dU,dW,dV,dQ])
    
    def test(self, X, Y, eos):
        loss = self.calculate_total_loss(X, Y, eos)
        print("\nTest Perplexity : %.2f" % 2.0**loss)
        return loss
                        
    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5,batch_size=1, X_test=[], Y_test=[], eos=False):
        num_examples_seen = 0
        losses = []
        test_losses = []
        for epoch in range(nepoch):
            # For each training example...
            num_examples_seen = 0
            data_size = len(Y)
            max_batch_loop = math.floor(data_size / batch_size)
            number = [i for i in range(max_batch_loop)] # データの処理の順番
            start = time.time()
                            
            if(batch_size == 1):
                print("training mode : online learning")
            else:
                print("training mode : minibatch learning (batch size %d)"%batch_size)
            my_count = 0
            for i in range(max_batch_loop):
                my_count += 1
                
                # online learning
                num_examples_seen += batch_size
                sys.stdout.write("\r%s / %s"%(num_examples_seen,data_size))
                sys.stdout.flush()

                if batch_size <= 1:
                    dU,dW,dV,dQ = self.sgd_step((X[i],Y[i],learning_rate))
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                    self.Q -= learning_rate * dQ
                    if my_count % 10 == 0:
                        print("loss : %f"%self.calculate_total_loss(X,Y,eos))
                # minibatch learning
                else:
                    data_list = []
                    for j in range(batch_size):
                        index = number[i] * batch_size + j
                        data_list.append([X[index],Y[index],learning_rate, index])
                    pool = mp.Pool(batch_size)
                    args = zip(itr.repeat(self),itr.repeat('sgd_step'),data_list)
                    dU,dW,dV,dQ = np.sum(np.array(pool.map(utils.tomap,args)),axis=0)
                    self.U -= learning_rate * dU
                    self.W -= learning_rate * dW
                    self.V -= learning_rate * dV
                    self.Q -= learning_rate * dQ
                    pool.close()
                '''
                if (i+1)%20 == 0:
                    self.test(X_test, Y_test)
                    print("eval func : %.2f"%self.calculate_ef_loss())
                '''
                
            print("training time %d[s]"%(time.time() - start))
            #np.random.shuffle(number)
                        
            loss = self.calculate_total_loss(X, Y, eos)
            ppl = 2.0 ** loss
            losses.append(ppl)
            dtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (dtime, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            print("Evaluation function : %.2f"%self.calculate_ef_loss())
            print("Training Perplexity : %.2f"%ppl)
            if X_test != []:
                test_loss = self.test(X_test, Y_test, eos)
                test_ppl = 2.0 ** test_loss
                test_losses.append(test_ppl)
                                            
        return (losses,test_losses)
