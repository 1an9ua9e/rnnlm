import numpy as np

class MultiplyGate:
    def forward(self,W, x):
        return np.dot(W, x)
    
    #クラスベース言語モデル用に、部分的な順伝搬計算を行う。
    def sub_forward(self, W, x, word_list):
        # W*xというベクトルと同じ次元のゼロベクトルをつくる
        b = np.zeros(W.size[0])
        # 最大クラスの単語のみを順伝搬する
        for i in word_list:
            b[i] += np.dot(W[i], x)
        return b

    def backward(self, W, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx

    def sub_backward():
        return a

class AddGate:
    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, x1, x2, dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2
