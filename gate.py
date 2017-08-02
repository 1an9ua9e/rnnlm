import numpy as np

class MultiplyGate:
    def forward(self,W, x):
        return np.dot(W, x)
    
    #クラスベース言語モデル用に、部分的な順伝搬計算を行う。
    def sub_forward(self, W, x, word_list):
        # W*xというベクトルと同じ次元のゼロベクトルをつくる
        b = np.zeros(W.shape[0])
        # 最大クラスの単語のみを順伝搬する
        for i in word_list:
            b[i] += np.dot(W[int(i)], x)
        return b

    def nce_forward(self, W, x, y, word_list):
        # W*xというベクトルと同じ次元のゼロベクトルをつくる
        b = np.zeros(W.shape[0])
        # 最大クラスの単語のみを順伝搬する
        for i in word_list:
            b[i] += np.dot(W[i], x)
        if y >= 0:
            b[y] += np.dot(W[y], x)
        return b

    def backward(self, W, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx
    
    def sub_backward(self, W, x, dz, sample_list):
        l1 = len(x)
        l2 = len(dz)
        dW = np.asmatrix(np.zeros(l1 * l2).reshape(l2, l1))
        mat_t_dz = np.transpose(np.samatrix(dz))
        mat_x = np.asmatrix(x)
        for v in sample_list:
            dW[v] += np.dot(mat_t_dz[v, 0] * mat_x)
        dW = np.asarray(dW)

        dx = np.zeros(l1)
        tW = np.transpose(W)
        for v in sample_list:
            dx[v] += np.dot(tW[v], dz)
        
        return dW, dx

    def nce_backward(self, W, x, dz, y, sample_list):
        l1 = len(x)
        l2 = len(dz)
        dW = np.asmatrix(np.zeros(l1 * l2).reshape(l2, l1))
        mat_t_dz = np.transpose(np.asmatrix(dz))
        mat_x = np.asmatrix(x)
        for v in sample_list:
            dW[v] += np.dot(mat_t_dz[v, 0], mat_x)
        if y >= 0:
            dW[y] += np.dot(mat_t_dz[y, 0], mat_x)
        dW = np.asarray(dW)
        '''
        dx = np.zeros(l1)
        tW = np.transpose(W)
        for v in sample_list:
            dx[v] += np.dot(tW[v], dz)
        if y >= 0:
            dx[y] += np.dot(tW[y], dz)
        '''
        dx = np.dot(np.transpose(W),dz)
        return dW, dx
    
class AddGate:
    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, x1, x2, dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2
    
# hadamard productを計算するゲート
class HadamardGate:
    def forward(self, x1, x2):
        return x1 * x2

    def backward(self, x1, x2, dz):
        dx1 = dz * x2
        dx2 = dz * x1
        return dx1, dx2
