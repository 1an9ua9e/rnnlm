import numpy as np
import cython
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE_t
ctypedef np.int_t INT_t

cdef class CMultiplyGate:
    cdef np.ndarray b0
    cdef np.ndarray dW0
    
    cdef nce_forward(self, W, x, y, word_list):
        # W*xというベクトルと同じ次元のゼロベクトルをつくる
        b0 = np.zeros(W.shape[0])
        cdef np.ndarray[DOUBLE_t, ndim=1] b
        b = b0
	
        # 最大クラスの単語のみを順伝搬する
        for i in word_list:
            b0[i] += np.dot(W[i], x)
        if y >= 0:
            b0[y] += np.dot(W[y], x)
        return b0
'''
    def nce_backward(self, W, x, dz, y, sample_list):
        l1 = len(x)
        l2 = len(dz)

	dW0 = np.zeros(np.zeros(l1 * l2).reshape(l2, l1))
	
        dW = np.asmatrix(np.zeros(l1 * l2).reshape(l2, l1))
        mat_t_dz = np.transpose(np.asmatrix(dz))
        mat_x = np.asmatrix(x)
        for v in sample_list:
            dW[v] += np.dot(mat_t_dz[v, 0], mat_x)
        if y >= 0:
            dW[y] += np.dot(mat_t_dz[y, 0], mat_x)
        dW = np.asarray(dW)
        dx = np.dot(np.transpose(W),dz)
        return dW, dx
'''