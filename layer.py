from activation import Tanh, Sigmoid, Inverse
from gate import AddGate, MultiplyGate, HadamardGate
import numpy as np
import sys

mulGate = MultiplyGate()
addGate = AddGate()
hadGate = HadamardGate()
tanh = Tanh()
sigmoid = Sigmoid()
inv = Inverse()

class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = tanh.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = tanh.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)
# 必要ないクラス？？？
class RNN_NCE_Layer:
    def forward(self, x, prev_s, U, W, V, forward_list=[]):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = tanh.forward(self.add)
        self.mulv = np.zeros(len(x))
        if forward_list:
            for i in forward_list:
                self.mulv[i] = mulGate.forward(V[i], self.s)
        else:
            self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv, forward_list=[]):
        self.forward(x, prev_s, U, W, V, forward_list)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = tanh.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)

class GRULayer:
    def forward(self, x, prev_s, U_z, W_z, U_r, W_r, U_u, W_u, V):
        self.mul_U_z = mulGate.forward(U_z, x)
        self.mul_W_z = mulGate.forward(W_z, prev_s)
        self.mul_U_r = mulGate.forward(U_r, x)
        self.mul_W_r = mulGate.forward(W_r, prev_s)

        self.add_z = addGate.forward(self.mul_U_z, self.mul_W_z)
        self.add_r = addGate.forward(self.mul_U_r, self.mul_W_r)

        self.z = sigmoid.forward(self.add_z)
        self.r = sigmoid.forward(self.add_r)
        self.inv_z = inv.forward(self.z)
        
        self.mul_U_u = mulGate.forward(U_u, x)
        self.had_r_s = hadGate.forward(self.r, prev_s)
        self.mul_W_u = mulGate.forward(W_u, self.had_r_s)

        self.add_u = addGate.forward(self.mul_U_u, self.mul_W_u)
        self.u = tanh.forward(self.add_u)
        
        self.prev_update = hadGate.forward(self.z, prev_s)
        self.new_update = hadGate.forward(self.inv_z, self.u)
        self.s = addGate.forward(self.prev_update, self.new_update)
        self.mul_V = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U_z, W_z, U_r, W_r, U_u, W_u, V, diff_s, dmul_V):
        self.forward(x, prev_s, U_z, W_z, U_r, W_r, U_u, W_u, V)
        dV, dsv = mulGate.backward(V, self.s, dmul_V)
        ds = dsv + diff_s
        dprev_update, dnew_update = addGate.backward(self.prev_update, self.new_update, ds)
        dz, dprev_s_ = hadGate.backward(self.z, prev_s, dprev_update)
        dprev_s = dprev_s_
        # dinv_zの値がぶっとぶ。 self.u か dnew_updateに問題あり？
        dinv_z, du = hadGate.backward(self.inv_z, self.u, dnew_update)
        dz += inv.backward(dinv_z)
        dadd_z = sigmoid.backward(self.z ,dz)
        dmul_U_z, dmul_W_z = addGate.backward(self.mul_U_z, self.mul_W_z, dadd_z)
        dU_z, dx_ = mulGate.backward(U_z, x, dmul_U_z)
        dx = dx_
        dW_z, dprev_s_ = mulGate.backward(W_z, prev_s, dmul_W_z)
        dprev_s += dprev_s_
        dadd_u = tanh.backward(self.add_u, du)
        dmul_U_u, dmul_W_u = addGate.backward(self.mul_U_u, self.mul_W_u, dadd_u)
        dU_u, dx_ = mulGate.backward(U_u, x, dmul_U_u)
        dx += dx_
        dW_u, dhad_r_s = mulGate.backward(W_u, self.had_r_s, dmul_W_u)
        dr, dprev_s_ = hadGate.backward(self.r, prev_s, dhad_r_s)
        dprev_s += dprev_s_
        dadd_r = sigmoid.backward(self.r, dr)
        dmul_U_r, dmul_W_r = addGate.backward(self.mul_U_r, self.mul_W_r, dadd_r)
        dU_r, dx_ = mulGate.backward(U_r, x, dmul_U_r)
        dx += dx_
        dW_r, dprev_s_ = mulGate.backward(W_r, prev_s, dmul_W_r)
        dprev_s += dprev_s_
        return (dprev_s, dU_z, dW_z, dU_r, dW_r, dU_u, dW_u, dV)

    
class ClassRNNLayer:
    def __init__(self, word_list):
        self.word_list = word_list            
    def forward(self, x ,prev_s, U, W, V, Q, dist=[]):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        #self.mulv = mulGate.forward(V, self.s)
        self.mulq = mulGate.forward(Q, self.s)

        # 学習やテスト時の順伝搬計算では望ましい出力y[t]のクラス分布distから
        # y[t]のクラスを選び出し、そのクラス内の単語に関してforward計算を行う
        class_index = np.argmax(self.mulq) if dist==[] else np.argmax(dist)

        # クラス分布で最大のクラスに属する単語のみを順伝搬させる
        self.mulv = mulGate.sub_forward(V, self.s, self.word_list[class_index])
            

    def backward(self, x, prev_s, U, W, V, Q, diff_s, dmulv, dmulq, dist=[]):
        self.forward(x, prev_s, U, W, V, Q, dist)
        # 以下のコードは部分的にbackwardする処理になっているか？
        # dV, dsv = mulGate.sub_backward(V, self.s, dmulv)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        dQ, dsq = mulGate.backward(Q, self.s, dmulq)
        ds = dsv + dsq + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV, dQ)
            
        
