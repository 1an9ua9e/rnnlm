from activation import Tanh
from gate import AddGate, MultiplyGate

mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)

    
class ClassRNNLayer:
    def __init__(self, word_list):
        self.word_list = word_list
        
    
    def forward(self, x ,prev_s, U, W, V, Q):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        #self.mulv = mulGate.forward(V, self.s)
        self.mulq = mulGate.forward(Q, self.s)
        class_index = np.argmax(self.mulq)
        #クラス分布で最大のクラスに属する単語のみを順伝搬させる
        self.mulv = mulGate.sub_forward(V, self.s, self.word_list[class_index])
        

    def backward(self, x, prev_s, U, W, V, Q, diff_s, dmulv, dmulq):
        self.forward(x, prev_s, U, W, V, Q)
        # 以下のコードは部分的にbackwardする処理になっているか？
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        dQ, dsq = mulGate.backward(Q, self.s, dmulq)
        ds = dsv + dsq + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV, dQ)
            
        
