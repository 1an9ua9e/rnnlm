import numpy as np

class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    # あるクラスに属する単語の索引集合lによって、
    # xのいくつかのノードを選択し、その部分だけで
    # 確率分布をつくる。それ以外の部分は０にする
    def rest_predict(self, x, l):
        exp_scores = np.zeros(len(x))
        sum = 0.0
        for w in l:
            exp_scores[w] = np.exp(x[w])
            sum += exp_scores[w]
        for w in l:
            exp_scores[w] /= sum
            
        return exp_scores
    
    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])
    
    def sub_loss(self, x, y, l, j):
        # リストlに含まれる単語からなるレイヤーをつくり、zとする
        # zは正解クラスjに対する出力層である
        # zのうち正解ラベルyに該当する単語idをindとし、
        # カテゴリカル交差エントロピーを求める
        if len(l) == 1:
            return 0.0
        
        z = np.zeros(len(l))
        for i, v in enumerate(l):
            z[i] = x[v]
        probs = self.predict(z)
        # word_listの各要素、つまりリストがnumpy配列かどうか確認しておいたほうがよい
        '''
        l2 = np.ndarray.tolist(l)
        ind = l2.index(y)
        '''
        ind = l.index(y)
        return -np.log(probs[ind])

    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs

    def sub_diff(self, x, y, l):
        probs = self.rest_predict(x, l)
        probs[y] -= 1.0
        return probs

class ClassSoftmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    # あるクラスに属する単語の索引集合lによって、
    # xのいくつかのノードを選択し、その部分だけで
    # 確率分布をつくる。それ以外の部分は０にする
    def rest_predict(self, x, l):
        exp_scores = np.zeros(len(x))
        sum = 0.0
        for w in l:
            exp_scores[w] = np.exp(x[w])
            sum += exp_scores[w]
        for w in l:
            exp_scores[w] /= sum
        return exp_scores

    def loss(self, x, c):
        probs = self.predict(x)
        return -np.sum(c * np.log(probs))

    def uni_loss(self, x, v_size, word_list, class_id):
        probs = self.predict(x)
        p = (1 - probs[class_id]) / (v_size - len(word_list))
        return -np.log(p)

    # 通常のsoftmax関数では、予測に対する正解データが1-hot vectorである
    # class softmax関数では、正解データ（この場合はクラスの出現分布）
    # の各要素をみてデルタを計算する
    #delt_j = sigma c_i * (c_hat_i - 1[i==j])
    def diff(self, x, y):
        probs = self.predict(x)
        delta = np.zeros(len(probs))
        for j in range(len(y)):
            sum = 0.0
            for i in range(len(y)):
                if j == i:
                    sum += y[i] * (probs[i] - 1)
                else:
                    sum += y[i] * probs[i]
            delta[j] = sum
            
        return delta
