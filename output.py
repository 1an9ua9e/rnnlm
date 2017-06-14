import numpy as np

class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])

    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs

class ClassSoftmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])

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
