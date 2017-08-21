
'''
重点サンプリングの例
E_pi[g(x)]を近似する。
'''
import scipy.special as sp
import math
import numpy as np
import numpy.random as nrand

class ImportanceSampling:
    def __init__(self, sample_size, pi_type, q_type):
        self.sample_size = sample_size
        self.pi_type = pi_type # 
        self.q_type = q_type # importance density

    def t_dist(self, x, nyu, sigma, myu):
            return sp.gamma((nyu+1)/2) / (sp.gamma(nyu/2) *
                                          ((nyu * math.pi * sigma ** 2) ** (0.5)) *
                                          ((1 + ((x - myu) ** 2 ) / (nyu * sigma ** 2)) ** ((nyu + 1)/2)))
    def normal_dist(self, x, myu, sigma):
        return np.exp(- (x - myu) ** 2 / sigma ** 2) / math.sqrt(2 * math.pi * sigma ** 2)
    
    def g(self, x):
        return x
    
    def pi(self, x):
        if self.pi_type == "t":
            nyu = 3.0
            sigma = 1.0
            myu = 5.0
            return self.t_dist(x, nyu, sigma, myu)
        
        elif self.pi_type == "gamma":
            return 1.0
        
        else:
            return 1.0
    
    def q(self, x):
        if self.q_type == "normal":
            myu = 5.0
            sigma = 50.0
            return self.normal_dist(x, myu, sigma)
        else:
            return 1.0

    def sample_from_q(self):
        myu = 5.0
        sigma = 1.0
        return nrand.normal(myu, sigma)
        
    def generate_samples(self):
        samples = []
        for i in range(self.sample_size):
            v = self.sample_from_q()
            samples.append(v)
        return samples
            
    def approximate(self):
        samples = self.generate_samples()
        approximation = 0.0
        for x in samples:
            approximation += self.g(x) * self.pi(x) / self.q(x)
            
        return approximation / self.sample_size
    def set_sample_size(self, size):
        self.sample_size = size

if __name__=="__main__":
    pi = "t"
    q = "normal"
    v = 5.0
    #for i in range(1000):
    size = 100000
    ismodel = ImportanceSampling(size, pi, q)
    error = abs(v - ismodel.approximate())
    print("size %d error %.4f"%(size, error))
        
