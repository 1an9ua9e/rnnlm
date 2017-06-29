import numpy as np

class Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
        '''
        if type(x) == float:
            if x < -600.0:
                x = -500.0
            return 1.0 / (1.0 + np.exp(-x))
        else:
            if min(x) < -600.0:
                indices = np.where(x < -600.0)
                for i in indices:
                    x[i] = -500.0
            return 1.0 / (1.0 + np.exp(-x))
        '''
    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - output) * output * top_diff

class Tanh:
    def forward(self, x):
        '''
        if type(x) == float:
            if x > 100.0:
                return 1.0
            elif x < -100.0:
                return -1.0
        else:
            if max(x)
        '''     
        return np.tanh(x)

    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff

class ReLU:
    def forward(self, x):
        return x
    
    def backward(self, x):
        return x
        
class Inverse:
    def forward(self, x):
        return np.ones(len(x)) - x
    
    def backward(self, top_diff):
        return - top_diff
