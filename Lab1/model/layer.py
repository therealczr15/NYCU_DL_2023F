import numpy as np
from numpy.random import randn

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output_grad):
        raise NotImplementedError
        
## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = randn(in_features, out_features) * 0.01
        self.bias = np.zeros((10, out_features))

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = np.dot(self.input.T, output_grad) 
        self.bias_grad =  output_grad 
        return input_grad

## by yourself .Finish your own NN framework
class ReLu(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.output = self.input
        self.output[self.output < 0] = 0
        return self.output

    def backward(self, output_grad):
        input_grad = self.output
        input_grad[input_grad > 0] = 1
        return np.multiply(output_grad, input_grad)

class SeLu(_Layer):
    def __init__(self):
        pass

    def forward(self, input, alpha = 1.673263242, labda = 1.050700987):
        self.input = input
        self.output = self.input
        self.output = np.where(self.output > 0, labda * self.output, labda * alpha * (np.exp(self.output) - 1))
        return self.output
        

    def backward(self, output_grad, alpha = 1.673263242, labda = 1.050700987):
        input_grad = self.output
        input_grad = np.where(input_grad <= 0, labda * alpha * np.exp(input_grad), labda)
        return np.multiply(output_grad, input_grad)
    
class Sigmoid(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_grad):
        input_grad = self.output * (1 - self.output)
        return np.multiply(output_grad, input_grad)

class Tanh(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.output = np.tanh(self.input)
        return self.output

    def backward(self, output_grad):
        input_grad = 1 - np.tanh(self.output) ** 2
        return np.multiply(output_grad, input_grad)

class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass

    def forward(self, input, target):
        '''Softmax'''
        exp = np.exp(input-np.max(input,axis=1,keepdims=True))
        self.predict = exp / np.sum(exp,axis=1,keepdims=True)
        self.target = target        
        '''Average loss'''
        your_loss = -np.average(np.sum(self.target * np.log(self.predict + 10**-100), axis = 1))
        return self.predict, your_loss

    def backward(self):
        input_grad = self.predict - self.target
        return input_grad

class Mse(_Layer):
    def __init__(self):
        pass

    def forward(self, input, target):
        self.predict = input
        self.target = target    
        your_loss = np.square(self.predict - self.target).sum()
        return self.predict, your_loss

    def backward(self):
        input_grad = 2 * (self.predict - self.target)
        return input_grad
    