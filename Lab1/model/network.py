from .layer import *

class Network(object):
    def __init__(self):
        self.fc1 = FullyConnected(28*28,256)
        self.act1 = ReLu()
        self.fc2 = FullyConnected(256,128)
        self.act2 = ReLu()
        self.fc3 = FullyConnected(128,32)
        self.act3 = ReLu()
        self.fc4 = FullyConnected(32,10)
        self.loss = SoftmaxWithloss()

    def forward(self, input, target):
        h1 = self.fc1.forward(input)
        a1 = self.act1.forward(h1)
        h2 = self.fc2.forward(a1)
        a2 = self.act2.forward(h2)
        h3 = self.fc3.forward(a2)   
        a3 = self.act3.forward(h3)
        h4 = self.fc4.forward(a3) 
        pred, loss = self.loss.forward(h4,target)
        return pred, loss

    def backward(self):
        h4_grad = self.loss.backward()
        a3_grad = self.fc4.backward(h4_grad)
        h3_grad = self.act3.backward(a3_grad)
        a2_grad = self.fc3.backward(h3_grad)
        h2_grad = self.act2.backward(a2_grad)
        a1_grad = self.fc2.backward(h2_grad)
        h1_grad = self.act1.backward(a1_grad)
        _ = self.fc1.backward(h1_grad)

    def update(self, lr):
        ## by yourself .Finish your own NN framework
        
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad
        self.fc2.weight -= lr * self.fc2.weight_grad
        self.fc2.bias -= lr * self.fc2.bias_grad
        self.fc3.weight -= lr * self.fc3.weight_grad
        self.fc3.bias -= lr * self.fc3.bias_grad
        self.fc4.weight -= lr * self.fc4.weight_grad
        self.fc4.bias -= lr * self.fc4.bias_grad
        
