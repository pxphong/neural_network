import numpy as np
from hidden_layer import HiddenLayer

__all__ = ['NeuralNetwork']


class NeuralNetwork:

    def __init__(self, learning_rate, num_classes=2, reg=1e-5):
        self.layers = []
        self.reg = reg
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.m = 0

    def add_layers(self, num_neurons, activation):
        """
        activation: a string is name of a activation function
        num_neurons: number neurons in each layer
        """
        assert activation in ['sigmoid', 'relu', 'tanh', 'softmax'], "activation must in ['sigmoid', 'relu', 'tanh', 'softmax']"
        self.layers.append(HiddenLayer(num_neurons, activation, self.reg))

    def forward(self, X):
        """
        compute forward in all network
        return list all_X contain all A
        """

        all_X = [X]
        for layer in self.layers:
            all_X.append(layer.forward(all_X[-1]))
        return all_X
 

    def compute_loss(self, Y, Y_hat):
        '''compute loss function with L2
        '''
        self.m = Y.shape[0]
        loss = -np.sum(Y * np.log(Y_hat)) / self.m      # with softmax in last layer
        reglu = 0
        for i in range(len(self.layers)):
            reglu +=  np.sum(self.layers[i].W ** 2)
        loss += self.reg * reglu / (2 * self.m)
        return loss


    def compute_delta_grad_last(self, Y, all_X):
        #compute dZ[L] and dW[L]

        delta_last = (all_X[-1] - Y) / self.m       # dZ[L] with softmax
        grad_last = all_X[-2].T @ delta_last + self.reg * self.layers[-1].W       # dW[L]

        return delta_last, grad_last


    def backward(self, Y, all_X):
        """
        compute backward
        
        return list grad_list contain all "dW" in network 
        """

        delta_prev, grad_last = self.compute_delta_grad_last(Y, all_X)        # dZ[L], dW[L]
        grad_list = [grad_last]     # [dW[L]]

        for i in reversed(range(len(self.layers) - 1)):       # i = L-2 to 1
            prev_layers = self.layers[i + 1]
            layer = self.layers[i]
            X = all_X[i]

            delta_prev = delta_prev @ prev_layers.W.T                   # dA[L-1]
            grad_W, delta_prev = layer.backward(X, delta_prev)          # dW[L-1], dZ[L-1]

            grad_list.insert(0, grad_W)   

        return grad_list


    # def update_weight(self, grad_list):
    #     #update Weight
    #     for i, layer in enumerate(self.layers):
    #         grad = grad_list[i]
    #         layer.W -= self.learning_rate * grad
    

    def update_weight(self, grad_list, momentum_rate = 0.9):
    #     if not hasattr(self, "momentum"):
    #         self.momentum = [np.zeros_like(grad) for grad in grad_list]
        self.momentum = [np.zeros_like(grad) for grad in grad_list]
        for i, layer in enumerate(self.layers):
            self.momentum[i] = self.momentum[i] * momentum_rate + self.learning_rate * grad_list[i]
            layer.W = layer.W - self.momentum[i]


    def predict(self, X_test):
        #compute predict Y_hat
        Y_hat = self.forward(X_test)[-1]
        return np.argmax(Y_hat, axis=1)
