import numpy as np
from activation_function import sigmoid, sigmoid_grad, relu, relu_grad, softmax

__all__ = ['HiddenLayer']

class HiddenLayer:

    def __init__(self, num_neurons, activation, reg = 0):
        assert activation in ["sigmoid", "relu", "tanh", "softmax"], "Activation must be in [sigmoid, relu, tanh, softmax]"

        self.num_neurons = num_neurons
        self.W = None 
        self.activation = activation
        self.reg = reg


    def forward(self, X):
        #compute A in each layer

        if self.W is None:
            W_shape = (X.shape[1], self.num_neurons)
            self.W = np.random.normal(loc=0, scale=np.sqrt(2/(X.shape[1]+self.num_neurons)), size=W_shape)
        
        activation_function = eval(self.activation)

        self.Z = np.dot(X, self.W)
        A = activation_function(self.Z)

        return A


    def backward(self, X, delta_prev):
        """
        compute dZ, dW in each layer

        delta_prev: dA[l]

        return dZ[l], dW[l]
        """

        activation_grad_function = eval(self.activation + "_grad")
        z = self.Z

        delta = delta_prev * activation_grad_function(z)        # dZ[l]
        W_grad = X.T.dot(delta)

        W_grad += self.reg*self.W/X.shape[0]                    # dW[l]
        
        return W_grad, delta