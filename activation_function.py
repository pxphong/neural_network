import numpy as np

__all__ = ['sigmoid', 'sigmoid_grad', 'relu', 'relu_grad', 'tanh', 'tanh_grad', 'softmax']

def sigmoid(z):
    ''' sigmoid function. output = 1 / (1 + exp(-1)).
        :param z: input
    '''
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoid_grad(z):
    sg = sigmoid(z) * (1 - sigmoid(z))
    return sg


def relu(z):
    r = np.maximum(0, z)
    return r


def relu_grad(z):
    rg = relu(z)
    rg[rg > 0] = 1
    return rg

def tanh(z):
    t = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return t


def tanh_grad(z):
    tg = 1 - tanh(z) ** 2
    return tg


def softmax(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    s = exps / np.sum(exps, axis=1, keepdims=True)
    return s