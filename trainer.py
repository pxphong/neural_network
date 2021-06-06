from neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(losses, title, color='b'):
    plt.plot(losses, color)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title(title)


def batch_train(X_train, Y_train, epochs, neural_network, use_batch=False):
    all_loss = []
    display_step = 100 if use_batch else 10

    for e in range(epochs):
        all_X = neural_network.forward(X_train)
        loss = neural_network.compute_loss(Y_train, all_X[-1])
        grad_list = neural_network.backward(Y_train, all_X)
        neural_network.update_weight(grad_list)
        
        all_loss.append(loss)

        # if (e + 1) % display_step == 0:
        #     if use_batch:
        #         y_hat = neural_network.forward(X_train)[-1]

        #         # visualize_point(X_train, np.argmax(Y_train, axis=1), y_hat)

        #     plot_loss(all_loss, title="Loss epoch %s: %.4f" % (e + 1, loss), color='r')
        #     plt.show()
        #     plt.pause(0.01)
        print("Loss epoch %s: %.4f" % (e, all_loss[-1]/(epochs)))


def minibatch_train(X_train, Y_train, epochs, batch_size, num_classes, neural_network, shuffle=True):
    """
    Using batch train.
    
    Parameters
    ----------
    X_train: training data X.
    Y_train: training data Y.
    epochs: number of iterations that we should use to train.
    batch_size: number of batch at each update.
    neural_network: NeuralNetwork object instance above.
    
    """
    data = np.concatenate((X_train, Y_train), axis=1)
    n = data.shape[0]
    for e in range(epochs):
        if shuffle:
            np.random.shuffle(data)

        X_train, Y_train = data[:, :-num_classes], data[:, -num_classes:]
        num_batches = n // batch_size if n % batch_size == 0 else n // batch_size + 1

        all_loss = 0.0
        for b in range(num_batches):
            all_X = neural_network.forward(X_train[b:b+batch_size])
            loss = neural_network.compute_loss(Y_train[b:b+batch_size], all_X[-1])
            grad_list = neural_network.backward(Y_train[b:b+batch_size], all_X)
            neural_network.update_weight(grad_list)
            all_loss += loss

        print("Loss epoch %s: %.4f" % (e, all_loss/num_batches))
