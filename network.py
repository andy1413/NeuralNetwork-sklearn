"""
network.py

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits many desirable features.



"""

# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """

        :param sizes: contains the number of neurons in the
        respective layers of the network. For example, if the list
        was [2,3,1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1. Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """Return th output of the network if "a" is input.
        see feedforward.png """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.
        :param training_data: is a list of tuples
        (x, y) representing the training inputs and the desired outputs.
        The other non-optional parameters are self-explanatory.
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param test_data: is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out. This is useful for
        tracking progress, but slows things down substantially.
        :return:
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        :param mini_batch: is a list of tuples "(x, y)"
        :param eta: is the learning rate.
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x."nabla_b"and
        "nabla_w" are layer-by-layer lists of numpy arrays,similar
        to "self.biases" and "self.weights".
        :param x:
        :param y:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] #list to store all the activations, layer by layer
        zs = [] #list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #transpose is T
        """
        Note that the variable l in the loop below is used a little
        differently to the notation in Chapter 2 of the book. Here,
        l = 1 means the last layer of neurons, l = 2 is the
        second-last layer, and so on. It's a renumvering of the
        scheme in the book, used here to take advantage of the fact
        that Python can use negative indices in lists.
        """
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """

        :param test_data:
        :return: Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] #self feedforward x  return to argmax
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """

        :param output_activations:
        :param y:
        :return: the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return (output_activations-y)

### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    :param z:
    :return:
    """
    return sigmoid(z)*(1-sigmoid(z))


"""
Epoch 0: 8456 / 10000
Epoch 1: 8813 / 10000
Epoch 2: 8989 / 10000
Epoch 3: 9046 / 10000
Epoch 4: 9127 / 10000
Epoch 5: 9160 / 10000
Epoch 6: 9193 / 10000
Epoch 7: 9228 / 10000
Epoch 8: 9269 / 10000
Epoch 9: 9278 / 10000
Epoch 10: 9302 / 10000
Epoch 11: 9333 / 10000
Epoch 12: 9335 / 10000
Epoch 13: 9345 / 10000
Epoch 14: 9346 / 10000
Epoch 15: 9359 / 10000
Epoch 16: 9368 / 10000
Epoch 17: 9373 / 10000
Epoch 18: 9371 / 10000
Epoch 19: 9359 / 10000
Epoch 20: 9375 / 10000
Epoch 21: 9386 / 10000
Epoch 22: 9397 / 10000
Epoch 23: 9410 / 10000
Epoch 24: 9388 / 10000
Epoch 25: 9410 / 10000
Epoch 26: 9416 / 10000
Epoch 27: 9416 / 10000
Epoch 28: 9438 / 10000
Epoch 29: 9424 / 10000
"""



































