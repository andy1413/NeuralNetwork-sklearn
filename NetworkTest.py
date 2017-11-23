import  mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 0.5, test_data)