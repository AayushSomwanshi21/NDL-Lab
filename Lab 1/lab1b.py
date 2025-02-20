import numpy as np


class Neuron:

    def __init__(self, num_neurons):
        self.weights = np.random.rand(num_neurons)
        self.bias = np.random.rand()

    def activate(self, input):

        weighted_sum = np.dot(input, self.weights) + self.bias
        return self.sigmoid(weighted_sum)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, input, output, learning_rate):

        actual_output = self.activate(input)

        error = output - actual_output

        self.weights += learning_rate*error*input
        self.bias += learning_rate*error


if __name__ == '__main__':

    num_neurons = 3
    iterations = 10000
    learning_rate = 0.1
    neuron = Neuron(num_neurons)

    x_train = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])
    n = len(x_train)

    y_train = np.array([0, 1, 1, 0])

    for _ in range(iterations):

        i = np.random.randint(n)
        input = x_train[i]
        output = y_train[i]

        neuron.train(input, output, learning_rate)

    test_data = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])

    for i, test in enumerate(test_data):

        output = neuron.activate(test)
        print(
            f'Input:{test} Desired Output:{y_train[i]} Actual Output:{round(output, 4)}')
