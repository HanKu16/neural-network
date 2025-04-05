from NeuralNetwork import NeuralNetwork
from TrainingData import TrainingData

number_of_inputs: int = 4
number_of_outputs: int = 6
number_of_neurons_in_each_layer = []
learning_coefficient = 0.1
network = NeuralNetwork()
network.setup_for_learning(number_of_inputs, number_of_outputs, number_of_neurons_in_each_layer, learning_coefficient)
network.use_sigmoid_activation_function()
training_data1 = TrainingData([1, 0, 0, 1], [0, 0, 1, 0, 0, 0])
training_data2 = TrainingData([1, 1, 0, 1], [0, 0, 0, 0, 1, 0])
training_data3 = TrainingData([0, 1, 0, 1], [0, 0, 0, 0, 0, 1])

for i in range(10000):
    if i % 3 == 0:
        print("1")
        network.learn(training_data1)
    elif i % 3 == 1:
        print("2")
        network.learn(training_data2)
    else:
        print("3")
        network.learn(training_data3)


