from NeuralNetwork import NeuralNetwork

network = NeuralNetwork(2, [4, 1])
print(network)
output_from_network = network.process([-3, 2])
print(output_from_network)