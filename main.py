import random
from NeuralNetwork import NeuralNetwork

network = NeuralNetwork()
network.setup_for_learning(6, [4, 3, 5])
network.use_jump_activation_function()
print(network)
for i in range(100):
    a = random.randint(-50, 50)
    b = random.randint(-50, 50)
    c = random.randint(-50, 50)
    d = random.randint(-50, 50)
    e = random.randint(-50, 50)
    f = random.randint(-50, 50)
    output_from_network = network.process([a, b, c, d, e, f])
    print(output_from_network)
