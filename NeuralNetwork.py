import random


def jump_function(u : float) -> float:
    return 1 if u >= 0 else 0

class NeuralNetwork:
    def __init__(self, number_of_inputs: int, number_of_neurons_in_each_layer: list[int]) -> None:
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_layer
        self.weights = self._init_weights_for_all_layers()
        self.activation_function = jump_function

    def process(self, inputs: list):
        output_from_layer = []
        for layer_index in range(len(self.weights)):
            output_from_layer = []
            one_in_input_for_bias_weight = 1
            bias_weight_index = 0
            inputs.insert(bias_weight_index, one_in_input_for_bias_weight)
            for neuron_index in range(len(self.weights[layer_index])):
                u = 0
                for input_index in range(len(self.weights[layer_index][neuron_index])):
                    u = u + inputs[input_index] * self.weights[layer_index][neuron_index][input_index]
                neuron_output = jump_function(u)
                output_from_layer.append(neuron_output)
            inputs = output_from_layer[:]
        return output_from_layer

    def use_jump_function(self):
        self.activation_function = jump_function

    def _init_weights_for_all_layers(self):
        weights_for_all_layers = []
        weights_in_layer_zero = self._init_weights_for_layer_zero()
        weights_for_all_layers.append(weights_in_layer_zero)
        weights_for_deeper_layers = self._init_weights_for_deeper_layers()
        for weights in weights_for_deeper_layers:
            weights_for_all_layers.append(weights)
        return weights_for_all_layers

    def _init_weights_for_layer_zero(self):
        weights_in_layer_zero = []
        for i in range(self.number_of_neurons_in_each_layer[0]):
            weights_for_neuron_i_in_layer_zero = []
            for j in range(self.number_of_inputs+1):
                random_weight = self._generate_random_weight()
                weights_for_neuron_i_in_layer_zero.append(random_weight)
            weights_in_layer_zero.append(weights_for_neuron_i_in_layer_zero)
        return weights_in_layer_zero

    def _init_weights_for_deeper_layers(self):
        if len(self.number_of_neurons_in_each_layer) < 1:
            return
        weights_for_deeper_layers = []
        for layer_index in range(1, len(self.number_of_neurons_in_each_layer)):
            weights_for_layer = []
            for neuron_index in range(self.number_of_neurons_in_each_layer[layer_index]):
                weights_for_neuron = []
                for neuron_input_index in range(self.number_of_neurons_in_each_layer[layer_index-1]+1):
                    random_weight = self._generate_random_weight()
                    weights_for_neuron.append(random_weight)
                weights_for_layer.append(weights_for_neuron)
            weights_for_deeper_layers.append(weights_for_layer)
        return weights_for_deeper_layers

    @staticmethod
    def _generate_random_weight() -> float:
        random_weight = random.random()
        return random_weight if random.randint(0, 1) == 0 else random_weight * (-1)

    def __str__(self):
        network_as_text = ""
        layer_index = 0
        for layer in self.weights:
            network_as_text += f"layer[{layer_index}]\n"
            neuron_index = 0
            for neuron in layer:
                input_index = 0
                for weight in neuron:
                    network_as_text += f"w[{neuron_index}][{input_index}] = {weight}\t"
                    input_index = input_index + 1
                neuron_index = neuron_index + 1
            layer_index = layer_index + 1
            network_as_text += "\n"
        return network_as_text