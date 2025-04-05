import ActivationFunctions
import random
import json


class NeuralNetwork:
    def __init__(self) -> None:
        self.number_of_inputs = None
        self.number_of_outputs = None
        self.number_of_neurons_in_each_layer = None
        self.weights = None
        self.activation_function = None

    def setup_for_learning(self, number_of_inputs: int,
                           number_of_outputs: int,
                           number_of_neurons_in_each_hidden_layer: list[int]) -> None:
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_hidden_layer
        self.number_of_neurons_in_each_layer.append(number_of_outputs)
        self.weights = self._init_weights_for_all_layers()
        self.activation_function = ActivationFunctions.jump_function

    def process(self, inputs: list) -> list[float]:
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
                neuron_output = self.activation_function(u)
                output_from_layer.append(neuron_output)
            inputs = output_from_layer[:]
        return output_from_layer

    def learn(self, learning_set: list[list[float]], desired_outputs: list[int]):
        return

    def save_to_file(self, path_to_file: str):
        network_as_dict = {
            "number_of_inputs": self.number_of_inputs,
            "number_of_outputs": self.number_of_outputs,
            "number_of_neurons_in_each_layer": self.number_of_neurons_in_each_layer,
            "weights": self.weights,
        }
        if self.activation_function == ActivationFunctions.jump_function:
            network_as_dict["activation_function"] = "jump_function"
        elif self.activation_function == ActivationFunctions.sigmoid_function:
            network_as_dict["activation_function"] = "sigmoid_function"
        network_json = json.dumps(network_as_dict, indent=4)
        with open(path_to_file, 'w') as json_file:
            json_file.write(network_json)

    @staticmethod
    def read_from_file(path_to_file: str):
        with open(path_to_file, 'r') as file:
            data = json.load(file)
            network = NeuralNetwork()
            for key, value in data.items():
                if hasattr(network, key):
                    if key != "activation_function":
                        setattr(network, key, value)
                    else:
                        if value == "jump_function":
                            network.use_jump_activation_function()
                        elif value == "sigmoid_function":
                            network.use_sigmoid_activation_function()
            return network

    def use_jump_activation_function(self) -> None:
        self.activation_function = ActivationFunctions.jump_function

    def use_sigmoid_activation_function(self) -> None:
        self.activation_function = ActivationFunctions.sigmoid_function

    def _init_weights_for_all_layers(self) -> list[list[list[float]]]:
        weights_for_all_layers = []
        weights_for_layer_zero = self._init_weights_for_layer_zero()
        weights_for_all_layers.append(weights_for_layer_zero)
        if len(self.number_of_neurons_in_each_layer) > 1:
            weights_for_layers_different_than_zero = self._init_weights_for_layers_different_than_zero()
            for weights in weights_for_layers_different_than_zero:
                weights_for_all_layers.append(weights)
        return weights_for_all_layers

    def _init_weights_for_layer_zero(self) -> list[list[float]]:
        """Initialize weights for neurons that are in layer which is
        directly connected to inputs of the network

        Returns:
            list[list[float]: List (layer 0 of neural network) which contains lists,
            where each nested list contains weights for each neuron in layer 0
        """
        weights_for_layer_zero: list[list[float]] = []
        number_of_inputs_with_bias: int = self.number_of_inputs + 1
        for neuron_index in range(self.number_of_neurons_in_each_layer[0]):
            weights_for_neuron: list[float] = []
            for input_index in range(number_of_inputs_with_bias):
                random_weight = self._generate_random_weight()
                weights_for_neuron.append(random_weight)
            weights_for_layer_zero.append(weights_for_neuron)
        return weights_for_layer_zero

    def _init_weights_for_layers_different_than_zero(self) -> list[list[list[float]]]:
        """Initialize weights for neurons that are in layers that are not directly
        connected to input layer.

        Returns:
            list[list[list[float]]]: List of layers, each layer contains neurons
            and that last list contains weights associated with each neuron
        """
        layers: list[list[list[float]]] = []
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        starting_layer_index: int = 1
        for layer_index in range(starting_layer_index, number_of_layers):
            weights_for_layer: list[list[float]] = []
            number_of_neurons_in_layer: int = self.number_of_neurons_in_each_layer[layer_index]
            for neuron_index in range(number_of_neurons_in_layer):
                weights_for_neuron: list[float] = []
                number_of_outputs_from_previous_layer_with_bias: int = (
                        self.number_of_neurons_in_each_layer[layer_index - 1] + 1)
                for neuron_input_index in range(number_of_outputs_from_previous_layer_with_bias):
                    random_weight = self._generate_random_weight()
                    weights_for_neuron.append(random_weight)
                weights_for_layer.append(weights_for_neuron)
            layers.append(weights_for_layer)
        return layers

    @staticmethod
    def _generate_random_weight() -> float:
        random_weight = random.random()
        return random_weight if random.randint(0, 1) == 0 else random_weight * (-1)

    def __str__(self):
        network_as_text = ""
        layer_index = 0
        for layer in self.weights:
            network_as_text += f"l[{layer_index}]\n"
            neuron_index = 0
            for neuron in layer:
                input_index = 0
                for weight in neuron:
                    network_as_text += f"w[{neuron_index}][{input_index}] = {weight}\t"
                    input_index = input_index + 1
                network_as_text += "\n"
                neuron_index = neuron_index + 1
            layer_index = layer_index + 1
            network_as_text += "\n"
        return network_as_text