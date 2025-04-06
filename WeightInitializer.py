import math
from random import uniform, random, randint

class WeightInitializer:
    def __init__(self, number_of_inputs: int, number_of_neurons_in_each_layer: list[int]):
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_layer

    def xavier_init(self) -> list[list[list[float]]]:
        """Xavier initialization for weights in all layers. Works well with sigmoid
        activation function. To get specific value from returned value refer by
        [index_layer][neuron_layer][input_layer].

        Returns:
            list[list[list[float]]]: List of layers, each layer contains neurons
            and each neuron contains weights.
        """
        weights_for_all_layers = []
        weights_for_layer_zero = self._xavier_init_for_layer_zero()
        weights_for_all_layers.append(weights_for_layer_zero)
        weights_for_not_input_layers = self._xavier_init_weights_for_not_input_layers()
        weights_for_all_layers.extend(weights_for_not_input_layers)
        return weights_for_all_layers

    def random_init(self) -> list[list[list[float]]]:
        """Random initialization for weights associated with neurons
        that are in layer which is directly connected to inputs of
        the network. Initialize weights be assigning to each weight
        random value between <-1, 1>.

        Returns:
            list[list[list[float]]]: List of layers, each layer contains neurons
            and each neuron contains weights.
        """
        weights_for_all_layers = []
        weights_for_layer_zero = self._random_init_for_layer_zero()
        weights_for_all_layers.append(weights_for_layer_zero)
        weights_for_not_input_layers = self._random_init_weights_for_not_input_layers()
        weights_for_all_layers.extend(weights_for_not_input_layers)
        return weights_for_all_layers

    def _xavier_init_for_layer_zero(self) -> list[list[float]]:
        """Xavier initialization for weights associated with neurons that
        are in layer which is directly connected to inputs of the network.

        Returns:
            list[list[float]]: Layer 0 of neural network which contains lists,
            where each nested list contains weights for each neuron in layer 0.
        """
        f_in: int = self.number_of_inputs
        f_out: int = self.number_of_neurons_in_each_layer[0]
        limit: float = self._xavier_calculate_limit(f_in, f_out)
        weights_for_layer_zero: list[list[float]] = []
        number_of_inputs_with_bias: int = self.number_of_inputs + 1
        for neuron_index in range(self.number_of_neurons_in_each_layer[0]):
            weights_for_neuron: list[float] = []
            for input_index in range(number_of_inputs_with_bias):
                random_weight = self._xavier_generate_weight(limit)
                weights_for_neuron.append(random_weight)
            weights_for_layer_zero.append(weights_for_neuron)
        return weights_for_layer_zero

    def _xavier_init_weights_for_not_input_layers(self) -> list[list[list[float]]]:
        """Xavier initialization for weights associated with neurons that are in
        layers that are not directly connected to input layer.

        Returns:
            list[list[list[float]]]: Layers that are not directly connected with
            actual inputs of the network.
        """
        not_input_layers: list[list[list[float]]] = []
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        starting_layer_index: int = 1
        for layer_index in range(starting_layer_index, number_of_layers):
            f_in = self.number_of_neurons_in_each_layer[layer_index-1]
            f_out = self.number_of_neurons_in_each_layer[layer_index]
            limit = self._xavier_calculate_limit(f_in, f_out)
            weights_for_layer: list[list[float]] = []
            number_of_neurons_in_layer: int = self.number_of_neurons_in_each_layer[layer_index]
            for neuron_index in range(number_of_neurons_in_layer):
                weights_for_neuron: list[float] = []
                number_of_inputs_for_neuron_plus_bias: int = (
                        self.number_of_neurons_in_each_layer[layer_index-1] + 1)
                for input_index in range(number_of_inputs_for_neuron_plus_bias):
                    random_weight: float = self._xavier_generate_weight(limit)
                    weights_for_neuron.append(random_weight)
                weights_for_layer.append(weights_for_neuron)
            not_input_layers.append(weights_for_layer)
        return not_input_layers

    @staticmethod
    def _xavier_calculate_limit(f_in: int, f_out: int):
        return math.sqrt(6 / (f_in + f_out))

    @staticmethod
    def _xavier_generate_weight(limit: float):
        return uniform(-limit, limit)

    def _random_init_for_layer_zero(self) -> list[list[float]]:
        """Random initialization for weights associated with neurons that
        are in layer which is directly connected to inputs of the network.

        Returns:
            list[list[float]]: Layer 0 of neural network which contains lists,
            where each nested list contains weights for each neuron in layer 0.
        """
        weights_for_layer_zero: list[list[float]] = []
        number_of_inputs_with_bias: int = self.number_of_inputs + 1
        for neuron_index in range(self.number_of_neurons_in_each_layer[0]):
            weights_for_neuron: list[float] = []
            for input_index in range(number_of_inputs_with_bias):
                random_weight = self._random_generate_weight()
                weights_for_neuron.append(random_weight)
            weights_for_layer_zero.append(weights_for_neuron)
        return weights_for_layer_zero

    def _random_init_weights_for_not_input_layers(self) -> list[list[list[float]]]:
        """Random initialization for weights associated with neurons that are in
        layers that are not directly connected to input layer.

        Returns:
            list[list[list[float]]]: Layers that are not directly connected with
            actual inputs of the network.
        """
        not_input_layers: list[list[list[float]]] = []
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        starting_layer_index: int = 1
        for layer_index in range(starting_layer_index, number_of_layers):
            weights_for_layer: list[list[float]] = []
            number_of_neurons_in_layer: int = self.number_of_neurons_in_each_layer[layer_index]
            for neuron_index in range(number_of_neurons_in_layer):
                weights_for_neuron: list[float] = []
                number_of_inputs_for_neuron_plus_bias: int = (
                        self.number_of_neurons_in_each_layer[layer_index-1] + 1)
                for input_index in range(number_of_inputs_for_neuron_plus_bias):
                    random_weight: float = self._random_generate_weight()
                    weights_for_neuron.append(random_weight)
                weights_for_layer.append(weights_for_neuron)
            not_input_layers.append(weights_for_layer)
        return not_input_layers

    @staticmethod
    def _random_generate_weight() -> float:
        random_weight = random()
        return random_weight if randint(0, 1) == 0 else random_weight * (-1)