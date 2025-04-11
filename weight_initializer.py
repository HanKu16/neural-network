import math
from random import uniform, random, randint
import numpy as np

class WeightInitializer:
    def __init__(self, number_of_inputs: int, number_of_neurons_in_each_layer: list[int]):
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_layer

    def xavier_init(self) -> list[np.ndarray]:
        """Xavier initialization for weights associated with each neuron in the network.
        Works well with sigmoid activation function. To get specific value from returned
        value refer by [index_layer][neuron_layer][input_layer].

        Returns:
            list[ndarray]: List of layers, each layer contains neurons and each neuron
            contains weights.
        """
        weights_for_all_layers = []
        weights_for_layer_zero = self._xavier_init_for_layer_zero()
        weights_for_all_layers.append(weights_for_layer_zero)
        weights_for_not_input_layers = self._xavier_init_weights_for_not_input_layers()
        weights_for_all_layers.extend(weights_for_not_input_layers)
        return weights_for_all_layers

    def random_init(self, lower_bound_for_weight: float,
                    upper_bound_for_weight: float) -> list[np.ndarray]:
        """Random initialization for weights associated with neurons
        that are in layer which is directly connected to inputs of
        the network. Initialize weights be assigning to each weight
        random value between <-1, 1>.

        Args:
            lower_bound_for_weight (float): minimal value that can be assigned to weight
            upper_bound_for_weight (float): maximal value that can be assigned to weight

        Returns:
            list[ndarray]: List of layers, each layer contains neurons and each neuron
            contains weights.
        """
        weights_for_all_layers = []
        weights_for_layer_zero = self._random_init_for_layer_zero(
            lower_bound_for_weight, upper_bound_for_weight)
        weights_for_all_layers.append(weights_for_layer_zero)
        weights_for_not_input_layers = self._random_init_weights_for_not_input_layers(
            lower_bound_for_weight, upper_bound_for_weight)
        weights_for_all_layers.extend(weights_for_not_input_layers)
        return weights_for_all_layers

    def _xavier_init_for_layer_zero(self) -> np.ndarray:
        """Xavier initialization for weights associated with neurons that
        are in layer which is directly connected to inputs of the network.

        Returns:
            ndarray: Weights for neurons in layer 0 of neural network which
            are directly connected with actual inputs of the network.
        """
        f_in: int = self.number_of_inputs
        f_out: int = self.number_of_neurons_in_each_layer[0]
        limit: float = self._xavier_calculate_limit(f_in, f_out)
        number_of_inputs_with_bias: int = self.number_of_inputs + 1
        number_of_neurons_in_layer_zero: int = self.number_of_neurons_in_each_layer[0]
        weights_for_layer_zero: np.ndarray = self._xavier_generate_weights_for_layer(
            number_of_neurons_in_layer_zero, number_of_inputs_with_bias, limit)
        return weights_for_layer_zero

    def _xavier_init_weights_for_not_input_layers(self) -> list[np.ndarray]:
        """Xavier initialization for weight associated with neurons that
        are in layers which are not directly connected to inputs of the
        network.

        Returns:
            list[ndarray]: Weights for neurons in layers that are not
            directly connected with actual inputs of the network.
        """
        not_input_layers: list[np.ndarray] = []
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        starting_layer_index: int = 1
        for layer_index in range(starting_layer_index, number_of_layers):
            f_in: int = self.number_of_neurons_in_each_layer[layer_index-1]
            f_out: int = self.number_of_neurons_in_each_layer[layer_index]
            limit = self._xavier_calculate_limit(f_in, f_out)
            number_of_inputs_with_bias: int = self.number_of_neurons_in_each_layer[layer_index-1] + 1
            number_of_neurons_in_layer: int = self.number_of_neurons_in_each_layer[layer_index]
            weights_for_layer: np.ndarray = self._xavier_generate_weights_for_layer(
                number_of_neurons_in_layer, number_of_inputs_with_bias, limit)
            not_input_layers.append(weights_for_layer)
        return not_input_layers

    @staticmethod
    def _xavier_calculate_limit(f_in: int, f_out: int):
        return math.sqrt(6 / (f_in + f_out))

    @staticmethod
    def _xavier_generate_weights_for_layer(number_of_neurons_in_layer: int,
        number_of_inputs_with_bias: int, limit: float):
        return np.random.uniform(-limit, limit, size=(
            number_of_neurons_in_layer, number_of_inputs_with_bias))

    def _random_init_for_layer_zero(self, lower_bound_for_weight: float,
                    upper_bound_for_weight: float) -> np.ndarray:
        """Random initialization for weight associated with neurons that
         are in layers which are not directly connected to inputs of the
         network.

        Args:
            lower_bound_for_weight (float): minimal value that can be assigned to weight
            upper_bound_for_weight (float): maximal value that can be assigned to weight

        Returns:
            list[ndarray]: Weights for neurons in layers that are not
            directly connected with actual inputs of the network.
        """
        number_of_inputs_with_bias: int = self.number_of_inputs + 1
        number_of_neurons_in_layer_zero: int = self.number_of_neurons_in_each_layer[0]
        return self._random_generate_weights_for_layer(lower_bound_for_weight,
            upper_bound_for_weight, number_of_neurons_in_layer_zero,
            number_of_inputs_with_bias)

    def _random_init_weights_for_not_input_layers(self, lower_bound_for_weight: float,
                    upper_bound_for_weight: float) -> list[np.ndarray]:
        """Random initialization for weight associated with neurons that
        are in layers which are not directly connected to inputs of the
        network.

        Args:
            lower_bound_for_weight (float): minimal value that can be assigned to weight
            upper_bound_for_weight (float): maximal value that can be assigned to weight

        Returns:
            list[ndarray]: Weights for neurons in layers that are not
            directly connected with actual inputs of the network.
        """
        not_input_layers: list[np.ndarray] = []
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        starting_layer_index: int = 1
        for layer_index in range(starting_layer_index, number_of_layers):
            number_of_inputs_with_bias: int = self.number_of_neurons_in_each_layer[layer_index-1] + 1
            number_of_neurons_in_layer: int = self.number_of_neurons_in_each_layer[layer_index]
            weights_for_layer: np.ndarray = self._random_generate_weights_for_layer(
                lower_bound_for_weight, upper_bound_for_weight,
                number_of_neurons_in_layer, number_of_inputs_with_bias)
            not_input_layers.append(weights_for_layer)
        return not_input_layers

    @staticmethod
    def _random_generate_weights_for_layer(lower_bound_for_weight: float,
            upper_bound_for_weight: float, number_of_neurons_in_layer: int,
            number_of_inputs_with_bias: int) -> np.ndarray:
        return np.random.uniform(lower_bound_for_weight, upper_bound_for_weight, size=(
            number_of_neurons_in_layer, number_of_inputs_with_bias))