from TrainingData import TrainingData
import ActivationFunctions
import json
from WeightInitializer import WeightInitializer

class NeuralNetwork:
    def __init__(self) -> None:
        self.number_of_inputs = None
        self.number_of_outputs = None
        self.number_of_neurons_in_each_layer = None
        self.activation_function = None
        self.learning_rate = None
        self.weight_init_algorithm = None
        self.weights = None

    def setup_for_learning(self, number_of_inputs: int,
                           number_of_outputs: int,
                           number_of_neurons_in_each_hidden_layer: list[int],
                           learning_rate: float) -> None:
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_hidden_layer
        self.number_of_neurons_in_each_layer.append(number_of_outputs)
        self.activation_function = ActivationFunctions.sigmoid_function
        self.learning_rate = learning_rate
        self.weight_init_algorithm = "xavier"

    def process(self, inputs: list[float]) -> list[float]:
        """Process input vector given to the network and return output vector
        which is product the network and inputs vector.

        Args:
            inputs (list[float]): vector which contains all the input values for given neuron

        Returns:
            list[float]: output vector from the network
        """
        output_from_layer: list[float] = []
        number_of_layers: int = len(self.weights)
        for layer_index in range(number_of_layers):
            output_from_layer = []
            inputs = self._get_input_vector_with_bias_input(inputs)
            number_of_neurons_in_layer: int = len(self.weights[layer_index])
            for neuron_index in range(number_of_neurons_in_layer):
                weights_for_neuron: list[float] = self.weights[layer_index][neuron_index]
                u: float = self._calculate_u(inputs, weights_for_neuron)
                neuron_output: float = self.activation_function(u)
                output_from_layer.append(neuron_output)
            inputs = output_from_layer[:]
        return output_from_layer

    def _process_for_learning(self, inputs_from_sample: list[float]) -> (list[list[float]], list[list[float]]):
        outputs_from_all_layers: list[list[float]] = []
        inputs_for_all_layers: list[list[float]] = []
        number_of_layers: int = len(self.weights)
        inputs_vector_with_bias = self._get_input_vector_with_bias_input(inputs_from_sample)
        for layer_index in range(number_of_layers):
            output_from_layer = []
            inputs_for_all_layers.append(inputs_vector_with_bias[:])
            number_of_neurons_in_layer: int = self.number_of_neurons_in_each_layer[layer_index]
            for neuron_index in range(number_of_neurons_in_layer):
                weights_for_neuron: list[float] = self.weights[layer_index][neuron_index]
                u: float = self._calculate_u(inputs_vector_with_bias, weights_for_neuron)
                neuron_output: float = self.activation_function(u)
                output_from_layer.append(neuron_output)
            outputs_from_all_layers.append(output_from_layer)
            inputs_vector_with_bias = self._get_input_vector_with_bias_input(output_from_layer[:])
        return outputs_from_all_layers, inputs_for_all_layers

    def learn(self, sample: TrainingData):
        outputs_from_all_layers, inputs_for_all_layers = self._process_for_learning(sample.inputs)
        errors: list[list[float]] = self._calculate_errors(sample, outputs_from_all_layers)
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        new_weights = []
        for layer_index in range(number_of_layers):
            number_of_neurons_in_layer = self.number_of_neurons_in_each_layer[layer_index]
            weights_for_layer = []
            for neuron_index in range(number_of_neurons_in_layer):
                number_of_weights_for_neuron = len(self.weights[layer_index][neuron_index])
                weights_for_neuron = []
                for weight_index in range(number_of_weights_for_neuron):
                    weight = self._adjust_weight(
                        self.weights[layer_index][neuron_index][weight_index],
                        errors[layer_index][neuron_index],
                        inputs_for_all_layers[layer_index][weight_index])
                    weights_for_neuron.append(weight)
                weights_for_layer.append(weights_for_neuron)
            new_weights.append(weights_for_layer)
        self.weights = new_weights
        return outputs_from_all_layers[-1]

    def init_weights(self) -> None:
        """Initialized weights for neural network. Should be called only
        before learning process, there is no need to init weight, when
        network is read from file. Uses initialization algorithm configured
        by api, if not specified explicitly uses Xavier Initialization by
        default.
        """
        weight_initializer: WeightInitializer = WeightInitializer(
            self.number_of_inputs, self.number_of_neurons_in_each_layer)
        if self.weight_init_algorithm == "xavier":
            self.weights = weight_initializer.xavier_init()
        elif self.weight_init_algorithm == "random":
            self.weights = weight_initializer.random_init()
        else:
            self.weights = weight_initializer.xavier_init()

    def _adjust_weight(self, weight: float, error: float, input_for_this_weight: float) -> float:
        """Calculate new weight based on backpropagation algorithm.

        Args:
            weight (float): weight to be adjusted for next iteration
            error (float): error calculated for neuron
            input_for_this_weight (float): input value that was used with weight

        Returns:
            float: New value of the weight.
        """
        return weight + self.learning_rate * error * input_for_this_weight

    def _calculate_errors(self, sample: TrainingData, outputs_from_all_layers: list[list[float]]) -> list[list[float]]:
        """Calculate error for each neuron in all layers.

        Args:
            sample (TrainingData): sample of data, which was processed by network in current iteration
            outputs_from_all_layers (list[list[float]]): output of each neuron in all layers for given sample

        Returns:
            list[list[float]]: Errors for each neuron in all layers.
        """
        output_from_last_layer: list[float] = outputs_from_all_layers[-1]
        errors_from_output_layer: list[float] = self._calculate_errors_for_output_layer(
            sample, output_from_last_layer)
        errors_for_hidden_layer: list[list[float]] = self._calculate_errors_for_hidden_layers(
            outputs_from_all_layers, errors_from_output_layer)
        return errors_for_hidden_layer + errors_from_output_layer

    def _calculate_errors_for_output_layer(self, sample: TrainingData, output_from_last_layer: list[float]) -> list[float]:
        """Calculate error for all neurons in the output layer.

        Args:
            sample (TrainingData): sample of data, which was processed by network in current iteration
            output_from_last_layer (list[float]): actual output, which was returned by neural network for processed sample

        Returns:
            list[float]: Errors for each neuron in output layer.
        """
        errors_for_output_layer: list[float] = []
        for output_index in range(self.number_of_outputs):
            d_for_neuron: float = sample.desired_outputs[output_index]
            y_for_neuron: float = output_from_last_layer[output_index]
            error_for_neuron: float = (d_for_neuron - y_for_neuron) * y_for_neuron * (1 - y_for_neuron)
            errors_for_output_layer.append(error_for_neuron)
        return errors_for_output_layer

    def _calculate_errors_for_hidden_layers(self, outputs_from_all_layers: list[list[float]],
                                            errors_for_output_layer: list[float]):
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        one_before_last_layer_index: int = number_of_layers - 2
        errors = [errors_for_output_layer]
        for layer_index in range(one_before_last_layer_index, -1, -1):
            errors_for_layer = []
            number_of_neurons_in_layer = self.number_of_neurons_in_each_layer[layer_index]
            for neuron_index in range(number_of_neurons_in_layer):
                neuron_error = self._calculate_error_for_neuron_in_hidden_layer(
                    layer_index, neuron_index, outputs_from_all_layers, errors[0])
                errors_for_layer.append(neuron_error)
            errors.insert(0, errors_for_layer)
        return errors

    def _calculate_error_for_neuron_in_hidden_layer(self, layer_index: int, neuron_index: int,
                                                    outputs_from_all_layers: list[list[float]],
                                                    errors_for_next_layer: list[float]) -> float:
        """Calculate error for neuron, what is needed to adjust weights for
        next iterations of backpropagation algorithm.

        Args:
            layer_index (int): index of the layer, where neuron exists
            neuron_index (int): index of neuron that you want to calculate error
            outputs_from_all_layers (list[list[float]]): outputs from each neuron in each layer
            errors_for_next_layer (list[float]): errors calculated for the neurons in layer of index layer_index+1

        Returns:
            float: Error for neuron.
        """
        error: float = 0
        number_of_neurons_in_next_layer: int = self.number_of_neurons_in_each_layer[layer_index+1]
        for neuron_in_next_layer_index in range(number_of_neurons_in_next_layer):
            weight = self.weights[layer_index+1][neuron_in_next_layer_index][neuron_index+1]
            error_for_neuron_in_next_layer = errors_for_next_layer[neuron_in_next_layer_index]
            error += error_for_neuron_in_next_layer * weight
        neuron_output = outputs_from_all_layers[layer_index][neuron_index]
        error = error * (-1) * neuron_output * (1 - neuron_output)
        return error

    def save_to_file(self, path_to_file: str):
        network_as_dict = {
            "number_of_inputs": self.number_of_inputs,
            "number_of_outputs": self.number_of_outputs,
            "number_of_neurons_in_each_layer": self.number_of_neurons_in_each_layer,
            "weights": self.weights,
            "learning_coefficient": self.learning_rate
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

    def use_xavier_weight_initialization(self) -> None:
        self.weight_init_algorithm = "xavier"

    def use_random_weight_initialization(self) -> None:
        self.weight_init_algorithm = "random"

    @staticmethod
    def _calculate_u(inputs: list[float], weights_for_neuron: list[float]) -> float:
        """Calculate value that is weighted sum of each input in inputs vector
        and associated weight.

        Args:
            inputs (list[float]): vector which contains all the input values for given neuron
            weights_for_neuron (list[float]): vector which contains all weights for neuron

        Returns:
            float: Value should be passed as argument to activation function
        """
        number_of_inputs_for_neuron: int = len(inputs)
        u = 0
        for input_index in range(number_of_inputs_for_neuron):
            u += inputs[input_index] * weights_for_neuron[input_index]
        return u

    @staticmethod
    def _get_input_vector_with_bias_input(inputs: list[float]) -> list[float]:
        """Create new inputs vector which contains artificial bias input of value 1
        at index 0.

        Args:
            inputs (list[float]): vector which contains all the input values for given neuron

        Returns:
            list[float]: New input vector containing bias input.
        """
        input_for_bias_weight: int = 1
        return [input_for_bias_weight] + inputs

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
