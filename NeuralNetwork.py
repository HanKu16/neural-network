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

    def learn(self, sample: TrainingData):
        outputs_from_all_layers, inputs_for_all_layers = self._process_for_learning(sample.inputs)
        deltas: list[list[float]] = self._calculate_deltas(sample, outputs_from_all_layers)
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
                        deltas[layer_index][neuron_index],
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

    def save_to_file(self, path_to_file: str) -> None:
        """Read neural network from .txt file at given path.

        Args:
            path_to_file (str): path where serialized object of neural network is saved

        Raises:
            FileNotFoundError: If given directory does not exist
            PermissionError: If user does not have permission to access file or directory
            IsADirectoryError: If 'path_to_file' is a directory not path to .txt file
            OSError: If something goes wrong on low level operations
        """
        network_as_dict = {
            "number_of_inputs": self.number_of_inputs,
            "number_of_outputs": self.number_of_outputs,
            "number_of_neurons_in_each_layer": self.number_of_neurons_in_each_layer,
            "learning_rate": self.learning_rate,
            "weight_init_algorithm": self.weight_init_algorithm,
            "weights": self.weights
        }
        if self.activation_function == ActivationFunctions.jump_function:
            network_as_dict["activation_function"] = "jump_function"
        elif self.activation_function == ActivationFunctions.sigmoid_function:
            network_as_dict["activation_function"] = "sigmoid_function"
        network_json = json.dumps(network_as_dict, indent=4)
        with open(path_to_file, "w") as json_file:
            json_file.write(network_json)

    @staticmethod
    def read_from_file(path_to_file: str) -> "NeuralNetwork":
        """Read neural network from .txt file at given path.

        Args:
            path_to_file (str): path where serialized object of neural network is saved

        Raises:
            FileNotFoundError: If no file is found at given 'path_to_file'
            PermissionError: If user does not have permission to access file
            IsADirectoryError: If 'path_to_file' is a directory not .txt file
            json.JSONDecodeError: If data in the file is not valid JSON

        Returns:
            NeuralNetwork: Deserialized object of neural network.
        """
        with open(path_to_file, "r") as file:
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


    def setup_for_learning(self, number_of_inputs: int,
                           number_of_outputs: int,
                           number_of_neurons_in_each_hidden_layer: list[int],
                           learning_rate: float) -> None:
        """Set up neural network before learning process. Should not be called when
        existing neural network instance was read from the file.

        Args:
            number_of_inputs (int): number of inputs for neural network based on data samples
            number_of_outputs (int): number of outputs from neural network
            number_of_neurons_in_each_hidden_layer (list[int]):
                each number in the list specifies number of neuron in each hidden layer.
                First number specifies number of neurons in input layer, second in the next
                layer etc. If list is empty neural network will have 0 hidden layers.
            learning_rate:
                floating point number which have influence how fast weight of each neuron
                changes, the bigger number the more rapid changes. It is recommended to
                use learning rate in scope of (0, 1)
        """
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_hidden_layer
        self.number_of_neurons_in_each_layer.append(number_of_outputs)
        self.activation_function = ActivationFunctions.sigmoid_function
        self.learning_rate = learning_rate
        self.weight_init_algorithm = "xavier"

    def use_jump_activation_function(self) -> None:
        """Set jump function as activation function for neural network,
        which outputs: 0 if argument is negative, otherwise 1.
        Function is not differentiable so it can not be use with
        backpropagation learning algorithm.
        """
        self.activation_function = ActivationFunctions.jump_function

    def use_sigmoid_activation_function(self) -> None:
        """Set sigmoid as activation function for neural network.
        For sigmoid function output is always in scope (0, 1).
        Learn more: https://en.wikipedia.org/wiki/Sigmoid_function.
        """
        self.activation_function = ActivationFunctions.sigmoid_function

    def use_xavier_weight_initialization(self) -> None:
        """Set xavier weight initialization. This initialization works
        fine with activation functions like sigmoid, tanh etc.
        """
        self.weight_init_algorithm = "xavier"

    def use_random_weight_initialization(self) -> None:
        """Set random weight initialization. Each weight
        in neural network is random number from scope <-1, 1>.
        """
        self.weight_init_algorithm = "random"

    def _process_for_learning(self, inputs_from_sample: list[float]) -> (list[list[float]], list[list[float]]):
        """Process given sample from 'inputs_from_sample'.

        Args:
            inputs_from_sample (list[float]): input vector from data sample

        Returns:
            (list[list[float]], list[list[float]]):
                Elements in the value of first tuple are all the outputs given by each neuron
                in the network for given sample 'inputs_from_sample'. To get output of
                particular neuron refer by [layer_index][neuron_index].
                Elements in the value of second tuple are all the inputs given to each neuron
                in the network fir given sample. To get input given to particular weight
                (input number) of neuron refer by [layer_index][weight_index]. Remember that
                for each layer on [layer_index][0] there is bias input equal 1.
        """
        outputs_from_all_layers: list[list[float]] = []
        inputs_for_all_layers: list[list[float]] = []
        number_of_layers: int = len(self.weights)
        input_vector_with_bias = self._get_input_vector_with_bias_input(inputs_from_sample)
        for layer_index in range(number_of_layers):
            output_from_layer = []
            inputs_for_all_layers.append(input_vector_with_bias[:])
            number_of_neurons_in_layer: int = self.number_of_neurons_in_each_layer[layer_index]
            for neuron_index in range(number_of_neurons_in_layer):
                weights_for_neuron: list[float] = self.weights[layer_index][neuron_index]
                u: float = self._calculate_u(input_vector_with_bias, weights_for_neuron)
                neuron_output: float = self.activation_function(u)
                output_from_layer.append(neuron_output)
            outputs_from_all_layers.append(output_from_layer)
            input_vector_with_bias = self._get_input_vector_with_bias_input(output_from_layer[:])
        return outputs_from_all_layers, inputs_for_all_layers

    def _adjust_weight(self, weight: float, delta: float, input_for_this_weight: float) -> float:
        """Calculate new weight based on backpropagation algorithm.

        Args:
            weight (float): weight to be adjusted for next iteration
            delta (float): delta calculated for neuron
            input_for_this_weight (float): input value that was used with weight

        Returns:
            float: New value of the weight.
        """
        return weight + self.learning_rate * delta * input_for_this_weight

    def _calculate_deltas(self, sample: TrainingData, outputs_from_all_layers: list[list[float]]) -> list[list[float]]:
        """Calculate delta for each neuron in all layers.

        Args:
            sample (TrainingData): sample of data, which was processed by network in current iteration
            outputs_from_all_layers (list[list[float]]): output of each neuron in all layers for given sample

        Returns:
            list[list[float]]: Deltas for each neuron in all layers.
        """
        output_from_last_layer: list[float] = outputs_from_all_layers[-1]
        deltas_for_output_layer: list[float] = self._calculate_deltas_for_output_layer(
            sample, output_from_last_layer)
        deltas_for_hidden_layers: list[list[float]] = self._calculate_deltas_for_hidden_layers(
            outputs_from_all_layers, deltas_for_output_layer)
        deltas_for_all_layers = deltas_for_hidden_layers
        deltas_for_all_layers.append(deltas_for_output_layer)
        return deltas_for_all_layers

    def _calculate_deltas_for_output_layer(self, sample: TrainingData, output_from_last_layer: list[float]) -> list[float]:
        """Calculate deltas for all neurons in the output layer.

        Args:
            sample (TrainingData): sample of data, which was processed by network in current iteration
            output_from_last_layer (list[float]): actual output, which was returned by neural network for processed sample

        Returns:
            list[float]: Errors for each neuron in output layer.
        """
        deltas_for_output_layer: list[float] = []
        for output_index in range(self.number_of_outputs):
            d_for_neuron: float = sample.desired_outputs[output_index]
            y_for_neuron: float = output_from_last_layer[output_index]
            delta_for_neuron: float = (d_for_neuron - y_for_neuron) * y_for_neuron * (1 - y_for_neuron)
            deltas_for_output_layer.append(delta_for_neuron)
        return deltas_for_output_layer

    def _calculate_deltas_for_hidden_layers(self, outputs_from_all_layers: list[list[float]],
                                            deltas_for_output_layer: list[float]) -> list[list[float]]:
        """Calculate deltas for all neurons in the hidden layers.

        Args:
            outputs_from_all_layers (Tlist[list[float]]): outputs from each neuron in neural network
            deltas_for_output_layer (list[float]): deltas calculated for each neuron in output layer of neural network

        Returns:
            list[float]: Errors for each neuron in output layer.
        """
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        one_before_last_layer_index: int = number_of_layers - 2
        deltas_for_hidden_layers = [deltas_for_output_layer]
        for layer_index in range(one_before_last_layer_index, -1, -1):
            deltas_for_neurons_in_layer = []
            number_of_neurons_in_layer = self.number_of_neurons_in_each_layer[layer_index]
            for neuron_index in range(number_of_neurons_in_layer):
                neuron_error = self._calculate_delta_for_neuron_in_hidden_layer(
                    layer_index, neuron_index, outputs_from_all_layers, deltas_for_hidden_layers[0])
                deltas_for_neurons_in_layer.append(neuron_error)
            deltas_for_hidden_layers.insert(0, deltas_for_neurons_in_layer)
        deltas_for_hidden_layers.pop() # removing deltas for output layer
        return deltas_for_hidden_layers

    def _calculate_delta_for_neuron_in_hidden_layer(self, layer_index: int, neuron_index: int,
                                                    outputs_from_all_layers: list[list[float]],
                                                    deltas_for_next_layer: list[float]) -> float:
        """Calculate delta for neuron, what is needed to adjust weights for
        next iterations of backpropagation algorithm.

        Args:
            layer_index (int): index of the layer, where neuron is
            neuron_index (int): index of neuron that you want to calculate error
            outputs_from_all_layers (list[list[float]]): outputs from each neuron in each layer
            deltas_for_next_layer (list[float]): errors calculated for the neurons in layer of index layer_index+1

        Returns:
            float: Delta for neuron.
        """
        delta: float = 0
        number_of_neurons_in_next_layer: int = self.number_of_neurons_in_each_layer[layer_index+1]
        for neuron_in_next_layer_index in range(number_of_neurons_in_next_layer):
            weight = self.weights[layer_index+1][neuron_in_next_layer_index][neuron_index+1]
            error_for_neuron_in_next_layer = deltas_for_next_layer[neuron_in_next_layer_index]
            delta += error_for_neuron_in_next_layer * weight
        neuron_output = outputs_from_all_layers[layer_index][neuron_index]
        delta = delta * (-1) * neuron_output * (1 - neuron_output)
        return delta

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

    def __str__(self) -> str:
        """
        Returns:
            str: Weights of neural network in text form.
        """
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
