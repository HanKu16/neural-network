import numpy as np
from training_data import TrainingData
import activation_functions
import json
from weight_initializer import WeightInitializer
from random import randint
import pandas as pd
from datetime import datetime

class NeuralNetwork:
    def __init__(self) -> None:
        self.number_of_inputs = None
        self.number_of_outputs = None
        self.number_of_neurons_in_each_layer = None
        self.activation_function = None
        self.learning_rate = None
        self.weight_init_algorithm = None
        self.t_max = None
        self.q_min = None
        self.weights = None
        self.test_set = None
        self.is_test_evaluation_enabled = False
        self.is_logging_training_progress_to_console_enabled = False
        self.is_reporting_training_progress_to_excel_file_enabled = False

    def process(self, inputs: list[float]) -> list[float]:
        """Process input vector given to the network and return output vector
        which is product the network and inputs vector.

        Args:
            inputs (list[float]): vector which contains all the input values for given neuron

        Returns:
            list[float]: Output vector from the network
        """
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        input_vector: np.ndarray = np.array(inputs)
        for layer_index in range(number_of_layers):
            input_vector = np.hstack(([1.0], input_vector))
            weighted_sums_vector: np.ndarray = self.weights[layer_index] @ input_vector
            input_vector = self.activation_function(weighted_sums_vector)
        output_vector: list[float] = input_vector.tolist()
        return output_vector

    def learn(self, learning_set: list[TrainingData]) -> dict:
        t: int = 0
        e: int = 0
        q_for_current_epoch: float = 0
        indexes_of_not_processed_samples_in_epoch: list[int] = list(range(len(learning_set)))
        training_progress: dict = self._init_training_progress_dict()
        started_at: datetime = datetime.now()
        finished_at: datetime = started_at

        while t < self.t_max:
            sample: TrainingData = self._get_random_sample_from_set(
                indexes_of_not_processed_samples_in_epoch, learning_set)
            outputs_from_all_layers, inputs_for_all_layers = (
                self._process_for_learning(sample.inputs))
            deltas: list[np.ndarray] = self._calculate_deltas(sample, outputs_from_all_layers)
            self._adjust_weights(deltas, inputs_for_all_layers)
            mse: float = self._calculate_mean_squared_error(
                outputs_from_all_layers[-1], sample.desired_outputs)
            t += 1
            q_for_current_epoch += mse
            if len(indexes_of_not_processed_samples_in_epoch) == 0:
                finished_at = datetime.now()
                training_progress = self._handle_epoch_end(
                    e, q_for_current_epoch, started_at, finished_at, training_progress)
                e += 1
                indexes_of_not_processed_samples_in_epoch: list[int] = list(range(len(learning_set)))
                if (e >= 150) and (e % 10 == 0):
                    choice = input("Do you want to stop [yes/no]: ")
                    if choice == 'yes':
                        break
                if q_for_current_epoch < self.q_min:
                    break
                else:
                    q_for_current_epoch = 0
                started_at = datetime.now()
        if self.is_reporting_training_progress_to_excel_file_enabled:
            self._report_training_progress_to_excel_file(training_progress)
        return training_progress

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
            self.weights = weight_initializer.random_init(-1, 1)
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
        weights = []
        for i in range(len(self.weights)):
            weights.append(self.weights[i].tolist())
        network_as_dict = {
            "number_of_inputs": self.number_of_inputs,
            "number_of_outputs": self.number_of_outputs,
            "number_of_neurons_in_each_layer": self.number_of_neurons_in_each_layer,
            "learning_rate": self.learning_rate,
            "weight_init_algorithm": self.weight_init_algorithm,
            "weights": weights
        }
        if self.activation_function == activation_functions.jump_function:
            network_as_dict["activation_function"] = "jump_function"
        elif self.activation_function == activation_functions.sigmoid_function:
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

    def setup_structure(self, number_of_inputs: int,
                        number_of_outputs: int,
                        number_of_neurons_in_each_hidden_layer: list[int]) -> None:
        """Set up neural network structure before learning process. Should not
        be called when existing neural network instance was read from the file.

        Args:
            number_of_inputs (int): number of inputs for neural network based on data samples
            number_of_outputs (int): number of outputs from neural network
            number_of_neurons_in_each_hidden_layer (list[int]):
                each number in the list specifies number of neuron in each hidden layer.
                First number specifies number of neurons in input layer, second in the next
                layer etc. If list is empty neural network will have 0 hidden layers.
        """
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_hidden_layer
        self.number_of_neurons_in_each_layer.append(number_of_outputs)
        self.activation_function = activation_functions.sigmoid_function

    def setup_learning_params(self, learning_rate: float, t_max: int, q_min: float) -> None:
        """Set up neural network structure before learning params. Should not
        be called when existing neural network instance was read from the file.

        Args:
            learning_rate (float):
                floating point number which have influence how fast weight of each neuron
                changes, the bigger number the more rapid changes. It is recommended to
                use learning rate in scope of (0, 1)
            t_max (int):
                max number of iterations to execute by learning algorithm, after reaching
                this iteration learning algorithm stops
            q_min (float):
                when sum of errors for outputs of neural network for each sample in the
                epoch is less than this number, learning algorithm stops
        """
        self.learning_rate = learning_rate
        self.t_max = t_max
        self.q_min = q_min
        self.weight_init_algorithm = "xavier"

    def use_jump_activation_function(self) -> None:
        """Set jump function as activation function for neural network,
        which outputs: 0 if argument is negative, otherwise 1.
        Function is not differentiable so it can not be use with
        backpropagation learning algorithm.
        """
        self.activation_function = activation_functions.jump_function

    def use_sigmoid_activation_function(self) -> None:
        """Set sigmoid as activation function for neural network.
        For sigmoid function output is always in scope (0, 1).
        Learn more: https://en.wikipedia.org/wiki/Sigmoid_function.
        """
        self.activation_function = activation_functions.sigmoid_function

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

    def enable_test_set_evaluation_after_each_epoch(self, test_set: list[TrainingData]):
        """After each processed epoch test set will be processed to check quality of model."""
        self.is_test_evaluation_enabled = True
        self.test_set = test_set

    def enable_logging_training_progress_to_console(self):
        """Information about each processed epoch will be logged in the console."""
        self.is_logging_training_progress_to_console_enabled = True

    def enable_reporting_training_progress_to_excel_file(self):
        """Information about each processed epoch will be saved to excel file."""
        self.is_reporting_training_progress_to_excel_file_enabled = True

    def _process_for_learning(self, inputs_from_sample: np.ndarray) \
            -> (list[np.ndarray], list[np.ndarray]):
        """Process given sample.

        Args:
            inputs_from_sample (ndarray): input vector from data sample

        Returns:
            (list[np.ndarray], list[np.ndarray]):
                Elements in the value of first tuple are all the outputs given by each neuron
                in the network for given sample 'inputs_from_sample'. To get output of
                particular neuron refer by [layer_index][neuron_index].
                Elements in the value of second tuple are all the inputs given to each neuron
                in the network fir given sample. To get input given to particular weight
                (input number) of neuron refer by [layer_index][weight_index]. Remember that
                for each layer on [layer_index][0] there is bias input equal 1.
        """
        inputs_for_all_layers: list[np.ndarray] = []
        outputs_from_all_layers: list[np.ndarray] = []
        number_of_layers: int = len(self.weights)
        input_vector: np.ndarray = inputs_from_sample
        for layer_index in range(number_of_layers):
            input_vector_with_bias: np.ndarray = np.hstack(([1.0], input_vector[:]))
            inputs_for_all_layers.append(input_vector_with_bias)
            weighted_sum_vector: np.ndarray = self.weights[layer_index] @ input_vector_with_bias
            output_vector: np.ndarray = self.activation_function(weighted_sum_vector)
            outputs_from_all_layers.append(output_vector)
            input_vector = output_vector
        return outputs_from_all_layers, inputs_for_all_layers

    def _adjust_weights(self, deltas_for_all_layers: list[np.ndarray],
                        inputs_for_all_layers: list[np.ndarray]) -> None:
        """Adapt weights based on backpropagation algorithm.

        Args:
            deltas_for_all_layers (list[np.ndarray]): deltas for each neuron in the network
            inputs_for_all_layers (list[np.ndarray]): inputs given for each layer
        """
        weights: list[np.ndarray] = []
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        for layer_index in range(number_of_layers):
            weights_for_layer: list[np.ndarray] = []
            number_of_neurons_in_layer: int = self.number_of_neurons_in_each_layer[layer_index]
            for neuron_index in range(number_of_neurons_in_layer):
                weights_for_neuron: np.ndarray = (self.weights[layer_index][neuron_index] + self.learning_rate *
                                      deltas_for_all_layers[layer_index][neuron_index] * inputs_for_all_layers[layer_index])
                weights_for_layer.append(weights_for_neuron)
            weights.append(np.array(weights_for_layer))
        self.weights = weights

    def _process_test_set(self) -> float:
        """Process test set to check quality of model at current state.

        Returns:
            float:
                Sum of mean squared error for all samples in test set.
        """
        q_for_test_set: float = 0
        indexes_of_not_processed_samples: list[int] = list(range(len(self.test_set)))

        while len(indexes_of_not_processed_samples) != 0:
            sample: TrainingData = self._get_random_sample_from_set(
                indexes_of_not_processed_samples, self.test_set)
            outputs: list[float] = self.process(sample.inputs)
            mse: float = self._calculate_mean_squared_error(np.array(outputs), sample.desired_outputs)
            q_for_test_set += mse
        return q_for_test_set

    def _handle_epoch_end(self, e: int, q_for_epoch: float, started_at, finished_at,
                          training_progress: dict) -> dict:
        """Logs messages if set. Updated training progress information.

        Args:
            e (int): index of epoch that finished
            q_for_epoch (float): error for epoch
            training_progress (dict): dictionary storing information about training

        Returns:
            dict: Dictionary updated with information about last processed epoch.
        """
        training_progress["e"].append(e)
        training_progress["q_for_epoch"].append(q_for_epoch)
        training_progress["started_at"].append(started_at)
        training_progress["finished_at"].append(finished_at)
        training_progress["total_time"].append(finished_at-started_at)
        if self.is_test_evaluation_enabled:
            q_for_test_set = self._process_test_set()
            log_message = (f"epoch {e} | q for epoch {q_for_epoch} | q for test {q_for_test_set} | "
                           f"started at {started_at} | finished at {finished_at} | "
                           f"total time {finished_at - started_at}")
            training_progress["q_for_test_set"].append(q_for_test_set)
        else:
            log_message = (f"epoch {e} | q for epoch {q_for_epoch} | "
                           f"started at {started_at} | finished at {finished_at} | "
                           f"total time {finished_at - started_at}")
        if self.is_logging_training_progress_to_console_enabled:
            print(log_message)
        return training_progress

    def _calculate_deltas(self, sample: TrainingData, outputs_from_all_layers: list[np.ndarray]) \
            -> list[np.ndarray]:
        """Calculate delta for each neuron in all layers.

        Args:
            sample (TrainingData):
                sample of data, which was processed by network in current iteration
            outputs_from_all_layers (list[list[float]]):
                output from each neuron in all layers for processed sample

        Returns:
            list[ndarray]: List of vectors containing deltas for neurons.
        """
        output_from_last_layer: np.ndarray = outputs_from_all_layers[-1]
        deltas_for_output_layer: np.ndarray = self._calculate_deltas_for_output_layer(
            sample.desired_outputs, output_from_last_layer)
        deltas_for_hidden_layers: list[np.ndarray] = self._calculate_deltas_for_hidden_layers(
            outputs_from_all_layers, deltas_for_output_layer)
        deltas_for_all_layers: list[np.ndarray] = deltas_for_hidden_layers
        deltas_for_all_layers.append(deltas_for_output_layer)
        return deltas_for_all_layers

    @staticmethod
    def _calculate_deltas_for_output_layer(target_vector: np.ndarray,
                                            output_vector: np.ndarray) -> np.ndarray:
        """Calculate deltas for all neurons in the output layer.

        Args:
            target_vector (ndarray):
                vector which contains desired outputs for output layer
            output_vector (ndarray):
                vector which contains actual outputs from output layer

        Returns:
            ndarray: Vector with deltas for each neuron from output layer.
        """
        return (target_vector - output_vector) * output_vector * (1 - output_vector)

    def _calculate_deltas_for_hidden_layers(self, outputs_from_all_layers: list[np.ndarray],
                                            deltas_for_output_layer: np.ndarray) -> list[np.ndarray]:
        """Calculate deltas for all neurons in the hidden layers.

        Args:
            outputs_from_all_layers (list[np.ndarray]):
                outputs from each neuron in neural network in form of list of vectors
            deltas_for_output_layer (np.ndarray):
                vector consisting of deltas calculated for each neuron in output layer

        Returns:
            list[np.ndarray]:
                List of vectors containing deltas for neurons that are in hidden layers.
        """
        number_of_layers: int = len(self.number_of_neurons_in_each_layer)
        one_before_last_layer_index: int = number_of_layers - 2
        deltas_for_hidden_layers: list[np.ndarray] = [deltas_for_output_layer]
        for layer_index in range(one_before_last_layer_index, -1, -1):
            deltas_for_next_layer: np.ndarray = deltas_for_hidden_layers[0]
            transposed_weights_for_layer_without_biases: np.ndarray = (
                (self.weights[layer_index+1])[:,1:].T)
            weighted_sums: np.ndarray = (transposed_weights_for_layer_without_biases
                                         @ deltas_for_next_layer)
            deltas_for_layer: np.ndarray = self._calculate_deltas_vector_for_hidden_layer(
                weighted_sums, outputs_from_all_layers[layer_index])
            deltas_for_hidden_layers.insert(0, deltas_for_layer)
        deltas_for_hidden_layers.pop()  # removing deltas for output layer
        return deltas_for_hidden_layers

    @staticmethod
    def _calculate_deltas_vector_for_hidden_layer(weighted_sums: np.ndarray,
            output_vector_from_layer: np.ndarray) -> np.ndarray:
        return (-1) * weighted_sums * output_vector_from_layer * (1 - output_vector_from_layer)

    @staticmethod
    def _get_random_sample_from_set(indexes_of_not_processed_samples: list[int],
                                    sample_set: list[TrainingData]) -> TrainingData:
        """Get random sample from sample set, removes sample index from the set
        to not allow for getting the same sample more than once.

        Returns:
            TrainingData: Drawn random sample.
        """
        index_from_not_processed_samples: int = randint(
            0, len(indexes_of_not_processed_samples) - 1)
        sample_index: int = indexes_of_not_processed_samples[index_from_not_processed_samples]
        del indexes_of_not_processed_samples[index_from_not_processed_samples]
        sample: TrainingData = sample_set[sample_index]
        return sample

    @staticmethod
    def _calculate_mean_squared_error(outputs: np.ndarray, targets: np.ndarray) -> float:
        """Calculate mean squared error for output vector and desired outputs for that vector.

        Args:
            outputs (list[float]): vector which contains all the outputs given by output layer
            targets (list[float]): vector which contains all the desired outputs from last layer

        Returns:
            float: Mean squared error for given vector.
        """
        return np.sum((targets - outputs) ** 2) * 0.5

    @staticmethod
    def _init_training_progress_dict():
        return {
            "e": [],
            "q_for_epoch": [],
            "q_for_test_set": [],
            "started_at": [],
            "finished_at": [],
            "total_time": []
        }

    @staticmethod
    def _report_training_progress_to_excel_file(training_progress: dict) -> None:
        """Save information about training progress to excel file.

        Args:
            training_progress (dict): dictionary which stores information
            about each processed epoch
        """
        df = pd.DataFrame(training_progress)
        df.to_excel("report_from_training.xlsx", index=False)

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
