from neural_network import NeuralNetwork
from test_data import TestData
from training_data import TrainingData
from tensorflow.keras.datasets import mnist
from data_preparing import convert_training_samples_to_learning_set, convert_test_samples_to_test_set

# data set params
number_of_samples_for_training: int = 60_000
number_of_samples_for_test: int = int(number_of_samples_for_training * 0.1)
image_width: int = 28
image_height: int = 28
number_of_inputs: int = image_width * image_height
number_of_outputs: int = 10
((x_train, y_train), (x_test, y_test)) = mnist.load_data()

learning_set: list[TrainingData] = convert_training_samples_to_learning_set(
    number_of_samples_for_training, number_of_outputs,
    image_width, image_height, x_train, y_train)
test_set: list[TestData] = convert_test_samples_to_test_set(
    number_of_samples_for_test, number_of_outputs,
    image_width, image_height, x_test, y_test)

# learning params
number_of_neurons_in_each_hidden_layer = []
learning_rate = 0.005
number_of_epoch = 200
t_max = number_of_samples_for_training * number_of_epoch
q_min = 50

# neural_network = NeuralNetwork()
# neural_network.setup_structure(
#     number_of_inputs,
#     number_of_outputs,
#     number_of_neurons_in_each_hidden_layer)
neural_network = NeuralNetwork.read_from_file("C:\\Users\\Jakub\\Desktop\\network_6.txt")
# neural_network.setup_learning_params(
#     learning_rate,
#     t_max,
#     q_min)
# neural_network.use_xavier_weight_initialization()
# neural_network.init_weights()
# neural_network.enable_test_set_evaluation_after_each_epoch(test_set)
# neural_network.enable_logging_training_progress_to_console()
# neural_network.enable_reporting_training_progress_to_excel_file()

# learning_result = neural_network.learn(learning_set)

# neural_network.save_to_file("C:\\Users\\Jakub\\Desktop\\network_6.txt")

# tolerance = 0.2
correct_answers_total = 0
for sample_index in range(number_of_samples_for_test):
    as_text = f"d: {test_set[sample_index].label}, y:"
    outputs = neural_network.process(test_set[sample_index].inputs)
    print(outputs)
    correct_outputs = 0
    for output_index in range(number_of_outputs):
        index_of_output_that_should_be_active = int(test_set[sample_index].label)
        if (output_index != index_of_output_that_should_be_active and
                outputs[output_index] < 0.5):
            correct_outputs += 1
        elif output_index == index_of_output_that_should_be_active and outputs[output_index] > 0.5:
            correct_outputs += 1
        if outputs[output_index] > 0.5:
            as_text += f" {output_index}, "
    print(f"{as_text}")
    if correct_outputs == 10:
        correct_answers_total += 1

print(f"Correct answers: {correct_answers_total}/{len(test_set)} -> {(correct_answers_total/len(test_set))*100}%")



