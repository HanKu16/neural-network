from numpy import ndarray
from test_data import TestData
from training_data import TrainingData

def convert_training_samples_to_learning_set(
        number_of_samples: int, number_of_outputs: int,
        image_width: int, image_height: int,
        x_train: ndarray, y_train: ndarray) -> list[TrainingData]:
    learning_set: list[TrainingData] = []
    for sample_index in range(number_of_samples):
        input_vector_from_sample: list[float] = []
        for row_index in range(image_width):
            for col_index in range(image_height):
                normalized_pixel_value: float = float(x_train[sample_index, row_index, col_index] / 255)
                input_vector_from_sample.append(normalized_pixel_value)
        indexes_of_desired_active_outputs: list[int] = [int(y_train[sample_index])]
        label: str = f"{y_train[sample_index]}"
        training_data = TrainingData(input_vector_from_sample, number_of_outputs,
                                     indexes_of_desired_active_outputs, label)
        learning_set.append(training_data)
    return learning_set

def convert_test_samples_to_test_set(
        number_of_samples: int, number_of_outputs: int,
        image_width: int, image_height: int,
        x_test: ndarray, y_test: ndarray) -> list[TestData]:
    test_set: list[TestData] = []
    for sample_index in range(number_of_samples):
        input_vector_from_sample: list[float] = []
        for row in range(image_width):
            for col in range(image_height):
                normalized_pixel_value: float = float(x_test[sample_index][row][col] / 255)
                input_vector_from_sample.append(normalized_pixel_value)
        indexes_of_desired_active_outputs: list[int] = [int(y_test[sample_index])]
        label: str = f"{y_test[sample_index]}"
        test_data = TestData(input_vector_from_sample, number_of_outputs,
                             indexes_of_desired_active_outputs, label)
        test_set.append(test_data)
    return test_set