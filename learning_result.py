class LearningResult:
    def __init__(self, number_of_executed_iterations: int,
                 number_of_processed_epochs: int,
                 errors_for_each_epoch: list[float]):
        self.number_of_executed_iterations = number_of_executed_iterations
        self.number_of_processed_epochs = number_of_processed_epochs
        self.errors_for_each_epoch = errors_for_each_epoch

    def __str__(self):
        obj_as_text = (f"number of executed iterations: {self.number_of_executed_iterations}\n"
                       f"number of processed epochs: {self.number_of_processed_epochs}\n")
        for epoch_index in range(len(self.errors_for_each_epoch)):
            obj_as_text += f"Error for epoch {epoch_index}: {self.errors_for_each_epoch[epoch_index]}\n"
        return obj_as_text
