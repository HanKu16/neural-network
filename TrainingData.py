class TrainingData:
    def __init__(self, inputs: list[float], number_of_outputs: int,
                 indexes_of_desired_active_outputs: list[int], label: str) -> None:
        self.inputs = inputs
        self.desired_outputs = self._set_desired_outputs(
            number_of_outputs, indexes_of_desired_active_outputs)
        self.label = label

    def _set_desired_outputs(self, number_of_outputs: int,
                             indexes_of_desired_active_outputs: list[int]):
        desired_outputs = []
        for output_index in range(number_of_outputs):
            if output_index in indexes_of_desired_active_outputs:
                desired_outputs.append(1)
            else:
                desired_outputs.append(0)
        return desired_outputs
    