class TrainingData:
    def __init__(self, inputs: list[float], desired_outputs: list[float]) -> None:
        self.inputs = inputs
        self.desired_outputs = desired_outputs