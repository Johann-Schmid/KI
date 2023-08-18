import time
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class SimpleNeuralNetwork:
    def __init__(self):
        self.inputLayerNeurons, self.hiddenLayerNeurons, self.outputLayerNeurons = 3, 3, 1

        # Zufällige Gewichtung und Bias-Initialisierung
        self.hidden_weights = np.random.uniform(size=(self.inputLayerNeurons, self.hiddenLayerNeurons))
        self.hidden_bias = np.random.uniform(size=(1, self.hiddenLayerNeurons))
        self.output_weights = np.random.uniform(size=(self.hiddenLayerNeurons, self.outputLayerNeurons))
        self.output_bias = np.random.uniform(size=(1, self.outputLayerNeurons))

    def train(self, inputs, expected_output, epochs=10000, lr=0.1):
        for _ in range(epochs):
            # Forward Pass
            hidden_layer_activation = np.dot(inputs, self.hidden_weights)
            hidden_layer_activation += self.hidden_bias
            hidden_layer_output = sigmoid(hidden_layer_activation)

            output_layer_activation = np.dot(hidden_layer_output, self.output_weights)
            output_layer_activation += self.output_bias
            predicted_output = sigmoid(output_layer_activation)

            # Backpropagation
            error = expected_output - predicted_output
            d_predicted_output = error * sigmoid_derivative(predicted_output)

            error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
            d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

            # Aktualisieren der Gewichtungen und Bias
            self.output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
            self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
            self.hidden_weights += inputs.T.dot(d_hidden_layer) * lr
            self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    def predict(self, input_data):
        hidden_layer_activation = np.dot(input_data, self.hidden_weights)
        hidden_layer_activation += self.hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.output_weights)
        output_layer_activation += self.output_bias
        predicted_output = sigmoid(output_layer_activation)
        return predicted_output


# Trainingsdaten
inputs = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

expected_output = np.array([
    [0],
    [1],
    [1],
    [0],
    [1],
    [0],
    [0],
    [1]
])

# Train the neural network
nn = SimpleNeuralNetwork()
nn.train(inputs, expected_output)

def get_prediction(input_combination):
    value = nn.predict(np.array([input_combination]))
    return value

# Test the prediction again
start = time.time()
result = get_prediction([0, 0, 0])
print(result)
end = time.time()
print(f"Prediction took {end - start} seconds")

