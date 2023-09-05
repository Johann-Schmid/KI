import numpy as np
import matplotlib.pyplot as plt


class NeuronLayer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        # Random initial weights and biases for each neuron in the layer
        self.weights = np.random.randn(num_neurons, num_inputs_per_neuron)
        self.biases = np.random.randn(num_neurons)

    def feedforward(self, inputs):
        """Feed the inputs through the neurons in the layer and get the outputs"""
        # Calculate the net input for each neuron
        net_inputs = np.dot(self.weights, inputs) + self.biases
        # Apply the activation function (sigmoid) to each net input
        return 1 / (1 + np.exp(-net_inputs))


class NeuralNetwork:
    def __init__(self):
        # Define the hidden layer with 3 neurons, each having 2 inputs
        self.hidden_layer = NeuronLayer(3, 2)
        # Define the output layer with 1 neuron, having 3 inputs (from the hidden layer)
        self.output_layer = NeuronLayer(1, 3)

    def feedforward(self, inputs):
        """Feed the inputs through the network and get the final output"""
        hidden_layer_outputs = self.hidden_layer.feedforward(inputs)
        return self.output_layer.feedforward(hidden_layer_outputs)


def plot_neural_network_final(layer_weights, layer_biases):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define the number of layers
    num_layers = len(layer_weights) + 1  # +1 for the input layer

    # Maximum number of neurons in layers
    max_neurons = max([len(b) for b in layer_biases] + [layer_weights[0].shape[1]])

    # List to store neuron positions
    neuron_positions = []

    # Plot neurons as circles
    circle_radius = 0.2

    weight_text_positions = []

    for layer in range(num_layers):
        if layer == 0:  # input layer
            num_neurons = layer_weights[0].shape[1]
        else:
            num_neurons = len(layer_biases[layer - 1])
        y_pos = np.linspace(0, max_neurons - 1, num_neurons)
        neuron_positions.append(y_pos)
        for y in y_pos:
            circle = plt.Circle((layer, y), circle_radius, color='blue' if layer == 0 else 'green', zorder=4)
            ax.add_patch(circle)

    # Plot connections and weights
    for layer in range(1, num_layers):
        for neuron in range(len(layer_biases[layer - 1])):
            for prev_neuron in range(layer_weights[layer - 1].shape[1]):
                weight = layer_weights[layer - 1][neuron, prev_neuron]
                color = 'red' if weight < 0 else 'black'
                ax.plot([layer - 1, layer],
                        [neuron_positions[layer - 1][prev_neuron], neuron_positions[layer][neuron]],
                        c=color,
                        zorder=2)
                # Add the weight value as text beside the line and slightly above
                midpoint = [(layer - 1 + layer) / 2,
                            (neuron_positions[layer - 1][prev_neuron] + neuron_positions[layer][neuron]) / 2]
                vertical_offset = -0.2 if weight >= 0 else 0.2
                # Adjust position if there's an overlap
                while (midpoint[0], midpoint[1] + vertical_offset) in weight_text_positions:
                    vertical_offset += 0.1 if weight < 0 else -0.1

                weight_text_positions.append((midpoint[0], midpoint[1] + vertical_offset))
                ax.text(midpoint[0], midpoint[1] + vertical_offset, f"{weight:.2f}", fontsize=10,
                        ha="center", va="center", zorder=5)

    # Set the aspect and the limits
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.set_ylim(-0.5, max_neurons - 0.5)
    ax.set_xticks(list(range(num_layers)))
    ax.set_xticklabels(['Input Layer', 'Hidden Layer', 'Output Layer'])
    ax.set_yticks([])
    ax.set_title("Neuronales Netzwerk")

    plt.show()


# Create and test the neural network
nn = NeuralNetwork()
nn_output = nn.feedforward([0.5, 0.25])

# Plot the neural network
plot_neural_network_final([nn.hidden_layer.weights, nn.output_layer.weights],
                          [nn.hidden_layer.biases, nn.output_layer.biases])
