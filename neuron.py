import numpy as np

class NeuronWithThreshold:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def f_net(self, inputs):
        """Netzwerkeingabefunktion"""
        print("Netzwerkeingabefunktion: " + str(np.dot(inputs, self.weights) + self.bias))
        return np.dot(inputs, self.weights) + self.bias

    def f_activity(self, net):
        """Aktivierungsfunktion (Sigmoid)"""
        print ("Aktivierungsfunktion: " + str(1 / (1 + np.exp(-net))))
        return 1 / (1 + np.exp(-net))

    def f_out(self, inputs):
        """Ausgabefunktion mit Schwellwertfunktion"""
        activity = self.f_activity(self.f_net(inputs))
        return 0 if activity < 0.5 else 1

# Testen des Neurons mit Schwellwertfunktion
inputs = np.array([-0.7, -0.7])
neuron_threshold = NeuronWithThreshold([0.1, 0.8], 0.0)
neuron_output_threshold = neuron_threshold.f_out(inputs)
print("Schwellwertfunktion: " + str(neuron_output_threshold))
