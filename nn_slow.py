from PIL import Image
import math 
import random
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

class Node:
    def __init__(self, no_input):
        self.val = None
        self.bias_weight = None 
        self.af = None
        self.weights = [random.randint(0,1) for i in range(no_input)]
        self.delta = None
    
    def set_val(self, val):
        # Set value for input neuron
        self.val = val

    def set_bias_weight(self, bias_weight):
        # Set bias weight
        self.bias_weight = bias_weight
    
    def set_activation_function(self, activation_function):
        # Set activation function 
        self.af = activation_function
    
    def set_delta(self, delta):
        # Set delta value for backpropagation
        self.delta = delta
    
    def set_weights(self, weights):
        # Set a ist of weights from connections
        # coming into neuron
        self.weights = weights
    
    def print_node(self):
        # Print single node for debugging
        string =  "Value: " + str(self.val) + ", Bias: " + str(self.bias_weight) + ", Activation Function: " + str(self.af) + ", Node: " + str(self.weights)
        return string

class NN:
    def __init__(self, structure, bias_activation, rate):
        # structure e.g. [1, 3, 1]
        # Each layer is a list of nodes
        self.layers = self.construct_layers(structure, bias_activation)
        # List of output values to comapre with desired list for backprop
        self.output = self.get_output()
        # Desired values correspond to each output neuron and
        # are initialised just before backpropagation
        self.desired = None
        self.rate = rate

    def set_inputs_in_layer(self, inputs):
        # set values for input layer only
        input_layer = self.layers[0]
        for i in range(len(input_layer)):
            input_layer[i].set_val(inputs[i])

    def foward_pass_layer(self, layer_index):
        # Forward pass on a single layer
        if layer_index == 0:
            # Cannot carry this out on input layer
            return
        input_nodes = self.layers[layer_index-1]
        for output in self.layers[layer_index]:
            sum = 0
            w_index = 0
            for input in input_nodes:
                sum += input.val * output.weights[w_index]
                w_index += 1
            output.set_val(output.af(sum  + output.bias_weight))
    
    def foward_pass_nn(self, inputs):
        # Iterates through the whole nn to carry
        # out forward pass per layer
        for i in range(len(self.layers)):
            # Carry out forward pass on one layer to update output val
            if i == 0:
                self.set_inputs_in_layer(inputs)
            else:
                self.foward_pass_layer(i)
    
    def construct_layers(self, structure, bias_activation):
        # Builds the nn with specificed structure, bias and activation functions
        # Takes in bias/activation each as a list of lists
        # eg for 1-3-1 structure [[[b11,a11], [b12,a12], [b13,a13]], [[b21,a21]]] 
        layers = []
        for i in range(len(structure)):
            if i == 0:
                # Input layer nodes have 0 inputs
                nodes = [Node(0) for x in range(structure[i])]
            else:
                nodes = [ Node(structure[i-1]) for x in range(structure[i])]

            # Set biases and activation functions
            if i != 0:
                bias_activation_pair = bias_activation[i-1]
                for i in range(len(nodes)):
                    nodes[i].set_bias_weight(bias_activation_pair[i][0])
                    nodes[i].set_activation_function(bias_activation_pair[i][1])
    
            layers.append(nodes)
        return layers

    def print_layer(self, layer_index):
        # Print a single layer for debugging
        for node in self.layers[layer_index]:
            print ("[" + node.print_node() + "]")
    
    def print_nn(self):
        # Print the whole nn for debugging
        for i in range(len(self.layers)):
            print ("**********")
            print ("LAYER")
            print ("**********")
            self.print_layer(i)

    def get_output(self):
        # Go through every output neuron from
        # current nn to collect list of outputs
        output_vals = []
        for output in self.layers[-1]:
            output_vals.append(output.val)
    
    def set_desired(self, desired):
        # Set a list of designed outputs from nn
        # before backpropagation
        self.desired = desired
    
    def calc_output_delta(self):
        # Get delta values for neurons in output layer
        output_neurons = self.layers[-1]
        for i in range(len(output_neurons)):
            output = output_neurons[i]
            af = output.af
            delta = (self.desired[0][i]-output.val)*derivative(output.val,af)
            output.set_delta(delta)
    
    def update_weights(self, layer_index):
        # Update weights in each layer for backprop
        front_neurons = self.layers[layer_index]
        back_neurons = self.layers[layer_index-1]
        for i in range(len(front_neurons)):
            for j in range(len(back_neurons)):
                neuron = front_neurons[i]
                w_old = neuron.weights[j]
                prev_neuron_val = back_neurons[j].val
                neuron.weights[j] = w_old + self.rate*neuron.delta*prev_neuron_val
    
    def calc_hidden_delta(self, layer_index):
        # Calculate delta values for neurons in any layer
        # apart from output layer.
        output_neurons = self.layers[layer_index+1] 
        curr_neurons = self.layers[layer_index]
        for i in range(len(curr_neurons)):
            curr_neuron = curr_neurons[i]
            sum_delta_weights = 0
            for j in range(len(output_neurons)):
                neuron = output_neurons[j]
                sum_delta_weights += neuron.delta*neuron.weights[i]
            curr_neuron.delta = sum_delta_weights*derivative(curr_neuron.val, curr_neuron.af)
    
    def update_bias_weight(self, layer_index):
        # Update bias weights per layer in each layer for backprop
        curr_neurons = self.layers[layer_index]
        for i in range(len(curr_neurons)):
            neuron = curr_neurons[i]
            old_bias_weight = neuron.bias_weight
            neuron.bias_weight = old_bias_weight + self.rate*neuron.delta


    def backpropagation(self):
        self.calc_output_delta()
        self.update_weights(-1)
        self.update_bias_weight(-1)
        for i in range(len(self.layers)-2, 0, -1):
            self.calc_hidden_delta(i)
            self.update_weights(i)
            self.update_bias_weight(i)
        
# Activation functions

def sigmoid(input):
    return 1/(1 + math.e**(-input))

def tanh(input):
    return math.tanh(input)

def linear(input):
    return input

def relu(input):
    return max(0, input)

def leaky_relu(input):
    return max(0.01*input, input)

def derivative(input, function):
    if function == sigmoid:
        return (1-input)*input
    elif function == tanh:
        return 1-input**2
    elif function == linear:
        return 1
    elif function == relu:
        if input < 0:
            return 0
        else:
            return 1
    elif function == leaky_relu:
        if input < 0:
            return 0.01
        else:
            return 1

def construct_nn():
    structure = [784, 25, 10, 24]
    bias_activation = []
    for i in range(len(structure)):
        if i != 0:
            ba = [[random.randint(0,1),tanh] for _ in range(structure[i])]
            bias_activation.append(ba)
    neural_network = NN(structure, bias_activation, 0.01)
    return neural_network

def single_epoch(neural_network, x, desired):
    for i in range(len(x)):
        neural_network.foward_pass_nn(x[i])
        neural_network.set_desired([desired[i]])
        neural_network.backpropagation()
    return neural_network

def get_test_output(neural_network, x):
    y = []
    for i in range(len(x)):
        inner_list = []
        neural_network.foward_pass_nn(list(x[i]))
        for j in range(len(neural_network.layers[-1])):
            inner_list.append(neural_network.layers[-1][j].val)
        y.append(inner_list)    
    return y

def set_dataframe(df):
    labels_df = sorted(list(df.label.unique()))
    normalize_func = np.vectorize(lambda t: t ** 1/255)
    x_data_norm = []
    for i in range(len(df)):
        row = normalize_func(np.array(df.iloc[i][1:]))
        x_data_norm.append(row)
    one_hot_titles = []
    for i in range(len(labels_df)):
        category = [0.2 if i != j else 0.8 for j in range(len(labels_df))]
        one_hot_titles.append(category)
    labels_organized = []
    for i in range(len(df)):
        nb = df.iloc[i][0]
        labels_organized.append(one_hot_titles[nb-1])
    return x_data_norm, labels_organized


neural_network = construct_nn()

data = pd.read_csv("dataset\sign_mnist_test.csv", sep=";", encoding="utf-8")
df = pd.DataFrame(data)
train_features, train_labels = set_dataframe(df)

no_epochs = 100
for i in range(no_epochs):
    nn_trained = single_epoch(neural_network, train_features, train_labels)
    print("ended epoch")

y = get_test_output(neural_network, [train_features[0]])
print(y)

