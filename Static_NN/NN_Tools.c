#include "Matrix_maths.c"
#include <stdlib.h>
#include <math.h>

typedef struct NN_Layer {
    int neurons; // Number of neurons in the layer
    int outneurons; // Number of neurons in the next layer
    Matrix inputs; // Input vector to the layer
    Matrix weights; // Weight matrix for the layer
    Matrix biases;  // Bias vector for the layer
    Matrix outputs; // Output vector for the layer
    Matrix deltas;  // Delta vector for backpropagation
    double (*activation)(double); // Activation function pointer
    double (*activation_derivative)(double); // Derivative of activation function pointer
} NN_Layer;

typedef struct Neural_Network {
    int num_layers; // Number of layers in the neural network
    NN_Layer* layers; // Array of layers
} Neural_Network;

NN_Layer NN_create_layer(int neurons, int outneurons, double* input_buffer, double* weight_buffer, double* bias_buffer, double* output_buffer) {
    NN_Layer layer;
    layer.neurons = neurons;
    layer.outneurons = outneurons;
    for (int i = 0; i < neurons * outneurons; i++) {
        weight_buffer[i] = (double)rand() / (double)RAND_MAX;; // Initialize weights to zero
    }
    for (int i = 0; i < outneurons; i++) {
        bias_buffer[i] = 0.0; // Initialize biases to zero
    }
    layer.inputs = matrix_create(neurons, 1, input_buffer);
    layer.weights = matrix_create(outneurons, neurons, weight_buffer);
    layer.biases = matrix_create(outneurons, 1, bias_buffer);
    layer.outputs = matrix_create(outneurons, 1, output_buffer);
    return layer;
}

Neural_Network NN_create(int num_layers, NN_Layer* layers) {
    Neural_Network net;
    net.num_layers = num_layers;
    net.layers = layers;
    return net;
}

NN_Layer NN_forward(const NN_Layer layer) {
    double temp_buffer[layer.outneurons]; // Temporary buffer for matrix multiplication
    Matrix weighted_sum = matrix_multiplication(layer.weights, layer.inputs, temp_buffer);
    for (int i = 0; i < layer.outneurons; i++) {
        double sum = matrix_get_entry(weighted_sum, i, 0) + matrix_get_entry(layer.biases, i, 0);
        matrix_set_entry(layer.outputs, i, 0, sum);
    }
    return layer;
}

Neural_Network NN_forward_pass(const Neural_Network net, int (*activation_func)(double)) {
    Neural_Network current_net = net;
    for (int i = 0; i < net.num_layers; i++) {
        current_net.layers[i] = NN_forward(current_net.layers[i]);
        if (i < net.num_layers - 1) {
            for (int r = 0; r < current_net.layers[i].outneurons; r++) {
                double activated_value = activation_func(matrix_get_entry(current_net.layers[i].outputs, r, 0));
                matrix_set_entry(current_net.layers[i + 1].inputs, r, 0, activated_value);
            }
        }
    }
    return current_net;
}

double NN_ReLU(double x) {
    return (x > 0) ? x : 0;
}
double NN_Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double NN_Tanh(double x) {
    return tanh(x);
}
double NN_Linear(double x) {
    return x;
}

NN_Layer NN_output_backpropagation(const NN_Layer layer, const Matrix target, double learning_rate) {
    for (int i = 0; i < layer.outneurons; i++) {
        double output = matrix_get_entry(layer.outputs, i, 0);
        double output_activated = layer.activation(output);
        double delta = (output_activated - matrix_get_entry(target, i, 0)) * layer.activation_derivative(output);
        matrix_set_entry(layer.deltas, i, 0, delta);
    }

    //------------------
    // UNCHECKED CODE
    //------------------
    // Update weights and biases
    for (int r = 0; r < layer.outneurons; r++) {
        for (int c = 0; c < layer.neurons; c++) {
            double weight = matrix_get_entry(layer.weights, r, c);
            double input = matrix_get_entry(layer.inputs, c, 0);
            double delta = matrix_get_entry(layer.deltas, r, 0);
            weight += learning_rate * delta * input;
            matrix_set_entry(layer.weights, r, c, weight);
        }
        double bias = matrix_get_entry(layer.biases, r, 0);
        double delta = matrix_get_entry(layer.deltas, r, 0);
        bias += learning_rate * delta;
        matrix_set_entry(layer.biases, r, 0, bias);
    }
    return layer;
}