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

