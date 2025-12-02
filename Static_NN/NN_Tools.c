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

NN_Layer NN_create_layer(int neurons, int outneurons, double* input_buffer, double* weight_buffer, double* bias_buffer, double* output_buffer, double* deltas_buffer,double (*activation)(double), double (*activation_derivative)(double)) {
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
    layer.activation = activation;
    layer.activation_derivative = activation_derivative;
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

Neural_Network NN_forward_pass(const Neural_Network net) {
    Neural_Network current_net = net;
    for (int i = 0; i < net.num_layers; i++) {
        current_net.layers[i] = NN_forward(current_net.layers[i]);
        if (i < net.num_layers - 1) {
            for (int r = 0; r < current_net.layers[i].outneurons; r++) {
                double activated_value = current_net.layers[i].activation(matrix_get_entry(current_net.layers[i].outputs, r, 0));
                matrix_set_entry(current_net.layers[i + 1].inputs, r, 0, activated_value);
            }
        }
    }
    return current_net;
}

double ReLU(double x) {
    return (x > 0) ? x : 0;
}
double ReLU_1(double x) {
    return (x > 0) ? 1 : 0;
}

double Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double Sigmoid_1(double x) {
    double sig = Sigmoid(x);
    return sig * (1 - sig);
}

double Tanh(double x) {
    return tanh(x);
}
double Tanh_1(double x) {
    return (1 / cosh(x))*(1 / cosh(x));
}

double Linear(double x) {
    return x;
}
double Linear_1(double x) {
    return 1;
}

NN_Layer NN_output_delta(const NN_Layer layer, const Matrix target) {
    for (int i = 0; i < layer.outneurons; i++) {
        double output = matrix_get_entry(layer.outputs, i, 0);
        double output_activated = layer.activation(output);
        double delta = (output_activated - matrix_get_entry(target, i, 0)) * layer.activation_derivative(output);
        matrix_set_entry(layer.deltas, i, 0, delta);
    }
    return layer;
}

NN_Layer NN_hidden_delta(const NN_Layer layer, const NN_Layer next_layer) {
    for (int i = 0; i < layer.outneurons; i++) {
        double sum = 0.0;
        for (int j = 0; j < next_layer.outneurons; j++) {
            double next_delta = matrix_get_entry(next_layer.deltas, j, 0);
            double weight = matrix_get_entry(next_layer.weights, j, i);
            sum += next_delta * weight;
        }
        double output = matrix_get_entry(layer.outputs, i, 0);
        double output_activated = layer.activation(output);
        double delta = sum * layer.activation_derivative(output);
        matrix_set_entry(layer.deltas, i, 0, delta);
    }
    return layer;
}

Neural_Network NN_backward_pass(Neural_Network net, const Matrix target) {
    Neural_Network current_net = net;
    for (int i = net.num_layers - 1; i >= 0; i--) {
        if (i == net.num_layers - 1) {
            current_net.layers[i] = NN_output_delta(current_net.layers[i], target);
        } else {
            current_net.layers[i] = NN_hidden_delta(current_net.layers[i], current_net.layers[i + 1]);
        }
    }
    return current_net;
}

Neural_Network NN_update_weights(Neural_Network net, double learning_rate) {
    Neural_Network current_net = net;
    for (int i = 0; i < net.num_layers; i++) {
        NN_Layer layer = current_net.layers[i];
        for (int r = 0; r < layer.outneurons; r++) {
            for (int c = 0; c < layer.neurons; c++) {
                double input = matrix_get_entry(layer.inputs, c, 0);
                double delta = matrix_get_entry(layer.deltas, r, 0);
                double weight = matrix_get_entry(layer.weights, r, c);
                double new_weight = weight - learning_rate * delta * input;
                matrix_set_entry(layer.weights, r, c, new_weight);
            }
            double bias = matrix_get_entry(layer.biases, r, 0);
            double delta = matrix_get_entry(layer.deltas, r, 0);
            double new_bias = bias - learning_rate * delta;
            matrix_set_entry(layer.biases, r, 0, new_bias);
        }
        current_net.layers[i] = layer;
    }
    return current_net;
}