#include "Matrix_maths.c"

typedef struct Neural_Net_Layer {
    int neurons; // Number of neurons in the layer
    int outneurons; // Number of neurons in the next layer
    Matrix inputs; // Input vector to the layer
    Matrix weights; // Weight matrix for the layer
    Matrix biases;  // Bias vector for the layer
    Matrix outputs; // Output vector for the layer
} Neural_Net_Layer;

typedef struct Neural_Net {
    int num_layers; // Number of layers in the neural network
    Neural_Net_Layer* layers; // Array of layers
} Neural_Net;

Neural_Net_Layer neural_net_create_layer(int neurons, int outneurons, double* input_buffer, double* weight_buffer, double* bias_buffer, double* output_buffer) {
    Neural_Net_Layer layer;
    layer.neurons = neurons;
    layer.outneurons = outneurons;
    layer.inputs = matrix_create(neurons, 1, input_buffer);
    layer.weights = matrix_create(outneurons, neurons, weight_buffer);
    layer.biases = matrix_create(outneurons, 1, bias_buffer);
    layer.outputs = matrix_create(outneurons, 1, output_buffer);
    return layer;
}

Neural_Net neural_net_create(int num_layers, Neural_Net_Layer* layers) {
    Neural_Net net;
    net.num_layers = num_layers;
    net.layers = layers;
    return net;
}

Neural_Net_Layer neural_net_forward(const Neural_Net_Layer layer) {
    double temp_buffer[layer.outneurons]; // Temporary buffer for matrix multiplication
    Matrix weighted_sum = matrix_multiplication(layer.weights, layer.inputs, temp_buffer);
    for (int i = 0; i < layer.outneurons; i++) {
        double sum = matrix_get_entry(weighted_sum, i, 0) + matrix_get_entry(layer.biases, i, 0);
        matrix_set_entry(layer.outputs, i, 0, sum); // No activation function applied
    }
    return layer;
}

Neural_Net neural_net_forward_pass(const Neural_Net net) {
    Neural_Net current_net = net;
    for (int i = 0; i < net.num_layers; i++) {
        current_net.layers[i] = neural_net_forward(current_net.layers[i]);
        if (i < net.num_layers - 1) {
            current_net.layers[i + 1].inputs = current_net.layers[i].outputs;
        }
    }
    return current_net;
}
