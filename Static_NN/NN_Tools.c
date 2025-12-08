#include "Matrix_maths.c"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

typedef struct ActivationFunction
{
    double (*func)(double);
    double (*deriv)(double);
} ActivationFunction;

typedef struct NN_Layer
{
    int neurons;                   // Number of neurons in the layer
    Matrix weights;                // Weight matrix for the layer
    Matrix biases;                 // Bias vector for the layer
    Matrix activated_values;       // Activated values of the layer
    Matrix biased_weighted_sums;   // Weighted sums plus biases
    Matrix deltas;                 // Delta vector for backpropagation
    ActivationFunction activation; // Activation function and its derivative
} NN_Layer;

typedef struct Neural_Network
{
    int num_layers;   // Number of layers in the neural network
    NN_Layer *layers; // Array of layers
} Neural_Network;

NN_Layer NN_create_layer(
    int in_neurons,
    int neurons,
    double *weight_buffer,
    double *bias_buffer,
    double *avalues_buffer,
    double *bws_buffer,
    double *deltas_buffer,
    ActivationFunction activation)
{

    for (int i = 0; i < in_neurons * neurons; i++)
    {
        weight_buffer[i] = (double)rand() / (double)RAND_MAX;
        ; // Initialize weights to zero
    }

    for (int i = 0; i < neurons; i++)
    {
        bias_buffer[i] = 0.0; // Initialize biases to zero
    }

    NN_Layer layer;
    layer.neurons = neurons;
    layer.activated_values = matrix_create(neurons, 1, avalues_buffer);

    layer.weights = matrix_create(neurons, in_neurons, weight_buffer);
    layer.biases = matrix_create(neurons, 1, bias_buffer);

    layer.biased_weighted_sums = matrix_create(neurons, 1, bws_buffer);
    layer.deltas = matrix_create(neurons, 1, deltas_buffer);
    layer.activation = activation;

    return layer;
}

Neural_Network NN_create(int num_layers, NN_Layer *layers)
{
    Neural_Network net;
    net.num_layers = num_layers;
    net.layers = layers;
    return net;
}

NN_Layer NN_forward(NN_Layer layer, Matrix inputs)
{
    assert(inputs.rows == layer.weights.cols);
    assert(inputs.cols == 1);
    assert(layer.weights.rows == layer.neurons);

    double temp_buffer[layer.neurons];                                                     // Temporary buffer for matrix multiplication
    Matrix weighted_sum = matrix_multiplication(layer.weights, inputs, temp_buffer);       // Add weights * inputs
    layer.biased_weighted_sums = matrix_addition(weighted_sum, layer.biases, temp_buffer); // Add biases and store in layer

    return layer;
}

Neural_Network NN_forward_pass(const Neural_Network net, Matrix inputs)
{
    for (int i = 0; i < net.num_layers; i++)
    {
        NN_Layer layer = net.layers[i];
        // Propagate through each layer
        NN_forward(layer, inputs);

        // Apply activation function
        for (int r = 0; r < layer.neurons; r++)
        {
            double val = matrix_get_entry(layer.biased_weighted_sums, r, 0);
            double activated_val = layer.activation.func(val);
            matrix_set_entry(layer.activated_values, r, 0, activated_val);
        }
    }
    return net;
}

ActivationFunction ReLU_Activation = {ReLU, ReLU_1};
ActivationFunction Linear_Activation = {Linear, Linear_1};

double ReLU(double x)
{
    return (x > 0) ? x : 0;
}
double ReLU_1(double x)
{
    return (x > 0) ? 1 : 0;
}

// double Sigmoid(double x) {
//     return 1.0 / (1.0 + exp(-x));
// }
// double Sigmoid_1(double x) {
//     double sig = Sigmoid(x);
//     return sig * (1 - sig);
// }

// double Tanh(double x) {
//     return tanh(x);
// }
// double Tanh_1(double x) {
//     return (1 / cosh(x))*(1 / cosh(x));
// }

double Linear(double x)
{
    return x;
}
double Linear_1(double x)
{
    return 1;
}

NN_Layer NN_output_delta(const NN_Layer layer, const Matrix target)
{
    for (int i = 0; i < layer.neurons; i++)
    {
        double output = matrix_get_entry(layer.biased_weighted_sums, i, 0);
        double output_activated = matrix_get_entry(layer.activated_values, i, 0);
        double delta = (output_activated - matrix_get_entry(target, i, 0)) * layer.activation.deriv(output);
        matrix_set_entry(layer.deltas, i, 0, matrix_get_entry(layer.deltas, i, 0) + delta);
    }
    return layer;
}

NN_Layer NN_hidden_delta(const NN_Layer layer, const NN_Layer next_layer)
{
    for (int i = 0; i < layer.neurons; i++)
    {
        // Accumulate delta from next layer
        double sum = 0.0;
        for (int j = 0; j < next_layer.neurons; j++)
        {
            double next_delta = matrix_get_entry(next_layer.deltas, j, 0);
            double weight = matrix_get_entry(next_layer.weights, j, i);
            sum += next_delta * weight;
        }

        double output = matrix_get_entry(layer.biased_weighted_sums, i, 0);
        double output_activated = matrix_get_entry(layer.activated_values, i, 0);
        double delta = sum * layer.activation.deriv(output);
        matrix_set_entry(layer.deltas, i, 0, matrix_get_entry(layer.deltas, i, 0) + delta);
    }
    return layer;
}

Neural_Network NN_backward_pass(Neural_Network net, const Matrix target)
{
    for (int i = net.num_layers; i > 0; i--)
    {
        NN_Layer layer = net.layers[i];
        NN_Layer next_layer = net.layers[i + 1];

        if (i == net.num_layers)
        {
            NN_output_delta(layer, target);
        }
        else
        {
            NN_hidden_delta(layer, next_layer);
        }
    }
    return net;
}
// Write a function to update weights and biases based on deltas and learning rate that uses matrix operations

Neural_Network NN_update_weights(Neural_Network net, double learning_rate)
{
    for (int i = 0; i < net.num_layers; i++)
    {
        NN_Layer layer = net.layers[i];
    }
    return net;
}

double NN_MSE(Neural_Network net, const Matrix target)
{
    double error = 0;
    int outneurons = net.layers[net.num_layers - 1].outneurons;
    for (int i = 0; i < outneurons; i++)
    {
        double predicted = matrix_get_entry(net.layers[net.num_layers - 1].outputs, i, 0);
        double actual = matrix_get_entry(target, i, 0);
        error += (predicted - actual) * (predicted - actual);
    }
    return error / outneurons;
};