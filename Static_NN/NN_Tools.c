#include "Matrix_maths.c"
#include <stdlib.h>
#include <math.h>

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

typedef struct NN_Input_Layer
{
    int neurons;   // Number of neurons
    Matrix inputs; // Input matrix
} NN_Input_Layer;

typedef struct Neural_Network
{
    int num_layers;             // Number of layers in the neural network
    NN_Input_Layer input_layer; // Input layer
    NN_Layer *layers;           // Array of layers
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

Neural_Network NN_create(int num_layers, NN_Input_Layer input, NN_Layer *layers)
{
    Neural_Network net;
    net.input_layer = input;
    net.num_layers = num_layers;
    net.layers = layers;
    return net;
}

void NN_forward(NN_Layer layer, const Matrix inputs)
{
    layer.biased_weighted_sums = matrix_addition(
        matrix_multiplication(layer.weights, inputs, layer.biased_weighted_sums.entries),
        layer.biases,
        layer.biased_weighted_sums.entries);
}

void NN_forward_pass(Neural_Network net, Matrix input)
{
    *net.input_layer.inputs.entries = *input.entries; // This might be expensive. Maybe figure out using pointers later.

    Matrix inputs = net.input_layer.inputs;
    for (int i = 0; i < net.num_layers; i++)
    {
        NN_Layer layer = net.layers[i];
        // Propagate through each layer
        NN_forward(layer, inputs);

        // Apply activation function
        matrix_apply_func(layer.biased_weighted_sums, layer.activation.func, layer.activated_values.entries);

        // Set inputs for next layer
        inputs = layer.activated_values;
    }
}

double ReLU(double x)
{
    return (x > 0) ? x : 0;
}
double ReLU_1(double x)
{
    return (x > 0) ? 1 : 0;
}

double Sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
double Sigmoid_1(double x)
{
    double sig = Sigmoid(x);
    return sig * (1 - sig);
}

double Tanh(double x)
{
    return tanh(x);
}
double Tanh_1(double x)
{
    return (1 / cosh(x)) * (1 / cosh(x));
}

double Linear(double x)
{
    return x;
}
double Linear_1(double x)
{
    return 1 + x - x; // Avoid compiler warning
}

ActivationFunction ReLU_Activation = {ReLU, ReLU_1};
ActivationFunction Linear_Activation = {Linear, Linear_1};
ActivationFunction Sigmoid_Activation = {Sigmoid, Sigmoid_1};
ActivationFunction Tanh_Activation = {Tanh, Tanh_1};

void NN_output_delta(NN_Layer layer, const Matrix target)
{
    double temp_buffer[layer.neurons];
    double temp_buffer2[layer.neurons];
    matrix_set_all(layer.deltas, 0.0);

    Matrix output = layer.biased_weighted_sums;
    Matrix output_activated = layer.activated_values;

    Matrix difference = matrix_subtraction(output_activated, target, temp_buffer);
    Matrix output_derivd = matrix_apply_func(output, layer.activation.deriv, temp_buffer2);
    layer.deltas = matrix_hadamard_product(difference, output_derivd, layer.deltas.entries);
}

void NN_hidden_delta(const NN_Layer layer, const NN_Layer next_layer)
{
    for (int i = 0; i < layer.neurons; i++)
    {
        // Accumulate delta from next layer
        double temp_buffer[next_layer.neurons];
        Matrix next_deltas = next_layer.deltas;
        Matrix next_weights = matrix_slice(next_layer.weights, 0, next_layer.neurons, i, i + 1, temp_buffer);
        double sum =
            matrix_grand_sum(
                matrix_multiplication(next_deltas, next_weights, temp_buffer));

        double output = matrix_get_entry(layer.biased_weighted_sums, i, 0);
        // double output_activated = matrix_get_entry(layer.activated_values, i, 0);
        double delta = sum * layer.activation.deriv(output);

        matrix_set_entry(layer.deltas, i, 0, delta);
    }
}

void NN_backward_pass(Neural_Network net, const Matrix target)
{
    for (int i = net.num_layers - 1; i >= 0; i--)
    {
        NN_Layer layer = net.layers[i];

        if (i == net.num_layers - 1)
        {
            NN_output_delta(layer, target);
        }
        else
        {
            NN_Layer next_layer = net.layers[i + 1];
            NN_hidden_delta(layer, next_layer);
        }
    }
}

void NN_update_weights(Neural_Network net, double learning_rate)
{
    for (int i = 0; i < net.num_layers; ++i)
    {
        Matrix prev_avals;
        int prev_neurons;
        NN_Layer layer = net.layers[i];
        if (i == 0)
        {
            prev_avals = net.input_layer.inputs;
            prev_neurons = net.input_layer.neurons;
        }
        else
        {
            prev_avals = net.layers[i - 1].activated_values;
            prev_neurons = net.layers[i - 1].neurons;
        }

        double temp_buffer[prev_neurons];

        double weights_change_buffer[layer.neurons * prev_neurons];
        Matrix weights_change = matrix_create(layer.neurons, prev_neurons, weights_change_buffer);

        // https://apxml.com/courses/introduction-to-neural-networks/chapter-4-backpropagation-gradient-descent/updating-weights-biases
        weights_change =
            matrix_scalar_product(
                matrix_multiplication(
                    layer.deltas,
                    matrix_transpose(prev_avals, temp_buffer),
                    weights_change_buffer),
                learning_rate,
                weights_change_buffer);

        layer.weights = matrix_subtraction(layer.weights, weights_change, layer.weights.entries);
    }
}

void NN_update_biases(Neural_Network net, double learning_rate)
{
    for (int i = 0; i < net.num_layers; ++i)
    {
        NN_Layer layer = net.layers[i];

        double biases_change_buffer[layer.neurons];
        Matrix biases_change = matrix_create(layer.neurons, 1, biases_change_buffer);

        biases_change =
            matrix_scalar_product(
                layer.deltas,
                learning_rate,
                biases_change_buffer);

        layer.biases = matrix_subtraction(layer.biases, biases_change, layer.biases.entries);
    }
}

double NN_MSE(Neural_Network net, const Matrix target)
{
    double error = 0;
    int outneurons = net.layers[net.num_layers - 1].neurons;
    for (int i = 0; i < outneurons; i++)
    {
        double predicted = matrix_get_entry(net.layers[net.num_layers - 1].activated_values, i, 0);
        double actual = matrix_get_entry(target, i, 0);
        error += (predicted - actual) * (predicted - actual);
    }
    return error / outneurons;
};