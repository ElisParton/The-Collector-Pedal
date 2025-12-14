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

NN_Layer NN_forward(NN_Layer layer, Matrix inputs)
{
    assert(inputs.rows == layer.weights.cols);
    assert(inputs.cols == 1);
    assert(layer.weights.rows == layer.neurons);

    double temp_buffer1[layer.neurons];
    double temp_buffer2[layer.neurons];                                                                           // Temporary buffer for matrix multiplication
    Matrix weighted_sum = matrix_multiplication(layer.weights, inputs, temp_buffer1);                             // Add weights * inputs
    layer.biased_weighted_sums = matrix_addition(weighted_sum, layer.biases, layer.biased_weighted_sums.entries); // Add biases and store in layer

    return layer;
}

Neural_Network NN_forward_pass(Neural_Network net)
{
    Matrix inputs = net.input_layer.inputs;
    for (int i = 0; i < net.num_layers; i++)
    {
        NN_Layer layer = net.layers[i];
        // Propagate through each layer
        net.layers[i] = NN_forward(layer, inputs);

        // Apply activation function
        matrix_map_to(layer.biased_weighted_sums, layer.activation.func, layer.activated_values.entries);

        // Set inputs for next layer
        inputs = layer.activated_values;
    }
    return net;
}

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

ActivationFunction ReLU_Activation = {ReLU, ReLU_1};
ActivationFunction Linear_Activation = {Linear, Linear_1};

NN_Layer NN_output_delta(const NN_Layer layer, const Matrix target)
{
    double temp_buffer[layer.neurons];
    double temp_buffer2[layer.neurons];
    matrix_set_all(layer.deltas, 0.0);

    Matrix output = layer.biased_weighted_sums;
    Matrix output_activated = layer.activated_values;

    Matrix deltas = layer.deltas;

    Matrix difference = matrix_subtraction(output_activated, target, temp_buffer);
    Matrix output_derivd = matrix_map_to(output, layer.activation.deriv, temp_buffer2);
    deltas = matrix_hadamard_product(difference, output_derivd, layer.deltas.entries);
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
    for (int i = net.num_layers - 1; i >= 0; i--)
    {
        NN_Layer layer = net.layers[i];

        if (i == net.num_layers - 1)
        {
            net.layers[i] = NN_output_delta(layer, target);
        }
        else
        {
            NN_Layer next_layer = net.layers[i + 1];
            net.layers[i] = NN_hidden_delta(layer, next_layer);
        }
    }
    return net;
}
// Write a function to update weights and biases based on deltas and learning rate that uses matrix operations
Neural_Network NN_update_weights(Neural_Network net, double learning_rate)
{
    for (int i = 0; i < net.num_layers; ++i)
    {
        // get pointer to layer so we modify the real structure, not a copy
        NN_Layer *layer = &net.layers[i];

        // previous activations: input layer (a0) or previous layer's activations
        Matrix a_prev = (i == 0) ? net.input_layer.inputs : net.layers[i - 1].activated_values;

        // sanity checks (remove if your matrix API already guarantees these)
        assert(a_prev.cols == 1);
        assert(a_prev.rows == layer->weights.cols); // #inputs == weight columns
        assert(layer->weights.rows == layer->neurons);

        for (int r = 0; r < layer->neurons; ++r)
        {
            double delta = matrix_get_entry(layer->deltas, r, 0);

            // weight update: for each input column c
            for (int c = 0; c < a_prev.rows; ++c)
            {
                double a_val = matrix_get_entry(a_prev, c, 0);
                double w = matrix_get_entry(layer->weights, r, c);
                double new_w = w - learning_rate * delta * a_val;
                matrix_set_entry(layer->weights, r, c, new_w);
            }

            // bias update
            double b = matrix_get_entry(layer->biases, r, 0);
            double new_b = b - learning_rate * delta;
            matrix_set_entry(layer->biases, r, 0, new_b);
        }
    }

    return net;
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