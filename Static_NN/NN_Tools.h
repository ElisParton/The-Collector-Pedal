#ifndef NN_TOOLS_H
#define NN_TOOLS_H

#include "Matrix_maths.h"

/* =========================
   Activation Function
   ========================= */

typedef struct ActivationFunction
{
    double (*func)(double);
    double (*deriv)(double);
} ActivationFunction;

/* =========================
   Network Structures
   ========================= */

typedef struct NN_Layer
{
    int neurons;
    Matrix weights;
    Matrix biases;
    Matrix activated_values;
    Matrix biased_weighted_sums;
    Matrix deltas;
    ActivationFunction activation;
} NN_Layer;

typedef struct NN_Input_Layer
{
    int neurons;
    Matrix inputs;
} NN_Input_Layer;

typedef struct Neural_Network
{
    int num_layers;
    NN_Input_Layer input_layer;
    NN_Layer *layers;
} Neural_Network;

/* =========================
   Creation / Setup
   ========================= */

NN_Layer NN_create_layer(
    int in_neurons,
    int neurons,
    double *weight_buffer,
    double *bias_buffer,
    double *avalues_buffer,
    double *bws_buffer,
    double *deltas_buffer,
    ActivationFunction activation);

Neural_Network NN_create(
    int num_layers,
    NN_Input_Layer input,
    NN_Layer *layers);

/* =========================
   Forward Propagation
   ========================= */

void NN_forward(
    NN_Layer *layer,
    const Matrix *inputs);

void NN_forward_pass(
    Neural_Network net);

/* =========================
   Backpropagation
   ========================= */

void NN_output_delta(
    NN_Layer layer,
    const Matrix target);

void NN_hidden_delta(
    const NN_Layer layer,
    const NN_Layer next_layer);

void NN_backward_pass(
    Neural_Network net,
    const Matrix target);

/* =========================
   Training
   ========================= */

void NN_update_weights(
    Neural_Network net,
    double learning_rate);

/* =========================
   Metrics
   ========================= */

double NN_MSE(
    Neural_Network net,
    const Matrix target);

/* =========================
   Activation Functions
   ========================= */

double ReLU(double x);
double ReLU_1(double x);

double Linear(double x);
double Linear_1(double x);

extern ActivationFunction ReLU_Activation;
extern ActivationFunction Linear_Activation;

#endif /* NN_TOOLS_H */