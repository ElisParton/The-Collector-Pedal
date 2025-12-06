#include "NN_Tools.c"
#include <stdlib.h>

int main(/*int argc, char *argv[]*/) {

    double test_input [10] = {
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9
    };
    double test_output [10] = {
        0.9, 0.8, 0.7, 0.6, 0.5,
        0.4, 0.3, 0.2, 0.1, 0.0
    };
    /*
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int c = atoi(argv[3]);
    int netconfig[3] = {a, b, c};
    */
    int a = 10;
    int b = 5;
    int c = 10;

    int netconfig[3] = {a, b, c};
    double inp_buffer1[netconfig[0]];
    double weight_buffer1[netconfig[0]*netconfig[1]];
    double bias_buffer1[netconfig[1]];
    double out_buffer1[netconfig[1]];
    double deltas_buffer1[netconfig[1]];

    double inp_buffer2[netconfig[1]];
    double weight_buffer2[netconfig[1]*netconfig[2]];
    double bias_buffer2[netconfig[2]];
    double out_buffer2[netconfig[2]];
    double deltas_buffer2[netconfig[2]];

    NN_Layer layer1 = NN_create_layer(a, b,
        inp_buffer1,
        weight_buffer1,
        bias_buffer1,
        out_buffer1,
        deltas_buffer1,
        Linear,
        Linear_1
    );

    NN_Layer layer2 = NN_create_layer(b, c,
        inp_buffer2,
        weight_buffer2,
        bias_buffer2,
        out_buffer2,
        deltas_buffer2,
        Linear,
        Linear_1
    );

    NN_Layer layers[2] = {layer1, layer2};
    Neural_Network net = NN_create(2, layers);
    net.layers[0].inputs = matrix_create(a, 1, test_input);
    NN_forward_pass(net);
    matrix_print(net.layers[1].outputs);
    for (int epoch = 0; epoch < 200; epoch++) {
        NN_backward_pass(net, matrix_create(c, 1, test_output));
        NN_update_weights(net, 0.01);
        NN_forward_pass(net);
        matrix_print(net.layers[1].outputs);
    }
    matrix_print(net.layers[0].weights);
    matrix_print(net.layers[0].biases);
    matrix_print(net.layers[1].weights);
    matrix_print(net.layers[1].biases);
    return 0;
}