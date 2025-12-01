#include "NN_Tools.c"
void main() {
    double inp_buffer1[4] = {1.0, 0.5, -1.5, 2.0};
    double weight_buffer1[12] = {
        0.2, -0.5, 0.1, 0.4,
        -0.3, 0.8, -0.6, 0.7,
        0.5, -0.2, 0.3, -0.1
    };
    double bias_buffer1[3] = {0.1, -0.2, 0.3};
    double out_buffer1[3];

    double inp_buffer2[3] = {0};
    double weight_buffer2[6] = {
        0.3, -0.7, 0.5,
        -0.4, 0.6, -0.2
    };
    double bias_buffer2[2] = {0.0, 0.1};
    double out_buffer2[2];
    NN_Layer layer1 = NN_create_layer(4, 3,
        inp_buffer1,
        weight_buffer1,
        bias_buffer1,
        out_buffer1);
    NN_Layer layer2 = NN_create_layer(3, 2,
        inp_buffer2,
        weight_buffer2,
        bias_buffer2,
        out_buffer2);
    NN_Layer layers[2] = {layer1, layer2};
    Neural_Network net = NN_create(2, layers);
    NN_forward_pass(net);
    matrix_print(net.layers[0].outputs);
    matrix_print(net.layers[1].outputs);
}