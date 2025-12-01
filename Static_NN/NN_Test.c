#include "NN_Tools.c"
#include <stdlib.h>

int main(int argc, char *argv[]) {

    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int c = atoi(argv[3]);
    int netconfig[3] = {a, b, c};

    double inp_buffer1[netconfig[0]];
    double weight_buffer1[netconfig[0]*netconfig[1]];
    double bias_buffer1[netconfig[1]];
    double out_buffer1[netconfig[1]];

    double inp_buffer2[netconfig[1]];
    double weight_buffer2[netconfig[1]*netconfig[2]];
    double bias_buffer2[netconfig[2]];
    double out_buffer2[netconfig[2]];

    NN_Layer layer1 = NN_create_layer(a, b,
        inp_buffer1,
        weight_buffer1,
        bias_buffer1,
        out_buffer1
    );

    NN_Layer layer2 = NN_create_layer(b, c,
        inp_buffer2,
        weight_buffer2,
        bias_buffer2,
        out_buffer2);

    NN_Layer layers[2] = {layer1, layer2};
    Neural_Network net = NN_create(2, layers);
    NN_forward_pass(net, NN_Linear);
    matrix_print(net.layers[0].weights);
    matrix_print(net.layers[1].weights);
    return 0;
}