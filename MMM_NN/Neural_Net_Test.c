#include "Neural_Net_Tools.c"
void main() {
    double inp_buffer[4] = {1.0, 0.5, -1.5, 2.0};
    double weight_buffer1[12] = {
        0.2, -0.5, 0.1, 0.4,
        -0.3, 0.8, -0.6, 0.7,
        0.5, -0.2, 0.3, -0.1
    };
    double bias_buffer1[3] = {0.1, -0.2, 0.3};
    double out_buffer1[3];
    Neural_Net_Layer layer1 = neural_net_create_layer(4, 3,
        inp_buffer,
        weight_buffer1,
        bias_buffer1,
        out_buffer1);

    neural_net_forward(layer1);
    matrix_print(layer1.outputs);
}