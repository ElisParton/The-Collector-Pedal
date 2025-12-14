#include "../NN_Tools.c"

double times10(double x)
{
    return x * 10.0;
}

int main()
{
    int a = 1;
    int b = 1;
    int netconfig[2] = {a, b};
    double inp_buffer1[a];
    double weights_buffer1[a * b];
    double bias_buffer1[b];
    double avalues_buffer1[b];
    double bws_buffer1[b];
    double deltas_buffer1[b];

    NN_Input_Layer input_layer = {
        .neurons = netconfig[0],
        .inputs = matrix_create(netconfig[0], 1, inp_buffer1)};

    NN_Layer layer1 = NN_create_layer(a, b,
                                      weights_buffer1,
                                      bias_buffer1,
                                      avalues_buffer1,
                                      bws_buffer1,
                                      deltas_buffer1,
                                      Linear_Activation);

    NN_Layer layers[1] = {layer1};
    Neural_Network net = NN_create(1, input_layer, layers);

    double target_buf[] = {5};
    Matrix target = matrix_create(1, 1, target_buf);
    net.input_layer.inputs.entries[0] = 5.0;
    for (int i = 0; i < 100; i++)
    {
        net = NN_forward_pass(net);
        net = NN_backward_pass(net, target);
        net = NN_update_weights(net, 0.01);
    }
    return 0;
}