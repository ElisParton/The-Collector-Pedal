#include "../NN_Tools.c"
#define LEARNING_RATE 0.05
#define EPOCHS 1000

int main()
{
    enum
    {
        INP = 1,
        OUP = 1
    };

    double inp_buffer1[INP];
    double weights_buffer1[INP * OUP];
    double bias_buffer1[OUP];
    double avalues_buffer1[OUP];
    double bws_buffer1[OUP];
    double deltas_buffer1[OUP];

    NN_Input_Layer input_layer = {
        .neurons = INP,
        .inputs = matrix_create(INP, 1, inp_buffer1)};

    NN_Layer layer1 = NN_create_layer(INP, OUP,
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

    double error;
    for (int i = 0; i < EPOCHS; i++)
    {
        NN_forward_pass(net);
        NN_backward_pass(net, target);
        NN_update_weights(net, LEARNING_RATE);
        error = NN_MSE(net, target);
        printf("Error at iteration %d: %f\n", i, error);
    }
    return 0;
}