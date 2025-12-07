#include "../NN_Tools.c"
#include <stdlib.h>

int main(/*int argc, char *argv[]*/) {

    double test_input [1*10] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
    };
    double test_output [1*10] = {
        2, 4, 6, 8, 10,
        12, 14, 16, 18, 20
    };
    /*
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int c = atoi(argv[3]);
    int netconfig[3] = {a, b, c};
    */
    int a = 1;
    int b = 5;
    int c = 1;

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
    Matrix train_x = matrix_create(1, 10, test_input);
    Matrix train_y = matrix_create(1, 10, test_output);
    net.layers[0].inputs = matrix_create(a, 1, test_input);
    for (int epoch = 1; epoch <= 1000; epoch++) {
        double error = 0.0;
        for (int i = 0; i < train_x.cols; i++) {
            double target_buf[1];
            Matrix target = matrix_slice(train_y, 0, 1, i, i + 1, target_buf);
            net.layers[0].inputs = matrix_slice(train_x, 0, 1, i, i + 1, inp_buffer1);
            NN_forward_pass(net);
            NN_backward_pass(net, target);
            error += NN_MSE(net, target);
        }
        printf("Epoch %d, MSE: %f\n", epoch, error);
        NN_update_weights(net, 0.0001);
    }
    while (1) {
        double myNum;
        printf("Enter an integer to double by the trained NN(-1 for exit): ");
        scanf("%lf", &myNum);
        if (myNum == -1) {
            break;
        }
        inp_buffer1[0] = myNum;
        NN_forward_pass(net);
        printf("NN says: %lf\n", net.layers[net.num_layers - 1].outputs.entries[0]);
    }
    return 0;
}