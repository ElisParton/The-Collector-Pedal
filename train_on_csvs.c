#include <stdio.h>
#include <stdlib.h>
#include "ann.h"


//------------------------------
// main program start
//------------------------------
int main(int argc, char *argv[])
{
	real *data;
	int rows, stride;
	char *filename = (argc > 1) ? argv[1] : "data_pairs.csv";


	printf("Loading %s...", filename);
    CHECK_OK(ann_load_csv(filename, CSV_NO_HEADER, &data, &rows, &stride));
    puts("done.");
    printf("rows_in=%d stride=%d\n", rows, stride);


	PNetwork pnet = ann_make_network(OPT_ADAPT, LOSS_MSE);
	if (!pnet)
		return ERR_FAIL;

	PTensor x_train = tensor_create_from_array(rows, stride, data);
	if (!x_train)
		return ERR_FAIL;

	PTensor y_train = tensor_slice_cols(x_train, stride / 2);
	if (!y_train)
		return ERR_FAIL;

	// define our network
	CHECK_OK(ann_add_layer(pnet, stride / 2, LAYER_INPUT, ACTIVATION_NULL));
	CHECK_OK(ann_add_layer(pnet, 50, LAYER_HIDDEN, ACTIVATION_SIGMOID));
    CHECK_OK(ann_add_layer(pnet, 50, LAYER_HIDDEN, ACTIVATION_SIGMOID));
	CHECK_OK(ann_add_layer(pnet, stride / 2, LAYER_OUTPUT, ACTIVATION_NULL));
    pnet->batchSize = 64;          // e.g. mini-batch of 16
    ann_set_convergence(pnet, 0.0025);
	ann_train_network(pnet, x_train, y_train, x_train->rows);
	
	real *outputs = malloc(sizeof(real) * stride / 2);
	if (!outputs) {
		fprintf(stderr, "failed to allocate outputs buffer\n");
		ann_free_network(pnet);
		free(data);
		return ERR_FAIL;
	}
	CHECK_OK(ann_predict(pnet, &data[0], outputs));
    ann_save_network(pnet, "trained_network.ann");

	print_outputs(pnet);

	ann_free_network(pnet);

	free(outputs);
	free(data);
	return ERR_OK;
}