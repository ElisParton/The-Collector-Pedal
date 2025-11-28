#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ann.h"



typedef struct WavFileHeader {         //       Offset  Size  Type       Description
// ----------------------------------------------------------------------------------------------------------------
unsigned             RiffChunk;      // 0       4     FourCC     'RIFF'
unsigned             ChunkSize;      // 4       4     DWord      size of the riff chunk (should be always filesize - 8)
unsigned             FileFormat;     // 8       4     FourCC     'WAVE'
unsigned             FormatChunk;    // 12      4     FourCC     'fmt '
unsigned             FormatSize;     // 16      4     DWord      size of Format structure (should be always 16 byte)
unsigned short       PcmFlags;       // 20      2     Word       bit 1: Signed data, bit 2: Float data.. 
unsigned short       Channels;       // 22      2     Word       samples per frame (example: one stereo frame consist from 2 samples)
unsigned             SampleRate;     // 24      4     DWord      frames per second (example: 44100 stereo-frames are played back per seccond)   
unsigned             ByteRate;       // 28      4     DWord      bytes per second (example: one second float32 stereo-track data: 44100frames * 2channels * 4bytes  )
unsigned short       BlockAlign;     // 32      2     Word       byte per frame (example: each float32 stero frame is 8 byte in size - one float is 4byte - 2 channels are 2 floats, each 4byte)
unsigned short       BitDepth;       // 34      2     Word       bits per sample (example: one float32 is 4 byte where each byte has 8 bit... so: 32 bit per sample )  */
unsigned             DataChunk;      // 36      4     FourCC     'data'
unsigned             DataSize;       // 40      4     DWord      size of of payload data (should be the total file size minus this headers size of 44 byte)
} WavFileHeader;


int main(int argc, char *argv[])
{
	real *data;
	int rows, stride;
	char *filename = "and.csv";
	// load the data
	if (argc > 1)
		filename = argv[1];

	printf("Loading %s...", filename);
	WavFileHeader wavheader;
    FILE* file = fopen(filename,"rb");
    fread(&wavheader,1,44,file);

	puts("done.");

	PNetwork pnet = ann_make_network(OPT_ADAPT, LOSS_MSE);
	if (!pnet)
		return ERR_FAIL;

	PTensor x_train = tensor_create_from_array(rows, stride, data);
	if (!x_train)
		return ERR_FAIL;

	PTensor y_train = tensor_slice_cols(x_train, 2);
	if (!y_train)
		return ERR_FAIL;

	// define our network
	CHECK_OK(ann_add_layer(pnet, 2, LAYER_INPUT, ACTIVATION_NULL));
	CHECK_OK(ann_add_layer(pnet, 1, LAYER_OUTPUT, ACTIVATION_SIGMOID));

	ann_train_network(pnet, x_train, y_train, x_train->rows);
	
	real outputs[1];
	CHECK_OK(ann_predict(pnet, &data[0], outputs));

	print_outputs(pnet);

	ann_free_network(pnet);

	free(data);
	return ERR_OK;
}
