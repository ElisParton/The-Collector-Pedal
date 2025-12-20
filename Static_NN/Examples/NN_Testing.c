#include "../NN_Tools.c"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#define _USE_MATH_DEFINES
#define LEARNING_RATE 0.0005
#define TRAINING_SAMPLES 10000
#define TRAINING_SAMPLES_MIN -1.0
#define TRAINING_SAMPLES_MAX 1.0
#define ACCEPTABLE_ERROR 0.00005
FILE *fptr; // Output file pointer for trained net

double sin_function(double x)
{
    return (tanh(30 * x) / 10);
}

/* =========================
   WAV structure
   ========================= */

typedef struct
{
    uint16_t audio_format; // PCM = 1
    uint16_t num_channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;

    uint32_t num_frames; // frames per channel
    float *samples;      // interleaved [-1, 1]
} WavFile;

/* =========================
   Binary helpers
   ========================= */

static int read_u16(FILE *f, uint16_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1;
}

static int read_u32(FILE *f, uint32_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1;
}

static void write_u16(FILE *f, uint16_t v)
{
    fwrite(&v, sizeof(v), 1, f);
}

static void write_u32(FILE *f, uint32_t v)
{
    fwrite(&v, sizeof(v), 1, f);
}

/* =========================
   WAV loader (PCM only)
   ========================= */

int load_wav(const char *path, WavFile *wav)
{
    memset(wav, 0, sizeof(*wav));

    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;

    char id[4];
    uint32_t size;

    fread(id, 1, 4, f); // "RIFF"
    read_u32(f, &size);
    fread(id, 1, 4, f); // "WAVE"

    if (memcmp(id, "WAVE", 4) != 0)
        goto fail;

    uint32_t data_size = 0;
    long data_offset = 0;

    while (fread(id, 1, 4, f) == 4)
    {
        read_u32(f, &size);

        if (!memcmp(id, "fmt ", 4))
        {
            read_u16(f, &wav->audio_format);
            read_u16(f, &wav->num_channels);
            read_u32(f, &wav->sample_rate);

            uint32_t byte_rate;
            uint16_t block_align;

            read_u32(f, &byte_rate);
            read_u16(f, &block_align);
            read_u16(f, &wav->bits_per_sample);

            if (size > 16)
                fseek(f, size - 16, SEEK_CUR);
        }
        else if (!memcmp(id, "data", 4))
        {
            data_offset = ftell(f);
            data_size = size;
            fseek(f, size, SEEK_CUR);
        }
        else
        {
            fseek(f, size, SEEK_CUR);
        }
    }

    if (wav->audio_format != 1 || data_size == 0)
        goto fail;

    uint32_t bytes_per_sample = wav->bits_per_sample / 8;
    uint32_t total_samples = data_size / bytes_per_sample;

    wav->num_frames = total_samples / wav->num_channels;
    wav->samples = malloc(sizeof(float) * total_samples);
    if (!wav->samples)
        goto fail;

    fseek(f, data_offset, SEEK_SET);

    for (uint32_t i = 0; i < total_samples; ++i)
    {
        if (wav->bits_per_sample == 16)
        {
            int16_t s;
            fread(&s, 2, 1, f);
            wav->samples[i] = s / 32768.0f;
        }
        else if (wav->bits_per_sample == 24)
        {
            uint8_t b[3];
            fread(b, 1, 3, f);
            int32_t s = (b[2] << 24) | (b[1] << 16) | (b[0] << 8);
            wav->samples[i] = s / 2147483648.0f;
        }
        else if (wav->bits_per_sample == 32)
        {
            int32_t s;
            fread(&s, 4, 1, f);
            wav->samples[i] = s / 2147483648.0f;
        }
        else
        {
            goto fail;
        }
    }

    fclose(f);
    return 1;

fail:
    fclose(f);
    free(wav->samples);
    return 0;
}

/* =========================
   WAV writer (PCM)
   ========================= */

int write_wav(const char *path, const WavFile *wav)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return 0;

    uint16_t bytes_per_sample = wav->bits_per_sample / 8;
    uint32_t data_size =
        wav->num_frames * wav->num_channels * bytes_per_sample;

    /* RIFF */
    fwrite("RIFF", 1, 4, f);
    write_u32(f, 36 + data_size);
    fwrite("WAVE", 1, 4, f);

    /* fmt */
    fwrite("fmt ", 1, 4, f);
    write_u32(f, 16);
    write_u16(f, 1);
    write_u16(f, wav->num_channels);
    write_u32(f, wav->sample_rate);
    write_u32(f, wav->sample_rate * wav->num_channels * bytes_per_sample);
    write_u16(f, wav->num_channels * bytes_per_sample);
    write_u16(f, wav->bits_per_sample);

    /* data */
    fwrite("data", 1, 4, f);
    write_u32(f, data_size);

    uint32_t total = wav->num_frames * wav->num_channels;

    for (uint32_t i = 0; i < total; ++i)
    {
        float x = wav->samples[i];

        if (x > 1.0f)
            x = 1.0f;
        if (x < -1.0f)
            x = -1.0f;

        if (wav->bits_per_sample == 16)
        {
            int16_t s = (int16_t)(x * 32767.0f);
            fwrite(&s, 2, 1, f);
        }
        else if (wav->bits_per_sample == 24)
        {
            int32_t s = (int32_t)(x * 8388607.0f);
            uint8_t b[3] = {s & 0xFF, (s >> 8) & 0xFF, (s >> 16) & 0xFF};
            fwrite(b, 1, 3, f);
        }
        else if (wav->bits_per_sample == 32)
        {
            int32_t s = (int32_t)(x * 2147483647.0f);
            fwrite(&s, 4, 1, f);
        }
    }

    fclose(f);
    return 1;
}

/* =========================
   Cleanup
   ========================= */

void free_wav(WavFile *wav)
{
    free(wav->samples);
    wav->samples = NULL;
}

int main()
{
    enum
    {
        INP = 1,
        HID1 = 10,
        HID2 = 15,
        HID3 = 10,
        OUP = 1
    };

    double inp_buffer[INP];

    double weights_buffer1[INP * HID1];
    double bias_buffer1[HID1];
    double avalues_buffer1[HID1];
    double bws_buffer1[HID1];
    double deltas_buffer1[HID1];

    double weights_buffer2[HID1 * HID2];
    double bias_buffer2[HID2];
    double avalues_buffer2[HID2];
    double bws_buffer2[HID2];
    double deltas_buffer2[HID2];

    double weights_buffer3[HID2 * HID3];
    double bias_buffer3[HID3];
    double avalues_buffer3[HID3];
    double bws_buffer3[HID3];
    double deltas_buffer3[HID3];

    double weights_buffer4[HID3 * OUP];
    double bias_buffer4[OUP];
    double avalues_buffer4[OUP];
    double bws_buffer4[OUP];
    double deltas_buffer4[OUP];

    NN_Input_Layer input_layer = {
        .neurons = INP,
        .inputs = matrix_create(INP, 1, inp_buffer)};

    NN_Layer layer1 = NN_create_layer(INP, HID1,
                                      weights_buffer1,
                                      bias_buffer1,
                                      avalues_buffer1,
                                      bws_buffer1,
                                      deltas_buffer1,
                                      Tanh_Activation);

    NN_Layer layer2 = NN_create_layer(HID1, HID2,
                                      weights_buffer2,
                                      bias_buffer2,
                                      avalues_buffer2,
                                      bws_buffer2,
                                      deltas_buffer2,
                                      Tanh_Activation);

    NN_Layer layer3 = NN_create_layer(HID2, HID3,
                                      weights_buffer3,
                                      bias_buffer3,
                                      avalues_buffer3,
                                      bws_buffer3,
                                      deltas_buffer3,
                                      Tanh_Activation);

    NN_Layer layer4 = NN_create_layer(HID3, OUP,
                                      weights_buffer4,
                                      bias_buffer4,
                                      avalues_buffer4,
                                      bws_buffer4,
                                      deltas_buffer4,
                                      Linear_Activation);

    NN_Layer layers[4] = {layer1, layer2, layer3, layer4};
    Neural_Network net = NN_create(4, input_layer, layers);

    double input_buf[TRAINING_SAMPLES];
    for (int i = 0; i < TRAINING_SAMPLES; i++)
    {
        input_buf[i] = TRAINING_SAMPLES_MIN + (double)i * (TRAINING_SAMPLES_MAX - TRAINING_SAMPLES_MIN) / (TRAINING_SAMPLES - 1);
    }

    Matrix input = matrix_create(1, TRAINING_SAMPLES, input_buf);
    double target_buf[TRAINING_SAMPLES];
    Matrix target = matrix_create(1, TRAINING_SAMPLES, target_buf);
    target = matrix_apply_func(input, sin_function, target_buf);

    double error = ACCEPTABLE_ERROR + 1;
    int i = 0;
    // for (int i = 0; i < EPOCHS; i++)
    while (error / input.cols > ACCEPTABLE_ERROR)
    {
        error = 0;
        for (int j = 0; j < input.cols; j++)
        {
            int indx = rand() % input.cols;
            static double current_input_buf[1];
            Matrix current_input = matrix_slice(input, 0, 1, indx, indx + 1, current_input_buf);
            static double current_target_buf[1];
            Matrix current_target = matrix_slice(target, 0, 1, indx, indx + 1, current_target_buf);
            NN_forward_pass(net, current_input);
            error += NN_MSE(net, current_target);
            NN_backward_pass(net, current_target);
            NN_update_weights(net, LEARNING_RATE);
            NN_update_biases(net, LEARNING_RATE);
        }
        printf("Epoch %d, MSE: %f\n", i + 1, error / input.cols);
        i++;
    }
    error = 0;
    for (int j = 0; j < input.cols; j = j + input.cols / 1000)
    {
        static double current_input_buf[1];
        Matrix current_input = matrix_slice(input, 0, 1, j, j + 1, current_input_buf);
        static double current_target_buf[1];
        Matrix current_target = matrix_slice(target, 0, 1, j, j + 1, current_target_buf);
        NN_forward_pass(net, current_input);
        printf("(%f, %f)\n", matrix_get_entry(current_input, 0, 0), matrix_get_entry(net.layers[net.num_layers - 1].activated_values, 0, 0));
        error += NN_MSE(net, current_target);
    }
    printf("Final MSE after training: %f\n", error / input.cols);

    WavFile wav;

    if (!load_wav("C:\\Users\\eyqp2\\Desktop\\The Collector Pedal - Testing\\audio_samples\\mono_speech.wav", &wav))
        return 1;

    uint32_t total = wav.num_frames * wav.num_channels;

    for (uint32_t i = 0; i < total; ++i)
    {
        double sample = wav.samples[i];
        Matrix input = matrix_create(1, 1, &sample);
        NN_forward_pass(net, input);
        wav.samples[i] = matrix_get_entry(net.layers[net.num_layers - 1].activated_values, 0, 0);
    }
    if (!write_wav("mono_speech_processed.wav", &wav))
        return 1;

    free_wav(&wav);
    return 0;
}