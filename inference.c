#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "ann.h"

typedef struct WavFileHeader {
    unsigned             RiffChunk;
    unsigned             ChunkSize;
    unsigned             FileFormat;
    unsigned             FormatChunk;
    unsigned             FormatSize;
    unsigned short       PcmFlags;
    unsigned short       Channels;
    unsigned             SampleRate;
    unsigned             ByteRate;
    unsigned short       BlockAlign;
    unsigned short       BitDepth;
    unsigned             DataChunk;
    unsigned             DataSize;
} WavFileHeader;

/* now also returns sample rate via out_sample_rate */
real *wav_load(const char *filename, long *out_frames, int *out_channels, unsigned *out_sample_rate)
{
    if (!filename || !out_frames || !out_channels || !out_sample_rate) return NULL;

    FILE *f = fopen(filename, "rb");
    if (!f) {
        /* perror omitted to reduce clutter */
        return NULL;
    }

    WavFileHeader hdr;
    if (fread(&hdr, 1, 44, f) != 44) {
        fclose(f);
        return NULL;
    }

    int channels = (int)hdr.Channels;
    int bitDepth = (int)hdr.BitDepth;
    unsigned dataBytes = hdr.DataSize;
    if (channels <= 0 || bitDepth <= 0 || dataBytes == 0) {
        fclose(f);
        return NULL;
    }

    int bytesPerSample = bitDepth / 8;
    long frames = (long)(dataBytes / (bytesPerSample * channels));
    if (frames <= 0) {
        fclose(f);
        return NULL;
    }

    size_t totalSamples = (size_t)frames * (size_t)channels;
    real *out = (real*)malloc(sizeof(real) * totalSamples);
    if (!out) { fclose(f); return NULL; }

    if (hdr.PcmFlags == 1 && bitDepth == 16) {
        int16_t *tmp = (int16_t*)malloc(dataBytes);
        if (!tmp) { free(out); fclose(f); return NULL; }
        if (fread(tmp, 1, dataBytes, f) != dataBytes) {
            free(tmp); free(out); fclose(f); return NULL;
        }
        for (size_t i = 0; i < totalSamples; ++i)
            out[i] = (real)tmp[i] / 32768.0f;
        free(tmp);
    }
    else if (hdr.PcmFlags == 3 && bitDepth == 32) {
        float *tmp = (float*)malloc(dataBytes);
        if (!tmp) { free(out); fclose(f); return NULL; }
        if (fread(tmp, 1, dataBytes, f) != dataBytes) {
            free(tmp); free(out); fclose(f); return NULL;
        }
        for (size_t i = 0; i < totalSamples; ++i)
            out[i] = (real)tmp[i];
        free(tmp);
    }
    else {
        free(out);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *out_frames = frames;
    *out_channels = channels;
    *out_sample_rate = hdr.SampleRate;
    return out;
}

void wav_save(const char *filename, unsigned sample_rate, real *samples, long num_samples, int channels)
{
    if (!filename || !samples || num_samples <= 0 || channels <= 0) return;

    FILE *f = fopen(filename, "wb");
    if (!f) return;

    WavFileHeader hdr = {0};
    hdr.RiffChunk = 0x46464952;
    hdr.FileFormat = 0x45564157;
    hdr.FormatChunk = 0x20746d66;
    hdr.FormatSize = 16;
    hdr.PcmFlags = 1;
    hdr.Channels = channels;
    hdr.SampleRate = sample_rate;
    hdr.BitDepth = 16;
    hdr.ByteRate = sample_rate * channels * 2;
    hdr.BlockAlign = channels * 2;
    hdr.DataChunk = 0x61746164;
    hdr.DataSize = (unsigned)(num_samples * (long)channels * 2);
    hdr.ChunkSize = 36 + hdr.DataSize;

    fwrite(&hdr, 1, 44, f);

    int16_t *tmp = (int16_t*)malloc((size_t)num_samples * (size_t)channels * sizeof(int16_t));
    if (!tmp) { fclose(f); return; }
    for (long i = 0; i < num_samples * channels; ++i)
        tmp[i] = (int16_t)fmaxf(-32768.0f, fminf(32767.0f, samples[i] * 32767.0f));

    fwrite(tmp, 1, (size_t)num_samples * channels * 2, f);
    free(tmp);
    fclose(f);
}

int main(int argc, char *argv[])
{
    char *network_filename = "mnist-fashion.nna";
    char *input_wav = "input.wav";

    real *test_data;
    int test_rows, test_stride;

    if (argc > 1)
        network_filename = argv[1];

    if (argc > 2)
        input_wav = argv[2];

    printf("Loading %s...", network_filename);
    PNetwork pnet = ann_load_network(network_filename);
    if (!pnet)
        return ERR_FAIL;
    puts("done.");

    // Load WAV file
    long frames;
    int channels;
    unsigned sample_rate;
    real *input_samples = wav_load(input_wav, &frames, &channels, &sample_rate);
    if (!input_samples)
        return ERR_FAIL;

    size_t totalSamples = (size_t)frames * (size_t)channels;

    // Predict for each sample (assumes network takes 1 sample and returns 1 sample)
    real *output_samples = (real *)malloc(sizeof(real) * totalSamples);
    if (!output_samples) { free(input_samples); return ERR_FAIL; }

    for (size_t i = 0; i < totalSamples; ++i) {
        real input[1] = { input_samples[i] };
        real output_buf[1] = { 0 };
        /* ann_predict is expected to fill the output_buf */
        ann_predict(pnet, input, output_buf);
        output_samples[i] = output_buf[0];
    }

    // Save output WAV file (frames, channels)
    wav_save("output.wav", sample_rate, output_samples, frames, channels);

    free(input_samples);
    free(output_samples);

    // free network memory
    ann_free_network(pnet);
    return ERR_OK;
}