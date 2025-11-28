#include <stdio.h>
#include "portaudio.h"
#include <stdint.h>
#include <string.h>

typedef struct {
    FILE *file;
    int channels;
    double sampleRate;
    uint64_t framesRecorded;
} RecordingData;

/* write a WAV header with a placeholder data chunk size (0). We'll patch sizes later. */
static void write_wav_header(FILE *f, int channels, double sampleRate, uint32_t dataBytesPlaceholder)
{
    /* RIFF header */
    fwrite("RIFF", 1, 4, f);
    uint32_t chunkSize = 36 + dataBytesPlaceholder; /* 4 + (8 + SubChunk1Size) + (8 + SubChunk2Size) */
    fwrite(&chunkSize, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    /* fmt subchunk */
    fwrite("fmt ", 1, 4, f);
    uint32_t subChunk1Size = 16;
    fwrite(&subChunk1Size, 4, 1, f);
    uint16_t audioFormat = 1; /* PCM */
    fwrite(&audioFormat, 2, 1, f);
    uint16_t numChannels = (uint16_t)channels;
    fwrite(&numChannels, 2, 1, f);
    uint32_t sampleRateU = (uint32_t)sampleRate;
    fwrite(&sampleRateU, 4, 1, f);
    uint16_t bitsPerSample = 16;
    uint16_t blockAlign = (uint16_t)(numChannels * bitsPerSample / 8);
    uint32_t byteRate = sampleRateU * blockAlign;
    fwrite(&byteRate, 4, 1, f);
    fwrite(&blockAlign, 2, 1, f);
    fwrite(&bitsPerSample, 2, 1, f);

    /* data subchunk header (size to be patched) */
    fwrite("data", 1, 4, f);
    uint32_t dataSize = dataBytesPlaceholder;
    fwrite(&dataSize, 4, 1, f);
}

/* callback: convert float input to int16 and append to file */
static int patestCallback( const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData )
{
    const float *in = (const float*)inputBuffer;
    float *out = (float*)outputBuffer;
    unsigned long i;
    (void) timeInfo;
    (void) statusFlags;

    RecordingData *rec = (RecordingData*)userData;
    int channels = rec ? rec->channels : 1;
    unsigned long samples = framesPerBuffer * (unsigned long)channels;

    /* output pass-through (optional) */
    if (out) {
        if (in == NULL) {
            for (i = 0; i < samples; ++i) out[i] = 0.0f;
        } else {
            for (i = 0; i < samples; ++i) out[i] = in[i];
        }
    }

    /* convert input floats to int16 and write to file (simple demo; not realtime-safe) */
    if (rec && rec->file) {
        if (in == NULL) {
            /* write silence */
            int16_t silence = 0;
            for (i = 0; i < samples; ++i) fwrite(&silence, sizeof(int16_t), 1, rec->file);
        } else {
            /* convert in-place into a small stack buffer then write */
            /* samples per buffer is usually small (framesPerBuffer * channels) */
            int16_t buf[4096];
            if (samples > (unsigned long)(sizeof(buf)/sizeof(buf[0]))) {
                /* fallback: write in chunks */
                unsigned long offset = 0;
                while (offset < samples) {
                    unsigned long chunk = samples - offset;
                    if (chunk > (unsigned long)(sizeof(buf)/sizeof(buf[0]))) chunk = (unsigned long)(sizeof(buf)/sizeof(buf[0]));
                    for (i = 0; i < chunk; ++i) {
                        float f = in[offset + i];
                        if (f > 1.0f) f = 1.0f;
                        if (f < -1.0f) f = -1.0f;
                        buf[i] = (int16_t)(f * 32767.0f);
                    }
                    fwrite(buf, sizeof(int16_t), chunk, rec->file);
                    offset += chunk;
                }
            } else {
                for (i = 0; i < samples; ++i) {
                    float f = in[i];
                    if (f > 1.0f) f = 1.0f;
                    if (f < -1.0f) f = -1.0f;
                    buf[i] = (int16_t)(f * 32767.0f);
                }
                fwrite(buf, sizeof(int16_t), samples, rec->file);
            }
        }
        rec->framesRecorded += framesPerBuffer;
    }

    return paContinue;
}

int main() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        printf("Pa_Initialize error: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    PaStream *stream = NULL;
    int channels = 1;                      /* desired channel count */
    double sampleRate = 44100.0;
    unsigned long framesPerBuffer = 256;   /* small buffer for low latency */

    RecordingData recData;
    memset(&recData, 0, sizeof(recData));
    recData.channels = channels;
    recData.sampleRate = sampleRate;
    recData.framesRecorded = 0;

    /* open output file and write placeholder header */
    recData.file = fopen("recording.wav", "wb");
    if (!recData.file) {
        printf("Failed to open output file for writing\n");
        Pa_Terminate();
        return 1;
    }
    write_wav_header(recData.file, channels, sampleRate, 0);

    err = Pa_OpenDefaultStream(&stream,
                               channels,       /* input channels */
                               channels,       /* output channels */
                               paFloat32,      /* sample format - matches callback */
                               sampleRate,
                               framesPerBuffer,
                               patestCallback, /* your callback */
                               &recData);     /* userData -> callback reads channel count and file */
    if (err != paNoError) {
        printf("Pa_OpenDefaultStream error: %s\n", Pa_GetErrorText(err));
        fclose(recData.file);
        Pa_Terminate();
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        printf("Pa_StartStream error: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        fclose(recData.file);
        Pa_Terminate();
        return 1;
    }

    printf("Recording. Press ENTER to stop.\n");
    getchar(); /* keep program running while callback runs */

    err = Pa_StopStream(stream);
    if (err != paNoError) printf("Pa_StopStream error: %s\n", Pa_GetErrorText(err));

    err = Pa_CloseStream(stream);
    if (err != paNoError) printf("Pa_CloseStream error: %s\n", Pa_GetErrorText(err));

    /* finalize WAV header: patch sizes */
    uint32_t dataBytes = (uint32_t)(recData.framesRecorded * recData.channels * (16/8));
    uint32_t riffChunkSize = 36 + dataBytes;
    fseek(recData.file, 4, SEEK_SET);
    fwrite(&riffChunkSize, 4, 1, recData.file);
    fseek(recData.file, 40, SEEK_SET);
    fwrite(&dataBytes, 4, 1, recData.file);

    fclose(recData.file);

    err = Pa_Terminate();
    if (err != paNoError) printf("Pa_Terminate error: %s\n", Pa_GetErrorText(err));
    return 0;
}
