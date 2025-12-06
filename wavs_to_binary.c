#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
#include "ann.h"  // for `real` type

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

real *wav_load(const char *filename, long *out_frames, int *out_channels)
{
    if (!filename || !out_frames || !out_channels) return NULL;

    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    WavFileHeader hdr;
    if (fread(&hdr, 1, 44, f) != 44) { fclose(f); return NULL; }

    int channels = (int)hdr.Channels;
    int bitDepth = (int)hdr.BitDepth;
    unsigned dataBytes = hdr.DataSize;
    if (channels <= 0 || bitDepth <= 0 || dataBytes == 0) { fclose(f); return NULL; }

    int bytesPerSample = bitDepth / 8;
    long frames = (long)(dataBytes / (bytesPerSample * channels));
    if (frames <= 0) { fclose(f); return NULL; }

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
    return out;
}

real *to_mono(const real *in, long frames, int channels)
{
    if (!in || frames <= 0 || channels <= 0) return NULL;
    real *mono = (real*)malloc(sizeof(real) * (size_t)frames);
    if (!mono) return NULL;
    if (channels == 1) {
        for (long i = 0; i < frames; ++i) mono[i] = in[i];
    } else {
        for (long f = 0; f < frames; ++f) {
            double sum = 0.0;
            for (int c = 0; c < channels; ++c)
                sum += in[(size_t)f * (size_t)channels + (size_t)c];
            mono[f] = (real)(sum / channels);
        }
    }
    return mono;
}

int is_wav_name(const char *name)
{
    if (!name) return 0;
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    return ( (ext[1]=='w' || ext[1]=='W') &&
             (ext[2]=='a' || ext[2]=='A') &&
             (ext[3]=='v' || ext[3]=='V') &&
             ext[4] == '\0' );
}

int main(int argc, char *argv[])
{
    const char *indir = (argc > 1) ? argv[1] : "./audio_samples";
    const char *outfile = (argc > 2) ? argv[2] : "all_audio.bin";

    DIR *d = opendir(indir);
    if (!d) { fprintf(stderr, "opendir failed\n"); return 1; }

    FILE *fout = fopen(outfile, "wb");
    if (!fout) { closedir(d); fprintf(stderr, "cannot open output file\n"); return 1; }

    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        const char *name = entry->d_name;
        if (!is_wav_name(name)) continue;

        char fullpath[1024];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", indir, name);

        long frames = 0;
        int channels = 0;
        real *samples = wav_load(fullpath, &frames, &channels);
        if (!samples) { fprintf(stderr, "failed to load %s\n", fullpath); continue; }

        real *mono = to_mono(samples, frames, channels);
        free(samples);
        if (!mono) { fprintf(stderr, "failed to convert %s to mono\n", fullpath); continue; }

        // Write mono data directly to binary file
        fwrite(mono, sizeof(real), frames, fout);
        free(mono);

        printf("Written %ld samples from %s\n", frames, fullpath);
    }

    fclose(fout);
    closedir(d);
    printf("Done. Output file: %s\n", outfile);
    return 0;
}
