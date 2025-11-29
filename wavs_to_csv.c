#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
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

real *wav_load(const char *filename, long *out_frames, int *out_channels)
{
    if (!filename || !out_frames || !out_channels) return NULL;

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
    return out;
}

/* convert interleaved samples -> mono by averaging channels */
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
            for (int c = 0; c < channels; ++c) {
                sum += in[(size_t)f * (size_t)channels + (size_t)c];
            }
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
    const char *indir = (argc > 1) ? argv[1] : "C:\\Users\\eyqp2\\Desktop\\The Collector Pedal - Testing\\audio_samples\\";
    const char *outdir = (argc > 2) ? argv[2] : "C:\\Users\\eyqp2\\Desktop\\The Collector Pedal - Testing\\processed_audio_samples\\";
    const char *csvfile = (argc > 3) ? argv[3] : "data_pairs.csv";

    DIR *d = opendir(indir);
    if (!d) { fprintf(stderr, "opendir failed\n"); return 1; }

    FILE *fpt = fopen(csvfile, "a");
    if (!fpt) { closedir(d); fprintf(stderr, "could not open csv for appending\n"); return 1; }

    struct dirent *entry;
    /* BLOCK is the window size (samples per input/output block). Default to 100 */
    const int BLOCK = 1;
    /* stride controls how far the window moves each step (overlap when stride < BLOCK).
       Can be provided as argv[4] (integer). Default is BLOCK/2 (50% overlap). */
    int stride = 1;
    if (stride <= 0) stride = 1;
    if (stride > BLOCK) stride = BLOCK;

    while ((entry = readdir(d)) != NULL) {
        const char *name = entry->d_name;
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;
        if (!is_wav_name(name)) continue;

        size_t dirlen = strlen(indir);
        int need_sep = dirlen == 0 || (indir[dirlen - 1] != '/' && indir[dirlen - 1] != '\\');
        size_t pathlen = dirlen + (need_sep ? 1 : 0) + strlen(name) + 1;
        char *inpath = (char*)malloc(pathlen);
        if (!inpath) continue;
        if (need_sep) snprintf(inpath, pathlen, "%s\\%s", indir, name);
        else snprintf(inpath, pathlen, "%s%s", indir, name);

        /* build processed path */
        size_t odirlen = strlen(outdir);
        int o_need_sep = odirlen == 0 || (outdir[odirlen - 1] != '/' && outdir[odirlen - 1] != '\\');
        const char *suffix = "_target";
        const char *ext = strrchr(name, '.');
        size_t base_len = ext ? (size_t)(ext - name) : strlen(name);
        /* outname = base + suffix + extension(if any) */
        size_t outname_len = base_len + strlen(suffix) + (ext ? strlen(ext) : 0);
        size_t outpathlen = odirlen + (o_need_sep ? 1 : 0) + outname_len + 1;
        char *outpath = (char*)malloc(outpathlen);
        if (!outpath) { free(inpath); continue; }
        /* start with outdir (and separator if needed) */
        if (o_need_sep) snprintf(outpath, outpathlen, "%s\\", outdir);
        else snprintf(outpath, outpathlen, "%s", outdir);
        /* append filename with suffix before extension */
        size_t written = strlen(outpath);
        if (ext) {
            /* copy base part, then suffix, then extension */
            snprintf(outpath + written, outpathlen - written, "%.*s%s%s",
                     (int)base_len, name, suffix, ext);
        } else {
            /* no extension found: just append suffix */
            snprintf(outpath + written, outpathlen - written, "%s%s", name, suffix);
        }

        long in_frames = 0, out_frames = 0;
        int in_ch = 0, out_ch = 0;
        real *in_samples = wav_load(inpath, &in_frames, &in_ch);
        real *out_samples = wav_load(outpath, &out_frames, &out_ch);

        if (!in_samples || !out_samples) {
            if (!in_samples) fprintf(stderr, "missing/invalid input: %s\n", inpath);
            if (!out_samples) fprintf(stderr, "missing/invalid processed: %s\n", outpath);
            free(in_samples); free(out_samples);
            free(inpath); free(outpath);
            continue;
        }

        if (in_frames != out_frames) {
            fprintf(stderr, "frame mismatch %s (%ld) vs %s (%ld)\n", inpath, in_frames, outpath, out_frames);
            free(in_samples); free(out_samples);
            free(inpath); free(outpath);
            continue;
        }

        if (in_frames < BLOCK) {
            fprintf(stderr, "too few frames in %s\n", inpath);
            free(in_samples); free(out_samples);
            free(inpath); free(outpath);
            continue;
        }

        real *in_mono = to_mono(in_samples, in_frames, in_ch);
        real *out_mono = to_mono(out_samples, out_frames, out_ch);
        free(in_samples); free(out_samples);
        if (!in_mono || !out_mono) {
            free(in_mono); free(out_mono);
            free(inpath); free(outpath);
            continue;
        }

        /* Write overlapping (or non-overlapping) blocks of BLOCK samples:
           inputBLOCK,outputBLOCK per line. Window advances by 'stride' samples. */
        for (long off = 0; off + BLOCK <= in_frames; off += stride) {
            /* write BLOCK input samples */
            for (int i = 0; i < BLOCK; ++i) {
                fprintf(fpt, "%f", (double)in_mono[off + i]);
                if (i < BLOCK - 1) fprintf(fpt, ",");
            }
            /* separator between input and output blocks */
            fprintf(fpt, ",");
            /* write BLOCK output samples */
            for (int i = 0; i < BLOCK; ++i) {
                fprintf(fpt, "%f", (double)out_mono[off + i]);
                if (i < BLOCK - 1) fprintf(fpt, ",");
            }
            fprintf(fpt, "\n");
        }
        fflush(fpt);

        free(in_mono); free(out_mono);
        free(inpath); free(outpath);
    }

    fclose(fpt);
    closedir(d);
    return 0;
}