#include <stdio.h>
#include "portaudio.h"

int main() {
    printf("Hello World!\n");
    PaError err = Pa_Initialize();
    printf("Init result: %s\n", Pa_GetErrorText(err));
    Pa_Terminate();
    return 0;
}
