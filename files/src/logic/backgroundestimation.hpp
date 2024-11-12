#pragma once


#include <cmath>
#include "labConverter.hpp"

#ifdef __cplusplus
    extern "C" {
#endif

typedef enum {
    CPU,
    GPU
} e_device_t;

typedef struct  {
    e_device_t device;
} Parameters;

struct lab
{
    float L, a, b;
};

void background_init(Parameters* params);

void background_process_frame(ImageView<lab> in, ImageView<lab> currentBackground, 
                ImageView<lab> candidateBackground, ImageView<int> currentTimePixels, 
                ImageView<float> currentDistancePixels);

#ifdef __cplusplus
    }
#endif
