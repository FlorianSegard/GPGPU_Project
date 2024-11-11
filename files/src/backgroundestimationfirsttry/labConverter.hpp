#pragma once

#include <cmath>
#include "../Image.hpp"


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

void labConv_init(Parameters* params);

void labConv_process_frame(ImageView<rgb> rgb_image, ImageView<lab> lab_image);

#ifdef __cplusplus
    }
#endif