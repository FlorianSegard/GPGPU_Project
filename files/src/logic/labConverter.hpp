#pragma once

#include <cmath>
#include "../Image.hpp"
#include "../filter_impl.h"

#ifdef __cplusplus
    extern "C" {
#endif


void labConv_init(Parameters* params);

void labConv_process_frame(ImageView<rgb8> rgb_image, ImageView<lab> lab_image);

void rgbtolab_converter_cu(ImageView<rgb8> rgb_image, ImageView<lab> backgroundLAB, int width, int height);

#ifdef __cplusplus
    }
#endif