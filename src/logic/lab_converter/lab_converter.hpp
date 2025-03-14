#pragma once

#include <cmath>
#include "../../common/Image.hpp"
#include "../../filters/filter_impl.hpp"
#include "../../Compute.hpp"

#ifdef __cplusplus
    extern "C" {
#endif


void lab_conv_init(Parameters* params);

void lab_conv_process_frame(ImageView<rgb8> rgb_image, ImageView<lab> lab_image);

void rgbtolab_converter_cu(ImageView<rgb8> rgb_image, ImageView<lab> backgroundLAB, int width, int height);

#ifdef __cplusplus
    }
#endif