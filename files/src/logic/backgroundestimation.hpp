#pragma once


#include <cmath>
#include "../Image.hpp"
#include "../filter_impl.h"
#include "../Compute.hpp"
// #include "labConverter.hpp"

#ifdef __cplusplus
    extern "C" {
#endif



// struct lab
// {
//     float L, a, b;
// };

void background_init(Parameters* params);

void background_process_frame(ImageView<lab> in, ImageView<lab> currentBackground,
                ImageView<lab> candidateBackground, ImageView<int> currentTimePixels,
                ImageView<float> currentDistancePixels);

void check_background_cu(ImageView<lab> in, ImageView<lab> currentBackground,
                            ImageView<lab> candidateBackground, ImageView<int> currentTimePixels,
                            ImageView<float> currentDistancePixels, int width, int height);

#ifdef __cplusplus
    }
#endif
