#pragma once


#include <cmath>
#include "../Image.hpp"
#include "../filter_impl.h"
// #include "labConverter.hpp"

#ifdef __cplusplus
    extern "C" {
#endif


// struct lab
// {
//     float L, a, b;
// };

void background_init(Parameters* params);

void background_process_frame(Image<lab> in, Image<lab> currentBackground,
                Image<lab> candidateBackground, Image<int> currentTimePixels,
                Image<float> currentDistancePixels);

void check_background_cu(Image<lab> in, Image<lab> currentBackground,
                            Image<lab> candidateBackground, Image<int> currentTimePixels,
                            Image<float> currentDistancePixels, int width, int height);

#ifdef __cplusplus
    }
#endif
