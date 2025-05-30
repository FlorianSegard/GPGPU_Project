#pragma once


#include <cmath>
#include "../../common/Image.hpp"
#include "../../filters/filter_impl.hpp"
#include "../../Compute.hpp"
// #include "labConverter.hpp"

#ifdef __cplusplus
    extern "C" {
#endif

void background_init(Parameters* params);

void background_process_frame(ImageView<lab> in, ImageView<lab> currentBackground,
                ImageView<lab> candidateBackground, ImageView<int> currentTimePixels,
                ImageView<float> currentDistancePixels, int bg_number_frame);

void check_background_cu(ImageView<lab> in, ImageView<lab> currentBackground,
                         ImageView<lab> candidateBackground, ImageView<int> currentTimePixels,
                         ImageView<float> currentDistancePixels, int width, int height, int bg_number_frame);

#ifdef __cplusplus
    }
#endif
