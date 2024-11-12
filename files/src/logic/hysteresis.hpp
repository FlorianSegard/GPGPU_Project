#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif



void hysteresis_cu(std::byte *opened_input, std::byte *hysteresis, int width, int height, int opened_input_pitch, int hysteresis_pitch, float lower_threshold, float upper_threshold);

#ifdef __cplusplus
}
#endif