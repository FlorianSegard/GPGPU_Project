#ifndef LABCONVERTER_HPP
#define LABCONVERTER_HPP

#include <cmath>
#include "../Image.hpp"

struct lab
{
    float L, a, b;
};

ImageView<lab> rgbtolab_converter(ImageView<rgb8> in);
float* XYZ_color_space(float r_linear, float g_linear, float b_linear);
float get_linear(float r_g_b);
float f(float t);

#endif
