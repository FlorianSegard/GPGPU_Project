#include "image.hpp"
#include <cmath>

struct lab
{
    float L, a, b;
}


ImageView<lab> currentBackground;


void check_background(ImageView<lab> in)
{
    for (int i = 0; i < in.width, i++)
    {
        for (int j = 0; j < in.height; j++)
        {
            float alpha = in.buffer[y * logo.stride + x] / 255.f;
        }
    }

}


ImageView<lab> rgbtolab_converter(ImageView<rgb8> in)
{
    ImageView<lab> backgroundLAB(in.width, in.height);
    buffer_lab = backgroundLAB.buffer;
    for (int x = 0; x < in.width, x++)
    {
        for (int y = 0; y < in.height; y++)
        {
            rgb8 currentpixel = in.buffer[y * in.stride + x];
            float r_normalized = currentpixel.r / 255.f;
            float g_normalized = currentpixel.g / 255.f;
            float b_normalized = currentpixel.b / 255.f;

            float r_linear = get_linear(r_normalized);
            float g_linear = get_linear(g_normalized);
            float b_linear = get_linear(b_normalized);

            float* result = XYZ_color_space(r_linear, g_linear, b_linear);

            float X = result[0];
            float Y = result[1];
            float Z = result[2];
            
            float X_n = X / 95.047;
            float Y_n = Y / 100;
            float Z_n = Z / 108.883;

            float X_n_sqrt = pow(X_n, 1/3);
            float Y_n_sqrt = pow(Y_n, 1/3);
            float Z_n_sqrt = pow(Z_n, 1/3);

            float L = 116 * Y_n_sqrt - 16;
            float a = 500 * (X_n_sqrt - Y_n_sqrt);
            float b = 200 * (Y_n_sqrt - Z_n_sqrt);

            lab currentpixel_lab = {L, a, b};
            buffer_lab[y * in.stride + x] = currentpixel_lab;
        }
    }
    return backgroundLAB;
}

float* XYZ_color_space(float r_linear, float g_linear, float b_linear)
{
    float matrix[9] = { 0.4124564, 0.3575761, 0.1804375
                        0.2126729, 0.7151522, 0.0721750
                        0.0193339, 0.1191920, 0.9503041 };

    float color_vector[3] = {r_linear, g_linear, b_linear};

    static float result[3];

    result[0] = matrix[0] * color_vector[0] + matrix[1] * color_vector[1] + matrix[2] * color_vector[2];
    result[1] = matrix[3] * color_vector[0] + matrix[4] * color_vector[1] + matrix[5] * color_vector[2];
    result[2] = matrix[6] * color_vector[0] + matrix[7] * color_vector[1] + matrix[8] * color_vector[2];

    return result;
}

float get_linear(float r_g_b)
{
    if r_g_b <= 0.04045
        return r_g_b / 12.92
    return pow((r_g_b + 0.055) / 1.055, 2.4)
}


void main()
{
    initializeBackground(backgroundFrame, currentFrame);



    while true
    {
        check_background()
    }
}