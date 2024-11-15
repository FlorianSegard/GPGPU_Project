#include <iostream>
#include "labConverter.hpp"
#include "labConverterUtils.hpp"

ImageView<lab> rgbtolab_converter_cpp(ImageView<rgb8> rgb_image, ImageView<lab> lab_image, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        rgb8* lineptr = (rgb8*)((std::byte*)rgb_image.buffer + y * rgb_image.stride);
        lab* lineptr_lab = (lab*)((std::byte*)lab_image.buffer + y * lab_image.stride);

        for (int x = 0; x < width; x++)
        {
            rgb8 currentpixel = lineptr[x];


            float r_normalized = currentpixel.r / 255.f;
            float g_normalized = currentpixel.g / 255.f;
            float b_normalized = currentpixel.b / 255.f;

            float r_result;
            float g_result;
            float b_result;
            get_linear(r_normalized, &r_result);
            get_linear(g_normalized, &g_result);
            get_linear(b_normalized, &b_result);

            float r_linear = r_result;
            float g_linear = g_result;
            float b_linear = b_result;

            float result[3];

            XYZ_color_space(r_linear, g_linear, b_linear, result);

            float X = result[0];
            float Y = result[1];
            float Z = result[2];

            float X_n = X / 0.95047;
            float Y_n = Y / 1.0;
            float Z_n = Z / 1.08883;

            float X_n_result_f;
            float Y_n_result_f;
            float Z_n_result_f;
            f(X_n, &X_n_result_f);
            f(Y_n, &Y_n_result_f);
            f(Z_n, &Z_n_result_f);

            float X_n_sqrt = X_n_result_f;
            float Y_n_sqrt = Y_n_result_f;
            float Z_n_sqrt = Z_n_result_f;

            float L = 116 * Y_n_sqrt - 16;
            float a = 500 * (X_n_sqrt - Y_n_sqrt);
            float b = 200 * (Y_n_sqrt - Z_n_sqrt);

            lab currentpixel_lab = {L, a, b};

            lineptr_lab[x] = currentpixel_lab;

        }
    }
    return lab_image;
}

extern "C" {

    static Parameters l_params;

    void labConv_init(Parameters* params)
    {
        l_params = *params;
    }

    void labConv_process_frame(ImageView<rgb8> rgb_image, ImageView<lab> lab_image)
    {
        int width = rgb_image.width;
        int height = rgb_image.height;
        if (l_params.device == e_device_t::CPU)
            rgbtolab_converter_cpp(rgb_image, lab_image, width, height);
        else if (l_params.device == e_device_t::GPU)
            rgbtolab_converter_cu(rgb_image, lab_image, width, height);
    }

}



// int main()
// {

//     int width = 100;
//     int height = 100;

//     Image<rgb8> zero_image(width, height, false);
//     Image<lab> lab_image(width, height, false);

//     for (int y = 0; y < height; ++y) {
//         rgb8* lineptr = (rgb8*)((std::byte*)zero_image.buffer + y * zero_image.stride);
//         for (int x = 0; x < width; ++x) {
//             // rgb8 pixel_value({1, 0, 0});
//             lineptr[x] = {1, 10, 1};
//             // std::cout << "Pixel at (" << x << ", " << y << "): r = " << (int)lineptr[x].r 
//             //           << ", g = " << (int)lineptr[x].g 
//             //           << ", b = " << (int)lineptr[x].b << std::endl;
//             // zero_image.buffer[y * width + x] = {0, 0, 0};
//             // std::cout << "x: " << x << ", y: " << y << std::endl;

//         }
//     }
//     rgbtolab_converter(zero_image, lab_image, width, height);



//     for (int y = 0; y < lab_image.height; ++y) {
//         lab* lineptr_lab = (lab*)((std::byte*)lab_image.buffer + y * lab_image.stride);

//         for (int x = 0; x < lab_image.width; ++x) {
//             lab pixel = lineptr_lab[x];
//             std::cout << "L: " << pixel.L << ", a: " << pixel.a << ", b: " << pixel.b << std::endl;
//         }
//     }

//     return 0;
// }
