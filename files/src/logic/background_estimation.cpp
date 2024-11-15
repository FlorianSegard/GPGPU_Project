#include <iostream>
#include "background_estimation.hpp"
#include "background_utils.hpp"

void check_background_cpp(ImageView<lab> in, ImageView<lab> currentBackground,
                            ImageView<lab> candidateBackground, ImageView<int> currentTimePixels,
                            ImageView<float> currentDistancePixels, int width, int height)
{
    for (int y = 0; y < width; y++)
    {
        lab* lineptr_lab = (lab*)((std::byte*)in.buffer + y * in.stride);
        lab* lineptr_lab_background = (lab*)((std::byte*)currentBackground.buffer + y * currentBackground.stride);
        lab* lineptr_lab_candidate = (lab*)((std::byte*)candidateBackground.buffer + y * candidateBackground.stride);
        int* lineptr_time = (int*)((std::byte*)currentTimePixels.buffer + y * currentTimePixels.stride);
        float* lineptr_distance = (float*)((std::byte*)currentDistancePixels.buffer + y * currentDistancePixels.stride);

        for (int x = 0; x < height; x++)
        {
            lab currentpixel = lineptr_lab[x];
            lab currentpixel_background = lineptr_lab_background[x];
            lab currentpixel_candidate = lineptr_lab_candidate[x];
            int currentpixel_time = lineptr_time[x];
            float distance;
            labDistance(currentpixel, currentpixel_background, &distance);
            lineptr_distance[x] = distance;
            if (distance >= 5.0f)
            {
                if (currentpixel_time == 0)
                {
                    lineptr_lab_candidate[x] = currentpixel;
                    lineptr_time[x] = 1;
                }
                else if (currentpixel_time < 100)
                {
                    lab average;
                    averageLAB(currentpixel, currentpixel_candidate, &average);

                    lineptr_lab_candidate[x] = average;
                    lineptr_time[x]++;
                }
                else
                {
                    lineptr_lab_background[x] = currentpixel_candidate;
                    lineptr_time[x] = 0;
                }
            }
            else
            {
                lab average;
                averageLAB(currentpixel, currentpixel_background, &average);
                lineptr_lab_background[x] = average;
                lineptr_time[x] = 0;
                lineptr_distance[x] = 0.0f;
            }
        }
    }
}



extern "C" {

    static Parameters b_params;

    void background_init(Parameters* params)
    {
        b_params = *params;
    }

    void background_process_frame(ImageView<lab> in, ImageView<lab> currentBackground,
                        ImageView<lab> candidateBackground, ImageView<int> currentTimePixels,
                        ImageView<float> currentDistancePixels)
    {
        int width = in.width;
        int height = in.height;
        if (b_params.device == e_device_t::CPU)
            check_background_cpp(in, currentBackground, 
                        candidateBackground, currentTimePixels, 
                        currentDistancePixels, width, height);

        else if (b_params.device == e_device_t::GPU)
            check_background_cu(in, currentBackground, 
                        candidateBackground, currentTimePixels, 
                        currentDistancePixels, width, height);
    }

}


// int main()
// {
//     int width = 100;
//     int height = 100;

//     Image<lab> zero_image(width, height, false);
    
//     Image<int> currentTimePixels(width, height, false);
//     Image<float> currentDistancePixels(width, height, false);

//     for (int y = 0; y < height; ++y) {
//         lab* lineptr = (lab*)((std::byte*)zero_image.buffer + y * zero_image.stride);
//         int* lineptr_time = (int*)((std::byte*)currentTimePixels.buffer + y * currentTimePixels.stride);
//         for (int x = 0; x < width; ++x) {
//             // rgb8 pixel_value({1, 0, 0});
//             lineptr[x] = {1, 10, 1};
//             lineptr_time[x] = {0};
//             // std::cout << "Pixel at (" << x << ", " << y << "): r = " << (int)lineptr[x].r 
//             //           << ", g = " << (int)lineptr[x].g 
//             //           << ", b = " << (int)lineptr[x].b << std::endl;
//             // zero_image.buffer[y * width + x] = {0, 0, 0};
//             // std::cout << "x: " << x << ", y: " << y << std::endl;

//         }
//     }
//     Image<lab> currentBackground = zero_image.clone();
//     Image<lab> candidateBackground = zero_image.clone();

//     int i = 0;
//     while (true)
//     {
//         check_background_cpp(zero_image, currentBackground, candidateBackground, currentTimePixels, currentDistancePixels, width, height);

//         for (int y = 0; y < height; y++) 
//         {
//             int* lineptr_time = (int*)((std::byte*)currentTimePixels.buffer + y * currentTimePixels.stride);
//             for (int x = 0; x < width; x++) 
//             {
//                 std::cout << lineptr_time[x] << std::endl;
//             }
//         }
//         if (i == 10)
//         {
//             for (int y = 0; y < height; ++y) 
//             {
//                 lab* lineptr = (lab*)((std::byte*)zero_image.buffer + y * zero_image.stride);
//                 for (int x = 0; x < width; ++x) 
//                 {
//                     lineptr[x] = {10, 100, 10};

//                 }
//             }
//         }
//         i++;
//     }

// }
