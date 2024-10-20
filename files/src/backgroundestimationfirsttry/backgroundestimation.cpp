#include <iostream>
#include "backgroundestimation.hpp"


// ImageView<lab> currentBackground;
// ImageView<lab> candidateBackground;
// ImageView<int> currentTimePixels;


lab averageLAB(lab p1, lab p2) {
    lab result;
    result.L = (p1.L + p2.L) / 2;
    result.a = (p1.a + p2.a) / 2;
    result.b = (p1.b + p2.b) / 2;
    return result;
}


// Distance euclidienne
float labDistance(lab p1, lab p2) {
    return sqrt(pow(p1.L - p2.L, 2) + pow(p1.a - p2.a, 2) + pow(p1.b - p2.b, 2));
}


void check_background(ImageView<lab> in, ImageView<lab> currentBackground, ImageView<lab> candidateBackground, ImageView<int> currentTimePixels)
{
    for (int y = 0; y < in.width; y++)
    {
        lab* lineptr_lab = (lab*)((std::byte*)in.buffer + y * in.stride);
        lab* lineptr_lab_background = (lab*)((std::byte*)currentBackground.buffer + y * currentBackground.stride);
        lab* lineptr_lab_candidate = (lab*)((std::byte*)candidateBackground.buffer + y * candidateBackground.stride);
        int* lineptr_time = (int*)((std::byte*)currentTimePixels.buffer + y * currentTimePixels.stride);

        for (int x = 0; x < in.height; x++)
        {
            lab currentpixel = lineptr_lab[x];
            lab currentpixel_background = lineptr_lab_background[x];
            lab currentpixel_candidate = lineptr_lab_candidate[x];
            int currentpixel_time = lineptr_time[x];

            float distance = labDistance(currentpixel, currentpixel_background);
            if (distance < 25)
            {
                if (currentpixel_time == 0)
                {
                    lineptr_lab_candidate[x] = currentpixel;
                    lineptr_time[x]++;
                }   
                else if (currentpixel_time < 100)
                {
                    lineptr_lab_candidate[x] = averageLAB(currentpixel, currentpixel_candidate);
                    lineptr_time[x]++;
                }
                else
                {
                    lineptr_lab_background[x] = currentpixel_candidate;
                    lineptr_time[x]++;
                }
            }
            else
            {
                lineptr_lab_background[x] = averageLAB(currentpixel, currentpixel_background);
                lineptr_time[x] = 0;
            }
        }
    }
}

int main()
{
    int width = 100;
    int height = 100;

    Image<lab> zero_image(width, height, false);
    Image<int> currentTimePixels(width, height, false);

    for (int y = 0; y < height; ++y) {
        lab* lineptr = (lab*)((std::byte*)zero_image.buffer + y * zero_image.stride);
        int* lineptr_time = (int*)((std::byte*)currentTimePixels.buffer + y * currentTimePixels.stride);
        for (int x = 0; x < width; ++x) {
            // rgb8 pixel_value({1, 0, 0});
            lineptr[x] = {1, 10, 1};
            lineptr_time[x] = {0};
            // std::cout << "Pixel at (" << x << ", " << y << "): r = " << (int)lineptr[x].r 
            //           << ", g = " << (int)lineptr[x].g 
            //           << ", b = " << (int)lineptr[x].b << std::endl;
            // zero_image.buffer[y * width + x] = {0, 0, 0};
            // std::cout << "x: " << x << ", y: " << y << std::endl;

        }
    }
    Image<lab> currentBackground = zero_image.clone();
    Image<lab> candidateBackground = zero_image.clone();

    int i = 0;
    while (true)
    {
        check_background(zero_image, currentBackground, candidateBackground, currentTimePixels);

        for (int y = 0; y < height; ++y) 
        {
            int* lineptr_time = (int*)((std::byte*)currentTimePixels.buffer + y * currentTimePixels.stride);
            for (int x = 0; x < width; ++x) 
            {
                std::cout << lineptr_time[x] << std::endl;
            }
        }
        if (i == 10)
        {
            for (int y = 0; y < height; ++y) 
            {
                lab* lineptr = (lab*)((std::byte*)zero_image.buffer + y * zero_image.stride);
                for (int x = 0; x < width; ++x) 
                {
                    lineptr[x] = {10, 100, 10};

                }
            }
        }
        i++;
    }

}
