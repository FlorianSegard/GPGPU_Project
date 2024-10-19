#include <iostream>
#include "backgroundestimation.hpp"


ImageView<lab> currentBackground;
int* currentTimePixels;


lab averageLAB(lab p1, lab p2) {
    lab result;
    result.L = (p1.L + p2.L) / 2;
    result.A = (p1.A + p2.A) / 2;
    result.B = (p1.B + p2.B) / 2;
    return result;
}


float labDistance(lab p1, lab p2) {
    return sqrt(pow(p1.L - p2.L, 2) + pow(p1.a - p2.a, 2) + pow(p1.b - p2.b, 2));
}



void check_background(ImageView<lab> in)
{
    for (int y = 0; y < in.width, y++)
    {
        lab* lineptr_lab = (lab*)((std::byte*)in.buffer + y * in.stride);
        lab* lineptr_lab_background = (lab*)((std::byte*)currentBackground.buffer + y * currentBackground.stride);

        for (int x = 0; x < in.height; x++)
        {
            lab currentpixel = lineptr_lab[x];
            lab currentpixel_background = lineptr_lab_background[x];

            float distance = labDistance(currentpixel, currentpixel_background);
            if (distance < 25)
                continue;
            else if 

        }
    }
}


