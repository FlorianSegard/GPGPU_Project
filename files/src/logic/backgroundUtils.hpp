#pragma once
#include <cmath>
#include "../Image.hpp"
#include "../filter_impl.h"

// Shared to average lab pixels
__device__ inline lab averageLAB(lab p1, lab p2) {
    lab result;
    result.L = (p1.L + p2.L) / 2.0f;
    result.a = (p1.a + p2.a) / 2.0f;
    result.b = (p1.b + p2.b) / 2.0f;
    return result;
}


// Shared Euclidean distance
__device__ inline float labDistance(lab p1, lab p2) {
    float dL = p1.L - p2.L;
    float da = p1.a - p2.a;
    float db = p1.b - p2.b;
    return sqrtf(dL * dL + da * da + db * db);
}