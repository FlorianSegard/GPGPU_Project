#pragma once
#include <cmath>
#include "../Image.hpp"
#include "../filter_impl.h"

// Shared to average lab pixels
__host__ __device__ lab averageLAB(lab p1, lab p2) {
    lab result;
    result.L = (p1.L + p2.L) / 2;
    result.a = (p1.a + p2.a) / 2;
    result.b = (p1.b + p2.b) / 2;
    return result;
}


// Shared Euclidean distance
__host__ __device__ float labDistance(lab p1, lab p2) {
    return sqrt(pow(p1.L - p2.L, 2) + pow(p1.a - p2.a, 2) + pow(p1.b - p2.b, 2));
}