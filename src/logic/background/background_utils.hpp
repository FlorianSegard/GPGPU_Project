#pragma once
#include <cmath>
#include "../../common/Image.hpp"
#include "../../filters/filter_impl.hpp"

#ifdef __CUDACC__

// Device-specific code

// Shared to average lab pixels
__device__ inline void averageLAB(lab p1, lab p2, lab* result) {
    result->L = (p1.L + p2.L) / 2.0f;
    result->a = (p1.a + p2.a) / 2.0f;
    result->b = (p1.b + p2.b) / 2.0f;
}

// Use rsqrtf instead of custom fast_sqrtf
/*__device__ inline float fast_sqrtf(float x) {
    return sqrtf(x); // Add epsilon to avoid division by zero
}*/

/*__device__ inline float fast_sqrtf(float x) {
    float xhalf = 0.5f * x;
    int i = __float_as_int(x);             // Get bits for floating VALUE
    i = 0x5f375a86 - (i >> 1);             // Magic number
    x = __int_as_float(i);                 // Convert bits back to float
    x = x * (1.5f - xhalf * x * x);        // Newton step
    // Optionally repeat Newton step for higher accuracy
    return 1.0f / x;
}*/


// Shared Euclidean distance
__device__ inline void labDistance(lab p1, lab p2, float* result) {
    float dL = p1.L - p2.L;
    float da = p1.a - p2.a;
    float db = p1.b - p2.b;
    float distance_squared = dL * dL + da * da + db * db;
    *result = sqrtf(distance_squared);
}

__device__ inline void labDistance1(lab p1, lab p2, float* result) {
    float dL = abs(p1.L - p2.L);
    float da = abs(p1.a - p2.a);
    float db = abs(p1.b - p2.b);
    *result = dL + da + db;
}

// Shared Euclidean distance squared
__device__ inline void labDistanceSquared(lab p1, lab p2, float* result) {
    float dL = p1.L - p2.L;
    float da = p1.a - p2.a;
    float db = p1.b - p2.b;
    *result = dL * dL + da * da + db * db;
}

#else

inline void averageLAB(lab p1, lab p2, lab* result) {
    result->L = (p1.L + p2.L) / 2.0f;
    result->a = (p1.a + p2.a) / 2.0f;
    result->b = (p1.b + p2.b) / 2.0f;
}

 inline void labDistance(lab p1, lab p2, float* result) {
    float dL = p1.L - p2.L;
    float da = p1.a - p2.a;
    float db = p1.b - p2.b;
    *result = sqrt(dL * dL + da * da + db * db);
 }
#endif // __CUDACC__

