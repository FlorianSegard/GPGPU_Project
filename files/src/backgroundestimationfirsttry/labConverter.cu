#include <iostream>
#include "labConverter.hpp"



// prob optional because in the backgroundestimation.cpp but need to test to check if it works because these are not gpu functions
// -------------------------------------------------------------------------------------------------------------------------------
float* XYZ_color_space(float r_linear, float g_linear, float b_linear)
{
    float matrix[9] = { 0.4124564, 0.3575761, 0.1804375,
                        0.2126729, 0.7151522, 0.0721750,
                        0.0193339, 0.1191920, 0.9503041 };

    float color_vector[3] = {r_linear, g_linear, b_linear};

    static float result[3];

    // basicly a matmult 3x3 * 3x1 matrix

    result[0] = matrix[0] * color_vector[0] + matrix[1] * color_vector[1] + matrix[2] * color_vector[2];
    result[1] = matrix[3] * color_vector[0] + matrix[4] * color_vector[1] + matrix[5] * color_vector[2];
    result[2] = matrix[6] * color_vector[0] + matrix[7] * color_vector[1] + matrix[8] * color_vector[2];

    return result;
}

// depends if it's rgb is already in linear
float get_linear(float r_g_b)
{
    if (r_g_b <= 0.04045)
        return r_g_b / 12.92;
    return pow((r_g_b + 0.055) / 1.055, 2.4);
}

float f(float t) {
    if (t > pow(6.0/29.0, 3)) {
        return pow(t, 1.0/3.0);
    } else {
        return (1.0/3.0) * pow(29.0/6.0, 2.0) * t + 4.0/29.0;
    }
}
// -------------------------------------------------------------------------------------------------------------------------------


__global__ void rgbtolab_converter_GPU(rgb* zero_image_buffer, std::ptrdiff_t stride_image, 
                                        lab* converted_image_buffer, std::ptrdiff_t stride_converted_image,
                                        int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        rgb* lineptr = (rgb*)((std::byte*)zero_image_buffer + y * stride_image);
        lab* lineptr_converted = (lab*)((std::byte*)converted_image_buffer + y * stride_converted_image);

        rgb8 currentpixel = lineptr[x];


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
        
        float X_n = X / 0.95047;
        float Y_n = Y / 1.0;
        float Z_n = Z / 1.08883;

        float X_n_sqrt = f(X_n);
        float Y_n_sqrt = f(Y_n);
        float Z_n_sqrt = f(Z_n);

        float L = 116 * Y_n_sqrt - 16;
        float a = 500 * (X_n_sqrt - Y_n_sqrt);
        float b = 200 * (Y_n_sqrt - Z_n_sqrt);

        lab currentpixel_lab = {L, a, b};

        lineptr_lab[x] = currentpixel_lab;
    }
}



__global__ void initialize_buffers(rgb* zero_image_buffer, int width, int height, std::ptrdiff_t stride_image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        rgb* lineptr = (rgb*)((std::byte*)zero_image_buffer + y * stride_image);

        lineptr[x] = {1, 10, 1};  
        lineptr_time[x] = 0; 
    }
}



int main()
{

    int width = 100;
    int height = 100;
    Image<rgb> zero_image(width, height, true);
    Image<lab> converted_image(width, height, true);


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // just initializing for the testing (zero image and initializing time map)
    initialize_buffers<<<threadsPerBlock, blocksPerGrid>>>(zero_image.buffer, zero_image.width, zero_image.height, zero_image.stride);
    cudaDeviceSynchronize();
    
    rgbtolab_converter_GPU<<<threadsPerBlock, blocksPerGrid>>>(zero_image.buffer, zero_image.stride, converted_image.buffer, converted_image.stride, zero_image.height, zero_image.stride)
    cudaDeviceSynchronize();

}
