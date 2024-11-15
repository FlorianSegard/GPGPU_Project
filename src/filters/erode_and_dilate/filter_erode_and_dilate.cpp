#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <chrono>
#include "filter_erode_and_dilate.hpp"


void erode_cpp(const ImageView<float> input, ImageView<float> output, int width, int height, int opening_size) {
    for (int y = 0; y < height; ++y) {
        const lab* lineptr = (const lab*)((const std::byte*)input.buffer + y * input.stride);
        lab* lineptr_out = (lab*)((std::byte*)output.buffer + y * output.stride);
        
        for (int x = 0; x < width; ++x) {
            lab min_val = lineptr[x];
            
            // Kernel size 3x3
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const lab* neighbor = (const lab*)((const std::byte*)input.buffer + ny * input.stride);
                        min_val.L = std::min(min_val.L, neighbor[nx].L);
                        min_val.a = std::min(min_val.a, neighbor[nx].a);
                        min_val.b = std::min(min_val.b, neighbor[nx].b);
                    }
                }
            }
            lineptr_out[x] = min_val;
        }
    }
}

void dilate_cpp(const ImageView<float> input, ImageView<float> output, int width, int height, int opening_size) {
    for (int y = 0; y < height; ++y) {
        const lab* lineptr = (const lab*)((const std::byte*)input.buffer + y * input.stride);
        lab* lineptr_out = (lab*)((std::byte*)output.buffer + y * output.stride);
        
        for (int x = 0; x < width; ++x) {
            lab max_val = lineptr[x];
            
            // Kernel size 3x3
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const lab* neighbor = (const lab*)((const std::byte*)input.buffer + ny * input.stride);
                        max_val.L = std::max(max_val.L, neighbor[nx].L);
                        max_val.a = std::max(max_val.a, neighbor[nx].a);
                        max_val.b = std::max(max_val.b, neighbor[nx].b);
                    }
                }
            }
            lineptr_out[x] = max_val;
        }
    }
}


extern "C" {

static Parameters b_params;

void filter_init(Parameters *params) {
    b_params = *params;
}

void erode_process_frame(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size) {
    if (b_params.device == e_device_t::CPU)
        erode_cpp(input, output, width, height, opening_size);

    else if (b_params.device == e_device_t::GPU)
        erode_cu(input, output, width, height, opening_size);
}

void dilate_process_frame(ImageView<float> input, ImageView<float> output, int width, int height, int opening_size) {
    if (b_params.device == e_device_t::CPU)
        dilate_cpp(input, output, width, height, opening_size);

    else if (b_params.device == e_device_t::GPU)
        dilate_cu(input, output, width, height, opening_size);
}

}

/*
// Helper function to initialize test data
void initializeTestData(lab* data, int width, int height, std::ptrdiff_t stride) {
    for (int y = 0; y < height; y++) {
        lab* row = (lab*)((std::byte*)data + y * stride);
        for (int x = 0; x < width; x++) {
            // Create a simple pattern: higher values in the center
            float centerX = width / 2.0f;
            float centerY = height / 2.0f;
            float distance = std::sqrt(std::pow(x - centerX, 2) + std::pow(y - centerY, 2));
            row[x].L = 100.0f * (1.0f - distance / std::sqrt(centerX * centerX + centerY * centerY));
            row[x].a = 50.0f;
            row[x].b = 50.0f;
        }
    }
}

// Helper function to print a small section of the image
void printImageSection(const lab* data, int width, int height, std::ptrdiff_t stride, 
                      int startX, int startY, int sectionSize) {
    std::cout << "\nImage section (L channel only):\n";
    for (int y = startY; y < std::min(startY + sectionSize, height); y++) {
        const lab* row = (const lab*)((const std::byte*)data + y * stride);
        for (int x = startX; x < std::min(startX + sectionSize, width); x++) {
            printf("%.1f ", row[x].L);
        }
        std::cout << '\n';
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const std::ptrdiff_t stride = width * sizeof(lab);
    const int NUM_ITERATIONS = 100;  // Run multiple iterations for better timing

    // Allocate memory
    std::vector<lab> input(width * height);
    std::vector<lab> output(width * height);
    std::vector<lab> temp(width * height);

    // Initialize test data
    initializeTestData(input.data(), width, height, stride);

    // Print original image section
    std::cout << "Original image:";
    printImageSection(input.data(), width, height, stride, 0, 0, 5);

    // Timing variables
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();
    auto end = Clock::now();
    
    // Time erosion
    start = Clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        erode(input.data(), output.data(), width, height, stride);
    }
    end = Clock::now();
    std::cout << "\nCPU Erosion time (averaged over " << NUM_ITERATIONS << " iterations): " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / NUM_ITERATIONS 
              << " microseconds";
    
    std::cout << "\nAfter erosion:";
    printImageSection(output.data(), width, height, stride, 0, 0, 5);

    // Time dilation
    start = Clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        dilate(input.data(), output.data(), width, height, stride);
    }
    end = Clock::now();
    std::cout << "\nCPU Dilation time (averaged over " << NUM_ITERATIONS << " iterations): " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / NUM_ITERATIONS 
              << " microseconds";

    std::cout << "\nAfter dilation:";
    printImageSection(output.data(), width, height, stride, 0, 0, 5);

    // Time closing operation
    start = Clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++) {
        erode(input.data(), temp.data(), width, height, stride);
        dilate(temp.data(), output.data(), width, height, stride);
    }
    end = Clock::now();
    std::cout << "\nCPU Closing time (averaged over " << NUM_ITERATIONS << " iterations): " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / NUM_ITERATIONS 
              << " microseconds";

    std::cout << "\nAfter closing (erosion + dilation):";
    printImageSection(output.data(), width, height, stride, 0, 0, 5);

    return 0;
} */