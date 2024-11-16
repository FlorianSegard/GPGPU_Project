#include <iostream>
#include "../logic/lab_converter/lab_converter.hpp"
#include "../logic/background/background_estimation.hpp"
#include "erode_and_dilate/filter_erode_and_dilate.hpp"
#include "../logic/hysteresis/hysteresis.hpp"
#include "../logic/red_mask/red_mask.hpp"
#include "filter_impl.hpp"
#include <cstring>

Image<lab> current_background;
Image<lab> candidate_background;
Image<int> current_time_pixels;
bool isInitialized = false;

void initialize_globals_cpp(int width, int height, ImageView<lab> lab_image) {
    current_background = Image<lab>(width, height, false);
    candidate_background = Image<lab>(width, height, false);
    current_time_pixels = Image<int>(width, height, false);
    isInitialized = true;

    memcpy(current_background.buffer, lab_image.buffer, height * width * sizeof(lab));
    memcpy(candidate_background.buffer, lab_image.buffer, height * width * sizeof(lab));
}

void filter_impl_cpp(uint8_t* pixels_buffer, int width, int height, int plane_stride, const char* bg_uri,
                     int opening_size, int th_low, int th_high, int bg_sampling_rate, int bg_number_frame)
{
    std::cout << "CPU implem" << std::endl;
    // Init device and global variables
    Parameters params;
    params.device = CPU;


    // Clone pixels_buffer inside new allocated rgb_buffer
    Image<rgb8> rgb_image(width, height, false);
    memcpy(rgb_image.buffer, pixels_buffer, height * plane_stride);


    // Allocate lab converted image buffer
    lab_conv_init(&params);
    Image<lab> lab_image(width, height, false);

    // Convert RGB to LAB -> result stored inside lab_buffer
    lab_conv_process_frame(rgb_image, lab_image);


    if (!isInitialized)
        initialize_globals_cpp(width, height, lab_image);

    // Update background and get residual image
    background_init(&params);
    Image<float> residual_image(width, height, false);

    background_process_frame(lab_image, current_background, candidate_background,
                             current_time_pixels, residual_image, bg_number_frame);

    // Alloc and perform eroding operation
    filter_init(&params);
    Image<float> erode_image(width, height, false);

    erode_process_frame(
            residual_image, erode_image,
            width, height, opening_size / 2
    );

    // Keep old residual_image alloc and perform dilatation operation
    dilate_process_frame(
            erode_image, residual_image,
            width, height, opening_size / 2
    );

    // Alloc and perform hysteresis operation
    hysteresis_init(&params);
    Image<bool> hysteresis_image(width, height, false);

    hysteresis_process_frame(
            residual_image, hysteresis_image,
            width, height, th_low, th_high
    );


    // Alloc and red mask operation
    mask_init(&params);
    mask_process_frame(hysteresis_image, rgb_image, width, height);


    // Copy result back to pixels_buffer
    memcpy(pixels_buffer, rgb_image.buffer, height * plane_stride);
}