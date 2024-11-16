#include "red_mask.hpp"

void red_mask_cpp(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height)
{
    return;
}

extern "C" {

static Parameters b_params;

void mask_init(Parameters* params)
{
    b_params = *params;
}

void mask_process_frame(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height)
{
    if (b_params.device == e_device_t::CPU)
        red_mask_cpp(hysteresis_buffer, rgb_buffer, width, height);

    else if (b_params.device == e_device_t::GPU) {
        red_mask_cu(hysteresis_buffer, rgb_buffer, width, height);
        cudaDeviceSynchronize();
    }
}

}