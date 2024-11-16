#include "red_mask.hpp"

void red_mask_cpp(ImageView<bool> hysteresis_buffer, ImageView<rgb8> rgb_buffer, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        bool hyst_value = (bool*)((std::byte*)hysteresis_buffer.buffer + y * hysteresis_buffer.stride);
        rgb8* rgb_value = (rgb8*)((std::byte*)rgb_buffer.buffer + y * rgb_buffer.stride);

        for (int x = 0; x < width; ++x)
        {
            rgb_value[x].r = min(rgb_value[x].r + 127 * hyst_value[x], 255);
        }
    }
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