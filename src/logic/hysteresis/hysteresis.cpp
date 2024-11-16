#include "hysteresis.hpp"
#include <vector>


void hysteresis_thresholding_cpp(ImageView<float> input, ImageView<bool> output, int width, int height, float threshold)
{
    for (int y = 0; y < height; ++y)
    {
        float* input_lineptr = (float *)((std::byte*)input.buffer + y * input.stride);
        bool* output_lineptr = (bool *)((std::byte*)output.buffer  + y * output.stride);

        for (int x = 0; x < width; ++x)
        {
            float in_val = input_lineptr[x];
            output_lineptr[x] = in_val > threshold;
        }
    }
}

void hysteresis_kernel_cpp(ImageView<bool> upper, ImageView<bool> lower, int width, int height, bool &has_changed_global)
{
    bool has_changed = true;

    while (has_changed)
    {
        has_changed = false;

        for (int y = 0; y < height; ++y)
        {
            bool* upper_lineptr = (bool*)((std::byte*)upper.buffer + y * upper.stride);
            bool* lower_lineptr = (bool*)((std::byte*)lower.buffer + y * lower.stride);

            for (int x = 0; x < width; ++x)
            {
                if (upper_lineptr[x])
                    continue;

                if (!lower_lineptr[x])
                    continue;

                if ((x > 0 && upper_lineptr[x - 1]) ||
                    (x < width - 1 && upper_lineptr[x + 1]) ||
                    (y > 0 && (bool*)((std::byte*)upper.buffer + (y - 1) * upper.stride)[x]) ||
                    (y < height - 1 && (bool*)((std::byte*)upper.buffer + (y + 1) * upper.stride)[x]))
                {
                    upper_lineptr[x] = true;
                    has_changed = true;
                    has_changed_global = true;
                }
            }
        }
    }
}

void hysteresis_cpp(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold)
{
    Image<bool> lower_threshold_input(width, height, false);

    hysteresis_thresholding_cpp(opened_input, lower_threshold_input, width, height, lower_threshold);
    hysteresis_thresholding_cpp(opened_input, hysteresis, width, height, upper_threshold);

    bool has_changed_global = true;

    while (has_changed_global)
    {
        has_changed_global = false;
        hysteresis_kernel_cpp(hysteresis, lower_threshold_input, width, height, has_changed_global);
    }
}

extern "C" {

  static Parameters g_params;

  void hysteresis_init(Parameters* params)
  {
    g_params = *params;
  }

  void hysteresis_process_frame(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold)
  {
    if (g_params.device == e_device_t::CPU)
      hysteresis_cpp(opened_input, hysteresis, width, height, lower_threshold, upper_threshold);
    else if (g_params.device == e_device_t::GPU) {
        hysteresis_cu(opened_input, hysteresis, width, height, lower_threshold, upper_threshold);
        cudaDeviceSynchronize();
    }
  }
}