#include "hysteresis.hpp"
#include <vector>

/*
void hysteresis_thresholding_cpp(ImageView<float> input, std::byte* output, int width, int height, int input_pitch, int output_pitch, float threshold)
{
    for (int y = 0; y < height; ++y)
    {
        float *input_lineptr = reinterpret_cast<float *>(input.buffer + y * input_pitch);
        bool *output_lineptr = reinterpret_cast<bool *>(output + y * output_pitch);

        for (int x = 0; x < width; ++x)
        {
            float in_val = input_lineptr[x];
            output_lineptr[x] = in_val > threshold;
        }
    }
}

void hysteresis_kernel_cpp(ImageView<float> upper, ImageView<bool> lower, int width, int height, int upper_pitch, int lower_pitch, bool &has_changed_global)
{
    bool has_changed = true;

    while (has_changed)
    {
        has_changed = false;

        for (int y = 0; y < height; ++y)
        {
            bool *upper_lineptr = reinterpret_cast<bool *>(upper.buffer + y * upper_pitch);
            bool *lower_lineptr = reinterpret_cast<bool *>(lower.buffer + y * lower_pitch);

            for (int x = 0; x < width; ++x)
            {
                if (upper_lineptr[x])
                    continue;

                if (!lower_lineptr[x])
                    continue;

                if ((x > 0 && upper_lineptr[x - 1]) ||
                    (x < width - 1 && upper_lineptr[x + 1]) ||
                    (y > 0 && reinterpret_cast<bool *>(upper.buffer + (y - 1) * upper_pitch)[x]) ||
                    (y < height - 1 && reinterpret_cast<bool *>(upper.buffer + (y + 1) * upper_pitch)[x]))
                {
                    upper_lineptr[x] = true;
                    has_changed = true;
                    has_changed_global = true;
                }
            }
        }
    }
}

/// CPU Single threaded version of the Method
void hysteresis_cpp(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, int opened_input_pitch, int hysteresis_pitch, float lower_threshold, float upper_threshold)
{
    std::vector<std::byte> lower_threshold_input(width * height * sizeof(bool));
    int lower_threshold_pitch = width * sizeof(bool);

    hysteresis_thresholding_cpp(opened_input, lower_threshold_input.data(), width, height, opened_input_pitch, lower_threshold_pitch, lower_threshold);
    hysteresis_thresholding_cpp(opened_input, hysteresis, width, height, opened_input_pitch, hysteresis_pitch, upper_threshold);

    bool has_changed_global = true;

    while (has_changed_global)
    {
        has_changed_global = false;
        hysteresis_kernel_cpp(hysteresis, lower_threshold_input.data(), width, height, hysteresis_pitch, lower_threshold_pitch, has_changed_global);
    }
} */
void hysteresis_cpp(ImageView<float> opened_input, ImageView<bool> hysteresis, int width, int height, float lower_threshold, float upper_threshold)
{
    return;
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
    else if (g_params.device == e_device_t::GPU)
      hysteresis_cu(opened_input, hysteresis, width, height, lower_threshold, upper_threshold);
  }
}