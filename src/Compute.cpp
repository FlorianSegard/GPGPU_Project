#include "Compute.hpp"
#include "common/Image.hpp"

#include <chrono>
#include <thread>


#include "filters/filter_impl.hpp"

/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
// void compute_cpp(ImageView<rgb8> in);


/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
// void compute_cu(ImageView<rgb8> in);






/// CPU Single threaded version of the Method
// void compute_cpp(ImageView<rgb8> in)
// {
//   for (int y = 0; y < in.height; ++y)
//   {
//     rgb8* lineptr = (rgb8*)((std::byte*)in.buffer + y * in.stride);
//     for (int x = 0; x < in.width; ++x)
//     {
//       lineptr[x].r = 0; // Back out red component

//       if (x < logo_width && y < logo_height)
//       {
//         float alpha  = logo_data[y * logo_width + x] / 255.f;
//         lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1 - alpha) * 255);
//         lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1 - alpha) * 255);
//       }
//     }
//   }

//   // You can fake a long-time process with sleep
//    {
//      using namespace std::chrono_literals;
//      std::this_thread::sleep_for(50ms);
//    }
// }


extern "C" {

  static Parameters g_params;

  void cpt_init(Parameters* params)
  {
    g_params = *params;
  }

  void cpt_process_frame(uint8_t* buffer, int width, int height, int stride, e_device_t device, const char* bg_uri,
                         int opening_size, int th_low, int th_high, int bg_sampling_rate, int bg_number_frame)
  {
      filter_impl(buffer, width, height, stride, device, bg_uri, opening_size, th_low, th_high, bg_sampling_rate, bg_number_frame);
  }
}