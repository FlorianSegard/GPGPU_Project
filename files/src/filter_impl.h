#pragma once

#ifdef __cplusplus
    extern "C" {
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

void filter_impl(uint8_t* pixels_buffer, int width, int height, int plane_stride, int pixel_stride, GstClockTime timestamp);

#ifdef __cplusplus
    }
#endif