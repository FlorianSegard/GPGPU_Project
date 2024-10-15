/* GStreamer
 * Copyright (C) 2023 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstcudafilter
 *
 * The cudafilter element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v fakesrc ! cudafilter ! FIXME ! fakesink
 * ]|
 * FIXME Describe what the pipeline does.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include "gstfilter.h"
#include "Compute.hpp"

/* properties */
enum {
  PROP_0,
  PROP_METHOD
  /* FILL ME */
};

GST_DEBUG_CATEGORY_STATIC (gst_myfilter_debug_category);
#define GST_CAT_DEFAULT gst_myfilter_debug_category

/* prototypes */


static void gst_myfilter_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_myfilter_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_myfilter_dispose (GObject * object);
static void gst_myfilter_finalize (GObject * object);

static gboolean gst_myfilter_start (GstBaseTransform * trans);
static gboolean gst_myfilter_stop (GstBaseTransform * trans);
static gboolean gst_myfilter_set_info (GstVideoFilter * filter, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info);

//static GstFlowReturn gst_myfilter_transform_frame (GstVideoFilter * filter, GstVideoFrame * inframe, GstVideoFrame * outframe);
static GstFlowReturn gst_myfilter_transform_frame_ip (GstVideoFilter * filter, GstVideoFrame * frame);

/* pad templates */

/* FIXME: add/remove formats you can handle */
#define VIDEO_SRC_CAPS \
    GST_VIDEO_CAPS_MAKE("{ RGB }")

/* FIXME: add/remove formats you can handle */
#define VIDEO_SINK_CAPS \
    GST_VIDEO_CAPS_MAKE("{ RGB }")


/* class initialization */

G_DEFINE_TYPE_WITH_CODE (GstMyFilter, gst_myfilter, GST_TYPE_VIDEO_FILTER,
  GST_DEBUG_CATEGORY_INIT (gst_myfilter_debug_category, "cudafilter", 0,
  "debug category for cudafilter element"));

static void
gst_myfilter_class_init (GstMyFilterClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);
  GstVideoFilterClass *video_filter_class = GST_VIDEO_FILTER_CLASS (klass);

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  gst_element_class_add_pad_template (GST_ELEMENT_CLASS(klass),
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
        gst_caps_from_string (VIDEO_SRC_CAPS)));
  gst_element_class_add_pad_template (GST_ELEMENT_CLASS(klass),
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        gst_caps_from_string (VIDEO_SINK_CAPS)));

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "FIXME Long name", "Generic", "FIXME Description",
      "FIXME <fixme@example.com>");


  gobject_class->set_property = gst_myfilter_set_property;
  gobject_class->get_property = gst_myfilter_get_property;
  gobject_class->dispose = gst_myfilter_dispose;
  gobject_class->finalize = gst_myfilter_finalize;
  g_object_class_install_property (gobject_class, PROP_METHOD,
    g_param_spec_int ("method", "Method", "The method to use", 0, 3, 0, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_myfilter_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_myfilter_stop);
  video_filter_class->set_info = GST_DEBUG_FUNCPTR (gst_myfilter_set_info);
  //video_filter_class->transform_frame = GST_DEBUG_FUNCPTR (gst_myfilter_transform_frame);
  video_filter_class->transform_frame_ip = GST_DEBUG_FUNCPTR (gst_myfilter_transform_frame_ip);

}

static void
gst_myfilter_init (GstMyFilter *cudafilter)
{
}

void
gst_myfilter_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstMyFilter *cudafilter = GST_MYFILTER (object);

  GST_DEBUG_OBJECT (cudafilter, "set_property");

  switch (property_id) {
    case PROP_METHOD:
      cudafilter->device = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_myfilter_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstMyFilter *cudafilter = GST_MYFILTER (object);

  GST_DEBUG_OBJECT (cudafilter, "get_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_myfilter_dispose (GObject * object)
{
  GstMyFilter *cudafilter = GST_MYFILTER (object);

  GST_DEBUG_OBJECT (cudafilter, "dispose");

  /* clean up as possible.  may be called multiple times */

  G_OBJECT_CLASS (gst_myfilter_parent_class)->dispose (object);
}

void
gst_myfilter_finalize (GObject * object)
{
  GstMyFilter *cudafilter = GST_MYFILTER (object);

  GST_DEBUG_OBJECT (cudafilter, "finalize");

  /* clean up object here */

  G_OBJECT_CLASS (gst_myfilter_parent_class)->finalize (object);
}

static gboolean
gst_myfilter_start (GstBaseTransform * trans)
{
  GstMyFilter *cudafilter = GST_MYFILTER (trans);

  GST_DEBUG_OBJECT (cudafilter, "start");

  return TRUE;
}

static gboolean
gst_myfilter_stop (GstBaseTransform * trans)
{
  GstMyFilter *cudafilter = GST_MYFILTER (trans);

  GST_DEBUG_OBJECT (cudafilter, "stop");

  return TRUE;
}

static gboolean
gst_myfilter_set_info (GstVideoFilter * filter, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info)
{
  GstMyFilter *cudafilter = GST_MYFILTER (filter);

  GST_DEBUG_OBJECT (cudafilter, "set_info");

  return TRUE;
}

/* transform */
/* Uncomment if you want a transform not inplace

static GstFlowReturn
gst_myfilter_transform_frame (GstVideoFilter * filter, GstVideoFrame * inframe,
    GstVideoFrame * outframe)
{
  GstCudaFilter *cudafilter = GST_MYFILTER (filter);

  GST_DEBUG_OBJECT (cudafilter, "transform_frame");

  return GST_FLOW_OK;
}
*/

static GstFlowReturn
gst_myfilter_transform_frame_ip (GstVideoFilter * filter, GstVideoFrame * frame)
{
  GstMyFilter *cudafilter = GST_MYFILTER (filter);

  GST_DEBUG_OBJECT (cudafilter, "transform_frame_ip");



  int width = GST_VIDEO_FRAME_COMP_WIDTH(frame, 0);
  int height = GST_VIDEO_FRAME_COMP_HEIGHT(frame, 0);

  uint8_t* pixels = GST_VIDEO_FRAME_PLANE_DATA(frame, 0);
  int plane_stride = GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0);
  int pixel_stride = GST_VIDEO_FRAME_COMP_PSTRIDE(frame, 0);

  // get the timestamp of the frame
  GstClockTime timestamp = GST_BUFFER_TIMESTAMP(frame->buffer);


  //filter_impl(pixels, width, height, plane_stride, pixel_stride, timestamp);
  g_assert(pixel_stride == 3); // RGB

  cpt_process_frame(pixels, width, height, plane_stride);


  return GST_FLOW_OK;
}

// static gboolean
// plugin_init (GstPlugin * plugin)
// {
// 
//   /* FIXME Remember to set the rank if it's an element that is meant
//      to be autoplugged by decodebin. */
//   return gst_element_register (plugin, "myfilter", GST_RANK_NONE,
//       GST_TYPE_MYFILTER);
// }

/* FIXME: these are normally defined by the GStreamer build system.
   If you are creating an element to be included in gst-plugins-*,
   remove these, as they're always defined.  Otherwise, edit as
   appropriate for your external plugin package. */

// #ifndef VERSION
// #define VERSION "0.0.FIXME"
// #endif
// #ifndef PACKAGE
// #define PACKAGE "FIXME_package"
// #endif
// #ifndef PACKAGE_NAME
// #define PACKAGE_NAME "FIXME_package_name"
// #endif
// #ifndef GST_PACKAGE_ORIGIN
// #define GST_PACKAGE_ORIGIN "http://FIXME.org/"
// #endif

//GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
//    GST_VERSION_MINOR,
//    myfilter,
//    "FIXME plugin description",
//    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

