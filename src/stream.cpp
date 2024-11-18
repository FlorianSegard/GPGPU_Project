// Create a GStreamer pipeline to stream a video through our plugin

#include <gst/gst.h>
#include <format>
#include "gstfilter.h"
#include "common/argh.h"


static gboolean
plugin_init (GstPlugin * plugin)
{

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register (plugin, "myfilter", GST_RANK_NONE,
      GST_TYPE_MYFILTER);
}

static void my_code_init ()
{
  gst_plugin_register_static (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "myfilter",
    "Private elements of my application",
    plugin_init,
    "1.0",
    "LGPL",
    "",
    "",
    "");
}


int main(int argc, char* argv[])
{
  argh::parser cmdl(argc, argv);
  if (cmdl[{"-h", "--help"}])
  {
    g_printerr("Usage: %s --mode=[gpu,cpu] <filename> [options]\\n\\n", argv[0]);
    g_printerr("Options:\n");
    g_printerr("  --mode=[gpu,cpu]        Processing mode (required)\n");
    g_printerr("  --output, -o <file>     Output file (optional)\n");
    g_printerr("  --opening-size <int>     Opening size (default: 3)\n");
    g_printerr("  --th-low <int>          Low threshold (default: 3)\n");
    g_printerr("  --th-high <int>         High threshold (default: 30)\n");
    g_printerr("  --number-frame <int>     Number of frames (default: 10)\n");
    g_printerr("  -h, --help              Show this help message\n");
    return 0;
  }

  Parameters params;
  auto method = cmdl("mode", "cpu").str();
  auto filename = cmdl(1).str();
  auto output = cmdl({"-o", "--output"}, "").str();

  // New parameters
  auto opening_size = std::stoi(cmdl({"--opening-size"}, "3").str());
  auto th_low = std::stoi(cmdl({"--th-low"}, "3").str());
  auto th_high = std::stoi(cmdl({"--th-high"}, "30").str());
  auto number_frame = std::stoi(cmdl({"--number-frame"}, "100").str());

  g_print("===== Args =====\n");
  g_print("opening_size: %d\n", opening_size);
  g_print("th_low: %d\n", th_low);
  g_print("th_high: %d\n", th_high);
  g_print("number_frame: %d\n\n", number_frame);

  if (method == "cpu") {
      params.device = e_device_t::CPU;
  }
  else if (method == "gpu") {
      params.device = e_device_t::GPU;
  }
  else
  {
    g_printerr("Invalid method: %s\n", method.c_str());
    return 1;
  }

  g_debug("Using method: %s", method.c_str());
  gst_init(&argc, &argv);
  my_code_init();
  cpt_init(&params);

  // Create a GStreamer pipeline to stream a video through our plugin
  const char* pipe_str;
  g_print("Output: %s\n", output.c_str());
  if (output.empty())
    pipe_str = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter name=mfter ! videoconvert ! fpsdisplaysink sync=false";
  else
    pipe_str = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter name=mfter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink name=fdst";


  GError *error = NULL;
  auto pipeline = gst_parse_launch(pipe_str, &error);
  if (!pipeline)
  {
    g_printerr("Failed to create pipeline: %s\n", error->message);
    return 1;
  }

  auto filesrc = gst_bin_get_by_name (GST_BIN (pipeline), "fsrc");
  g_object_set (filesrc, "location", filename.c_str(), NULL);
  g_object_unref (filesrc);

  if (!output.empty())
  {
    auto filesink = gst_bin_get_by_name (GST_BIN (pipeline), "fdst");
    g_object_set (filesink, "location", output.c_str(), NULL);
    g_object_unref (filesink);
  }

  // Set myfilter properties
  auto filter = gst_bin_get_by_name(GST_BIN(pipeline), "mfter");
  if (filter) {
      g_object_set(filter, "opening_size", opening_size, NULL);
      g_object_set(filter, "th_low", th_low, NULL);
      g_object_set(filter, "th_high", th_high, NULL);
      g_object_set(filter, "number_frame", number_frame, NULL);
      g_object_unref(filter);
  }

  // Start the pipeline
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  // Wait until error or EOS
  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  // Free resources
  if (msg != nullptr)
    gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  return 0;
}
