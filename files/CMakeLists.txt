cmake_minimum_required (VERSION 3.18)
project (PRPA)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)


pkg_check_modules(GStreamer REQUIRED gstreamer-1.0 IMPORTED_TARGET)
pkg_check_modules(GStreamerVideo REQUIRED gstreamer-video-1.0 IMPORTED_TARGET)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")


add_library(stb_image src/common/stb_image.h src/common/stb_image.c)
add_library(Compute
    src/Compute.cpp
    src/Compute.cu
    src/logo.c
        src/filters/filter_impl.hpp
        src/filters/filter_impl.cu
        src/logic/lab_converter/lab_converter.cpp
        src/logic/lab_converter/lab_converter.hpp
        src/logic/lab_converter/lab_converter.cu
        src/logic/lab_converter/lab_converter_utils.hpp
        src/logic/background/background_estimation.cpp
        src/logic/background/background_estimation.cu
        src/logic/background/background_estimation.hpp
        src/filters/erode_and_dilate/filter_erode_and_dilate.cpp
        src/filters/erode_and_dilate/filter_erode_and_dilate.cu
        src/filters/erode_and_dilate/filter_erode_and_dilate.hpp
        src/logic/hysteresis/hysteresis.cpp
        src/logic/hysteresis/hysteresis.cu
        src/logic/hysteresis/hysteresis.hpp
        src/logic/red_mask/red_mask.cpp
        src/logic/red_mask/red_mask.cu
        src/logic/red_mask/red_mask.hpp
)
target_link_libraries(Compute PRIVATE CUDA::cudart)

# New
# add_library(filter_impl src/filter_impl.h src/filter_impl.cu)
# target_link_libraries(filter_impl PRIVATE CUDA::cudart)

add_executable(stream src/stream.cpp src/gstfilter.c src/gstfilter.h)

# target_link_libraries(stream PUBLIC PkgConfig::GStreamerVideo cudart filter_impl)
target_link_libraries(stream PUBLIC PkgConfig::GStreamerVideo CUDA::cudart)

## La beauté de NIX....
# target_link_libraries(stream PUBLIC $<TARGET_FILE:filter_impl>)
target_link_libraries(stream PUBLIC $<TARGET_FILE:Compute>)
target_link_directories(stream PUBLIC "${CUDAToolkit_LIBRARY_DIR}")
set_target_properties(stream PROPERTIES
  INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)

