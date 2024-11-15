#!/bin/sh

# Ensure correct usage
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu/cpu> <video.mp4>"
    exit 1
fi

# Check if build directory exists
BUILD_DIR="$HOME/build_test"
if [ ! -d "$BUILD_DIR" ]; then
    echo "$BUILD_DIR does not exist"
    exit 1
fi

# Navigate to the build directory and compile
cd "$BUILD_DIR"
make

# Run the video file
DEVICE=$1
VIDEO=$2
"$BUILD_DIR/stream" --mode=$DEVICE $VIDEO

echo "$0 succeeded"