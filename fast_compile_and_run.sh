#!/bin/sh

# Check if build directory exists
BUILD_DIR="$HOME/build_test"
if [ ! -d "$BUILD_DIR" ]; then
    echo "$BUILD_DIR does not exist"
    exit 1
fi

# Navigate to the build directory and compile
cd "$BUILD_DIR"
make

echo "$0 succeeded"