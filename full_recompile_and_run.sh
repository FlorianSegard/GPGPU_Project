#!/bin/sh

# Ensure correct usage
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gpu/cpu> <video.mp4> [Debug/Release]"
    exit 1
fi

# Set VERSION if not provided
if [ "$#" -eq 2 ]; then
    VERSION="Release"
else
    VERSION=$3
fi

# Capture the current project path
echo "Copy current project path"
current_path=$(pwd)

# Delete project directory
echo "Delete project directory..."
rm -rf ~/build_test

# Go to project files
echo "Go to project files..."
cd "$current_path" || exit 1

# Create project directory
echo "Create project directory..."
mkdir ~/build_test

# Export project directory
echo "Export project directory..."
export BUILD_DIR=~/build_test

# Run CMake with the specified mode
echo "CMake with ${VERSION} mode..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$VERSION"

# Build the project
cd "$BUILD_DIR" || exit 1
make

DEVICE=$1
VIDEO=$2
"$BUILD_DIR/stream" --mode=$DEVICE $VIDEO

echo "$0 succeeded"