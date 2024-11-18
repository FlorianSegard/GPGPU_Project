#!/bin/sh

# Set VERSION if not provided
if [ "$#" -eq 0 ]; then
    VERSION="Release"
    echo "No args provided, compilation flag set to Release"
else
    VERSION=$1
    echo "Compilation flag set to $VERSION"
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

echo "$0 succeeded"