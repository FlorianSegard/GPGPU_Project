#!/bin/sh

LOGIN=$1
VIDEO=$2

if [ $# -ne 2 ]; then
    echo "TG"
    exit 1
fi

echo "Nix develop..."
nix develop

echo "Create project directory..."
mkdir -p "/home/$LOGIN/projet"

echo "Export project directory..."
export buildir="/home/$LOGIN/projet"

echo "Cmake with Debug mode..."
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Debug

#echo "Cmake with Release mode..."
#cmake -S . -B $buildir -DCMAKE_BUILD_TYPE=Release

echo "Stream gpu/cpu on video..."
$buildir/stream --mode=[gpu,cpu] "$VIDEO" --output=output.mp4