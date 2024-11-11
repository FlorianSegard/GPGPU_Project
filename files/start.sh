#!/bin/sh

# LOGIN=$1
# VIDEO=$2
# 
# if [ $# -ne 2 ]; then
#     echo "TG"
#     exit 1
# fi
echo "Delete project directory..."
rm -rf ~/build_test

echo "Go to project"
cd /afs/cri.epita.fr/user/a/al/alexandre.devaux-riviere/u/GPGPU_Project/files

#echo "Nix develop..."
#nix develop

echo "Create project directory..."
mkdir ~/build_test

echo "Export project directory..."
export buildir=~/build_test

echo "Cmake with Debug mode..."
cmake -S . -B $buildir -DCMAKE_BUILD_TYPE=Debug

cd ~/build_test
make
#echo "Cmake with Release mode..."
#cmake -S . -B $buildir -DCMAKE_BUILD_TYPE=Release

#echo "Stream gpu/cpu on video..."
#$buildir/stream --mode=[gpu,cpu] "$VIDEO" --output=output.mp4
