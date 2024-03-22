#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
POM_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# POM_DIR="$(dirname "$CURRENT_DIR")"

start_time=$(date +"%s")
echo ""
echo ">>> Step 1. Building POM ..."
echo ""

# Got to the build directory.
cd "${POM_DIR}"
mkdir -p build
cd build

if [ ! -f "CMakeCache.txt" ]; then
    LLVM_DIR="${POM_DIR}/scalehls/build/lib/cmake/llvm" \
    MLIR_DIR="${POM_DIR}/scalehls/build/lib/cmake/mlir" \
    cmake -G Ninja .. 
    # -DMLIR_DIR="${POM_DIR}/scalehls/build/lib/cmake/mlir" \
    # -DLLVM_EXTERNAL_LIT="${POM_DIR}/scalehls/build/bin/llvm-lit" 
fi

cd ../
# Run building.
# targets=("edgeDetect" "gaussian" "blur" "vgg16"  "resnet" "jacobi" "jacobi2d" "heat" "seidel")


echo ""
echo ">>> Step 2. Initializing samples/{testbench}"
echo ""
folders=("gemm" "bicg" "gesummv" "2mm" "3mm" "edgeDetect" "gaussian" "blur" "vgg16"  "resnet18" "jacobi" "jacobi2d" "heat" "seidel")

for folder in "${folders[@]}"
do
    mkdir -p samples/"${folder}"
done


# for target in "${targets[@]}"
# do
#     cmake --build . --target "$target"
# done
# end_time=$(date +"%s")
# execution_time=$(($end_time - $start_time))
echo ""
echo ">>> Building finished!"
echo ""
