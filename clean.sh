#!/bin/bash


folders=("gemm" "bicg" "gesummv" "2mm" "3mm" "edgeDetect" "gaussian" "blur" "vgg16"  "resnet" "jacobi" "jacobi2d" "heat" "seidel")

for folder in "${folders[@]}"
do
    rm -rf "samples/${folder}"/*
done

