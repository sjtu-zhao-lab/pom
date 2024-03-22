#!/bin/bash
start_time=$(date +"%s")
echo ""
echo ">>> Step 4. Synthesising the optimized HLS C code..."
echo ""
export LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1

execute_tcl() {
    example=$1
    size=$2
    script_name="script_${size}.tcl"
    cd "samples/${example}"
    vitis_hls -f "$script_name"
    cd -  
}

execute_tcl2() {
    example=$1
    size=$2
    script_name="script_power.tcl"
    cd "samples/${example}"
    vitis_hls -f "$script_name"
    cd -  
}

max_parallel=20


examples=("edgeDetect" "gaussian" "blur" "jacobi" "jacobi2d" "heat" "seidel")
sizes=(4096)
for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        execute_tcl "$example" "$size" &
        if (( $(jobs | wc -l) >= $max_parallel )); then
            wait -n
        fi
    done
done
wait


examples=("vgg16"  "resnet")
sizes=(512)
for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        execute_tcl "$example" "$size" &
        if (( $(jobs | wc -l) >= $max_parallel )); then
            wait -n
        fi
    done
done
wait


sizes=(32 64 128 256 512 1024 2048 4096 8192)

examples=("2mm" "3mm" "gemm" "bicg" "gesummv")

for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        execute_tcl "$example" "$size" &
        if (( $(jobs | wc -l) >= $max_parallel )); then
            wait -n
        fi
    done
done
wait

examples=("2mm" "3mm" "gemm" "gesummv")
sizes=(4096)
for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        execute_tcl2 "$example" "$size" &
        if (( $(jobs | wc -l) >= $max_parallel )); then
            wait -n
        fi
    done
done
wait

examples=("bicg")
sizes=(4096)
for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        execute_tcl2 "$example" "$size" &
        if (( $(jobs | wc -l) >= $max_parallel )); then
            wait -n
        fi
    done
done
wait

cd /usr/src/workspace



end_time=$(date +"%s")
execution_time=$(($end_time - $start_time))
echo ""
echo ">>> Step 4 has been finished!"
echo ">>> Step 4 Total Execution Time: $execution_time seconds"
echo ""

