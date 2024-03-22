#!/usr/bin/env bash
start=$(date +"%s")
echo ""
echo ">>> Step 2. Compiling the object files and Generating the optimized HLS C code..."
echo ""
cd build

if [ -f "execution_times.txt" ]; then
    rm "execution_times.txt"
fi


targets=("edgeDetect" "gaussian" "blur" "vgg16"  "resnet" "jacobi" "jacobi2d" "heat" "seidel")
for target in "${targets[@]}"
do
    cmake --build . --target "$target"
done

# Run building.
targets=("vgg16"  "resnet")
for target in "${targets[@]}"
do  
    start_time=$(date +%s.%N)
    ./bin/"$target"
    mlir_file="../samples/${target%.*}/test_${target%.*}.mlir"
    cpp_file="../samples/${target%.*}/test_${target%.*}.cpp"
    ../scalehls/build/bin/scalehls-opt $mlir_file \
        --scalehls-func-preprocess="top-func=test_${target%.*}" \
        --cse -canonicalize \
        --scalehls-qor-estimation="target-spec=../samples/config.json" \
        | ../scalehls/build/bin/scalehls-translate -emit-hlscpp > $cpp_file

    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)
    echo "time[$target,512] $execution_time" >> execution_times.txt

    echo "The HLS C code of $target: test_${target%.*}.cpp is generated!"
    echo ""
done

targets=("edgeDetect" "gaussian" "blur" "jacobi" "jacobi2d" "heat" "seidel")
N_value=4096
for target in "${targets[@]}"
do  
    start_time=$(date +%s.%N)
    ./bin/"$target"
    mlir_file="../samples/${target%.*}/test_${target%.*}_$N_value.mlir"
    cpp_file="../samples/${target%.*}/test_${target%.*}_$N_value.cpp"
    ../scalehls/build/bin/scalehls-opt $mlir_file \
        --scalehls-func-preprocess="top-func=test_${target%.*}_$N_value" \
        --cse -canonicalize \
        | ../scalehls/build/bin/scalehls-translate -emit-hlscpp > $cpp_file
    
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)
    echo "time[$target,4096] $execution_time" >> execution_times.txt

    echo "The HLS C code of $target: test_${target%.*}_$N_value.cpp is generated!"
    echo ""
done


# code_file="../samples/seidel/test_seidel_4096.cpp"
# target_file="../samples/seidel/test_seidel_4096.cpp"
file_path="../samples/seidel/test_seidel_4096.cpp"
# Insert lines after the third for loop
sed -i '/for (int v5 = max(0, ((v4 \/ 2) - 2046)); v5 < min(4094, (v4 \/ 2)); v5 += 1) {/a #pragma HLS PIPELINE II=1\n#pragma HLS LOOP_TRIPCOUNT avg=1366 max=1366 min=1366' "$file_path"

# Insert line before the first for loop
sed -i '/for (int v3 = 0; v3 < 4096; v3 += 1) {/i #pragma HLS DEPENDENCE dependent=false type=inter variable=v1' "$file_path"

echo "Lines inserted successfully into the file."


N_values=(32 64 128 256 512 1024 2048 4096 8192)

targets=("2mm.cpp" "3mm.cpp" "gemm.cpp" "bicg.cpp" "gesummv.cpp")


execute_command() {
    source_file=$1
    N=$2

    sed -i "s/#define N 4096/#define N $N/" ../testbench/$source_file

    cmake --build . --target ${source_file%.*}
    start_time=$(date +%s.%N)
    ./bin/${source_file%.*}

    mlir_file="../samples/${source_file%.*}/test_${source_file%.*}_${N}.mlir"
    cpp_file="../samples/${source_file%.*}/test_${source_file%.*}_${N}.cpp"
    ../scalehls/build/bin/scalehls-opt $mlir_file \
        --scalehls-func-preprocess="top-func=test_${source_file%.*}_${N}" \
        --cse -canonicalize \
        --scalehls-qor-estimation="target-spec=../samples/config.json" \
        | ../scalehls/build/bin/scalehls-translate -emit-hlscpp > $cpp_file

    sed -i "s/#define N [0-9]*/#define N 4096/" ../testbench/$source_file

    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)
    echo "time[${source_file%.*},$N] $execution_time" >> execution_times.txt


    echo ""
    echo "The HLS C code of $source_file: test_${source_file%.*}_${N}.cpp is generated!"
    echo ""
}


max_parallel=1

for N in "${N_values[@]}"
do
    for source_file in "${targets[@]}"
    do
        execute_command "$source_file" "$N" &
        if (( $(jobs | wc -l) >= $max_parallel )); then
            wait -n
        fi
    done
done
wait

end=$(date +"%s")
execution=$(($end - $start))
echo ""
echo ">>> Step 2 has been finished!"
echo ">>> Step 2 Total Execution Time: $execution seconds"
echo ""


