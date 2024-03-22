#!/bin/bash
start_time=$(date +"%s")
echo ""
echo ">>> Step 3. Generating scripts for running Vitis_HLS..."
echo ""

examples=("vgg16" "resnet")
sizes=(512)
for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        script_name="script_${size}.tcl"
        
        cat > "samples/${example}/${script_name}" <<EOL
open_project -reset "test_${example}_${size}"
set_top test_${example}
add_files test_${example}.cpp
open_solution "test_${example}_${size}"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
set_directive_pipeline -off "test_${example}"
csynth_design
close_project
exit
EOL
    done
done

#!/bin/bash

examples=("2mm" "3mm" "gemm" "bicg" "gesummv")
sizes=(32 64 128 256 512 1024 2048 4096 8192)

# for example in "${examples[@]}"
# do  
#     for size in "${sizes[@]}"
#     do
#         script_name="script_${size}.tcl"
        
#         cat > "samples/${example}/${script_name}" <<EOL
# open_project -reset "test_${example}_${size}"
# set_top test_${example}_${size}
# add_files test_${example}_${size}.cpp
# open_solution "test_${example}_${size}"
# set_part {xc7z020clg400-1}
# create_clock -period 10 -name default
# csynth_design
# EOL

#         if [ "$size" -eq 4096 ]; then
#             echo "export_design -evaluate verilog -format ip_catalog -version 2.0.1" >> "samples/${example}/${script_name}"
#         fi

#         echo "close_project" >> "samples/${example}/${script_name}"
#         echo "exit" >> "samples/${example}/${script_name}"
#     done
# done
for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        script_name="script_${size}.tcl"
        
        cat > "samples/${example}/${script_name}" <<EOL
open_project -reset "test_${example}_${size}"
set_top test_${example}_${size}
add_files test_${example}_${size}.cpp
open_solution "test_${example}_${size}"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csynth_design
close_project
exit
EOL
    done
done

examples=("2mm" "3mm" "gemm" "bicg" "gesummv")
sizes=(4096)

for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        script_name="script_power.tcl"
        
        cat > "samples/${example}/${script_name}" <<EOL
open_project -reset "test_${example}_power"
set_top test_${example}_${size}
add_files test_${example}_${size}.cpp
open_solution "test_${example}_power"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csynth_design
EOL

        if [ "$size" -eq 4096 ]; then
            echo "export_design -evaluate verilog -format ip_catalog -version 2.0.1" >> "samples/${example}/${script_name}"
        fi

        echo "close_project" >> "samples/${example}/${script_name}"
        echo "exit" >> "samples/${example}/${script_name}"
    done
done



examples=("edgeDetect" "gaussian" "blur" "jacobi" "jacobi2d" "heat" "seidel")
sizes=(4096)
for example in "${examples[@]}"
do  
    for size in "${sizes[@]}"
    do
        script_name="script_${size}.tcl"
        
        cat > "samples/${example}/${script_name}" <<EOL
open_project -reset "test_${example}_${size}"
set_top test_${example}_${size}
add_files test_${example}_${size}.cpp
open_solution "test_${example}_${size}"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csynth_design
close_project
exit
EOL
    done
done

# echo ""
# echo ">>> Step 3 has been finished!"
# echo ""

end_time=$(date +"%s")
execution_time=$(($end_time - $start_time))
echo ""
echo ">>> Step 3 has been finished!"
echo ">>> Step 3 Total Execution Time: $execution_time seconds"
echo ""
