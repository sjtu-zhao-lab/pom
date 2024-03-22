
start_time=$(date +"%s")

echo ""
echo ">>> Step 5. Collecting experimental results..."
echo ""

declare -A baseline_latency
baseline_latency["gemm",32]=498753
baseline_latency["gemm",64]=3960961
baseline_latency["gemm",128]=31572225
baseline_latency["gemm",256]=252117505
baseline_latency["gemm",512]=2015101953
baseline_latency["gemm",1024]=19337840641
baseline_latency["gemm",2048]=154660769793
baseline_latency["gemm",4096]=1237118361601
baseline_latency["gemm",8192]=9896275755009

baseline_latency["bicg",32]=12353
baseline_latency["bicg",64]=49281
baseline_latency["bicg",128]=196865
baseline_latency["bicg",256]=786945
baseline_latency["bicg",512]=3146753
baseline_latency["bicg",1024]=14682113
baseline_latency["bicg",2048]=58724353
baseline_latency["bicg",4096]=234889217
baseline_latency["bicg",8192]=939540481

baseline_latency["gesummv",32]=12705
baseline_latency["gesummv",64]=49985
baseline_latency["gesummv",128]=198273
baseline_latency["gesummv",256]=789761
baseline_latency["gesummv",512]=3152385
baseline_latency["gesummv",1024]=14693377
baseline_latency["gesummv",2048]=58746881
baseline_latency["gesummv",4096]=234934273
baseline_latency["gesummv",8192]=939630593

baseline_latency["2mm",32]=697474
baseline_latency["2mm",64]=5542146
baseline_latency["2mm",128]=44188162
baseline_latency["2mm",256]=352912386
baseline_latency["2mm",512]=2820933634
baseline_latency["2mm",1024]=29004664834
baseline_latency["2mm",2048]=231982768130
baseline_latency["2mm",4096]=2199241375746
baseline_latency["2mm",8192]=17593058492418

baseline_latency["3mm",32]=1087683
baseline_latency["3mm",64]=8675715
baseline_latency["3mm",128]=69305091
baseline_latency["3mm",256]=554042883
baseline_latency["3mm",512]=4430760963
baseline_latency["3mm",1024]=45106599939
baseline_latency["3mm",2048]=360815013891
baseline_latency["3mm",4096]=2886369042435
baseline_latency["3mm",8192]=23090348212227


baseline_latency["jacobi",4096]=804925441
baseline_latency["jacobi2d",4096]=4668429217793
baseline_latency["heat",4096]=385699841
baseline_latency["seidel",4096]=4050540986369
baseline_latency["blur",4096]=3981925375233
baseline_latency["edgeDetect",4096]=2882880170
baseline_latency["gaussian",4096]=8694375278
baseline_latency["blur",4096]=2983445186
baseline_latency["vgg16",512]=3727670833
baseline_latency["resnet",512]=6602277677

declare -A execution_times

while read -r line; do
    if [[ $line =~ time\[(.*),(.*)\] ]]; then
        kernel=${BASH_REMATCH[1]}
        size=${BASH_REMATCH[2]}
        if [[ $line =~ ([0-9.]+)$ ]]; then
            time=${BASH_REMATCH[1]}
            time=$(printf "%.2f" $time)
            if [[ $time == .* ]]; then
                time="0$time"
            fi
            execution_times[$kernel,$size]=$time
        fi
    fi
done < "build/execution_times.txt"


result_file="experimental_results.csv"
# rm -f $result_file
if [ -f $result_file ]; then
    rm $result_file
fi
printf "/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" >> $result_file
printf "/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" >> $result_file
# printf "------------------------------Experimental Results----------------------------" >> $result_file

printf "\n" >> $result_file
printf "\n" >> $result_file
printf ">>> Results for TABLE III: \n" >> $result_file
printf "\n" >> $result_file
printf "%-20s %-25s %-20s %-20s %-20s %-20s %-15s %-15s\n" "Kernel" "Latency" "DSP" "FF" "LUT" "Power" "II" "Execution Time" >> $result_file

kernels=("gemm" "bicg" "gesummv" "2mm" "3mm" )
sizes=(4096)

for kernel in "${kernels[@]}"
do
    for size in "${sizes[@]}"
    do
        xml_file="samples/${kernel}/test_${kernel}_${size}/test_${kernel}_${size}/syn/report/csynth.xml"
        baseline=${baseline_latency[$kernel,$size]}
        best_case_latency=$(xmlstarlet sel -t -v "/profile/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency" $xml_file)
        acceleration=$(awk "BEGIN {printf \"%.1f\", $baseline / $best_case_latency}")
        acceleration_str="$acceleration x"
        latency_str="$best_case_latency($acceleration_str)"
        lut_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/LUT" $xml_file)
        lut_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/LUT" $xml_file)
        lut_percent=$((($lut_util * 100) / $lut_avail))
        lut_str="$lut_util($lut_percent%)"
        dsp_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/DSP" $xml_file)
        dsp_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/DSP" $xml_file)
        dsp_percent=$((($dsp_util * 100) / $dsp_avail))
        dsp_str="$dsp_util($dsp_percent%)"
        ff_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/FF" $xml_file)
        ff_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/FF" $xml_file)
        ff_percent=$((($ff_util * 100) / $ff_avail))
        ff_str="$ff_util($ff_percent%)"
        II=$(grep -m 1 -oP '(?<=<PipelineII>).*?(?=<\/PipelineII>)' $xml_file)
        time=${execution_times[$kernel,$size]}
        time_str="$time s"

        rpt_file="samples/${kernel}/test_${kernel}_power/test_${kernel}_power/impl/verilog/project.runs/impl_1/bd_0_wrapper_power_routed.rpt"
        total_power=$(grep "Total On-Chip Power (W)" "$rpt_file" | awk -F'|' '{print $3}' | tr -d '[:space:]')
        power_str="$total_power W"
        # echo "Total On-Chip Power: $total_power W"
        printf "%-20s %-25s %-20s %-20s %-20s %-20s %-15s %-15s\n" "${kernel}_${size}" "${latency_str}" "$dsp_str" "$ff_str" "$lut_str" "$power_str" "$II" "$time_str" >> $result_file
    done
done
printf "\n" >> $result_file
printf "\n" >> $result_file
printf ">>> Results for Fig. 12:\n" >> $result_file
printf "\n" >> $result_file
printf "%-20s %-25s %-20s %-20s %-20s %-15s %-15s\n" "Kernel" "Latency" "DSP" "FF" "LUT" "II" "Execution Time">> $result_file

kernels=("gemm" "bicg" "gesummv" "2mm" "3mm" )
sizes=(32 64 128 256 512 1024 2048 4096 8192)

for kernel in "${kernels[@]}"
do
    for size in "${sizes[@]}"
    do
        xml_file="samples/${kernel}/test_${kernel}_${size}/test_${kernel}_${size}/syn/report/csynth.xml"
        baseline=${baseline_latency[$kernel,$size]}
        best_case_latency=$(xmlstarlet sel -t -v "/profile/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency" $xml_file)
        acceleration=$(awk "BEGIN {printf \"%.1f\", $baseline / $best_case_latency}")
        acceleration_str="$acceleration x"
        latency_str="$best_case_latency($acceleration_str)"
        lut_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/LUT" $xml_file)
        lut_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/LUT" $xml_file)
        lut_percent=$((($lut_util * 100) / $lut_avail))
        lut_str="$lut_util($lut_percent%)"
        dsp_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/DSP" $xml_file)
        dsp_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/DSP" $xml_file)
        dsp_percent=$((($dsp_util * 100) / $dsp_avail))
        dsp_str="$dsp_util($dsp_percent%)"
        ff_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/FF" $xml_file)
        ff_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/FF" $xml_file)
        ff_percent=$((($ff_util * 100) / $ff_avail))
        ff_str="$ff_util($ff_percent%)"
        II=$(grep -m 1 -oP '(?<=<PipelineII>).*?(?=<\/PipelineII>)' $xml_file)
        time=${execution_times[$kernel,$size]}
        time_str="$time s"

        printf "%-20s %-25s %-20s %-20s %-20s %-15s %-15s\n" "${kernel}_${size}" "${latency_str}" "$dsp_str" "$ff_str" "$lut_str" "$II" "$time_str" >> $result_file
    done
    printf "\n" >> $result_file
done
printf "\n" >> $result_file
printf "\n" >> $result_file
printf ">>> Results for TABLE V and TABLE VII: \n" >> $result_file
printf "\n" >> $result_file
printf "%-20s %-25s %-20s %-20s %-20s %-15s %-15s\n" "Kernel" "Latency" "DSP" "FF" "LUT" "II" "Execution Time">> $result_file


kernels=("vgg16"  "resnet")
sizes=(512)
for kernel in "${kernels[@]}"
do
    for size in "${sizes[@]}"
    do
        xml_file="samples/${kernel}/test_${kernel}_${size}/test_${kernel}_${size}/syn/report/csynth.xml"
        baseline=${baseline_latency[$kernel,$size]}
        best_case_latency=$(xmlstarlet sel -t -v "/profile/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency" $xml_file)
        acceleration=$(awk "BEGIN {printf \"%.1f\", $baseline / $best_case_latency}")
        acceleration_str="$acceleration x"
        latency_str="$best_case_latency($acceleration_str)"
        lut_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/LUT" $xml_file)
        lut_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/LUT" $xml_file)
        lut_percent=$((($lut_util * 100) / $lut_avail))
        lut_str="$lut_util($lut_percent%)"
        dsp_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/DSP" $xml_file)
        dsp_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/DSP" $xml_file)
        dsp_percent=$((($dsp_util * 100) / $dsp_avail))
        dsp_str="$dsp_util($dsp_percent%)"
        ff_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/FF" $xml_file)
        ff_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/FF" $xml_file)
        ff_percent=$((($ff_util * 100) / $ff_avail))
        ff_str="$ff_util($ff_percent%)"
        II=$(grep -m 1 -oP '(?<=<PipelineII>).*?(?=<\/PipelineII>)' $xml_file)
        time=${execution_times[$kernel,$size]}
        time_str="$time s"

        printf "%-20s %-25s %-20s %-20s %-20s %-15s %-15s\n" "${kernel}_${size}" "${latency_str}" "$dsp_str" "$ff_str" "$lut_str"  "$II" "$time_str">> $result_file
    done
done




kernels=("edgeDetect" "gaussian" "blur" "jacobi" "jacobi2d" "heat" "seidel")
sizes=(4096)

for kernel in "${kernels[@]}"
do
    for size in "${sizes[@]}"
    do
        xml_file="samples/${kernel}/test_${kernel}_${size}/test_${kernel}_${size}/syn/report/csynth.xml"
        baseline=${baseline_latency[$kernel,$size]}
        best_case_latency=$(xmlstarlet sel -t -v "/profile/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency" $xml_file)
        acceleration=$(awk "BEGIN {printf \"%.1f\", $baseline / $best_case_latency}")
        acceleration_str="$acceleration x"
        latency_str="$best_case_latency($acceleration_str)"
        lut_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/LUT" $xml_file)
        lut_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/LUT" $xml_file)
        lut_percent=$((($lut_util * 100) / $lut_avail))
        lut_str="$lut_util($lut_percent%)"
        dsp_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/DSP" $xml_file)
        dsp_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/DSP" $xml_file)
        dsp_percent=$((($dsp_util * 100) / $dsp_avail))
        dsp_str="$dsp_util($dsp_percent%)"
        ff_util=$(xmlstarlet sel -t -v "/profile/AreaEstimates/Resources/FF" $xml_file)
        ff_avail=$(xmlstarlet sel -t -v "/profile/AreaEstimates/AvailableResources/FF" $xml_file)
        ff_percent=$((($ff_util * 100) / $ff_avail))
        ff_str="$ff_util($ff_percent%)"
        II=$(grep -m 1 -oP '(?<=<PipelineII>).*?(?=<\/PipelineII>)' $xml_file)
        time=${execution_times[$kernel,$size]}
        time_str="$time s"
        printf "%-20s %-25s %-20s %-20s %-20s %-15s %-15s\n" "${kernel}_${size}" "${latency_str}" "$dsp_str" "$ff_str" "$lut_str" "$II" "$time_str">> $result_file
    done
done
printf "\n" >> $result_file
printf ">>> Notes:\n" >> $result_file
printf "1. The Resnet speedup may be slightly different from the speedup in the paper. This is because we have modified some of the codegen methods and the 
overall latency is affected: we use fewer resources and achieve a slightly lower speedup. Note that the speedup of VGG-16 is sightly better than the speedup 
in the paper. \n" >> $result_file
printf "2. We are improving the optimization strategies for loops with small problem sizes and a final strategy for them have not been decided yet. So some 
of the small-problem-size results may be slightly different from the results in the paper. \n" >> $result_file
printf "\n" >> $result_file

printf "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" >> $result_file
#printf "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" >> $result_file
end_time=$(date +"%s")
execution_time=$(($end_time - $start_time))
echo ""
echo ">>> Step 5 has been finished!"
echo ">>> Step 5 Total Execution Time: $execution_time seconds"
echo ""
echo "The experimental results are collected in experimental_results.csv!"
