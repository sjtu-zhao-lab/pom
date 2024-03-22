
start_time=$(date +"%s")
echo ""
echo ">>> Start the experiment workflow"
echo ""
./build-pom.sh


./run-code.sh


./tcl-gen.sh


./vitis-reports.sh


./results-gen.sh

end_time=$(date +"%s")
execution_time=$(($end_time - $start_time))
echo ""
echo ">>> All Steps have been finished!"
echo ">>> Total Execution Time: $execution_time seconds"
echo ""
