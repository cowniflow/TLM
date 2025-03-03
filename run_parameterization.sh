#!/bin/bash

# Size of study region
A=$(echo "40*40*1e6" | bc)  # m2

# lake area data [in m2]
lake_file="input/synthetic_lake_data.nc"

# climate data
climate_data="forcing/synthetic_tdd.txt"

# OPTIONAL: subset of IDs (if empty, all IDs from lake_file will be used)
subset_file=""

# drainage event file in the style of Chen et al 2023 (OPTIONAL. If left empty, drainage rate will be calculated from lake area data)
drainage_file="" 

#output and error log files
current_date=$(date +"%Y%m%d")
error_log="parameterization_${current_date}_error.log"
echo "Error Log - $(date)" > $error_log

# Run scripts to simulate and plot
python3 scripts/parameterization.py "$A" "$lake_file" "$climate_data" "$subset_file" "$drainage_file" 2>> $error_log

# Check the exit status of the Python script
if [ $? -ne 0 ]; then
    echo "Python script failed to run." >> $error_log
else
    echo "Obtained parameter from $lake_file"
fi
