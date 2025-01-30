#!/bin/bash

# Size of simulated region
A=$(echo "40*40*1e6" | bc)  # m2

# lake area data
lake_file="input/UTM54_cleaned.nc"

# climate data
climate_data="forcing/tdd_forcing.txt"

# subset of IDs (OPTIONAL. If left empty, all IDs from lake_file will be used)
subset_file="input/UTM54_North_ini_40x40.shp"

# drainage event file in the style of Chen et al 2023 (OPTIONAL. If left empty, drainage rate will be calculated from lake area data)
drainage_file="input/Drainage_events_UTM54_North_40x40.shp"

#output and error log files
current_date=$(date +"%Y%m%d")
error_log="tlm_${current_date}_error.log"
echo "Error Log - $(date)" > $error_log

# Run scripts to simulate and plot
python3 scripts/parameterization.py "$A" "$lake_file" "$climate_data" "$subset_file" "$drainage_file" 2>> $error_log

# Check the exit status of the Python script
if [ $? -ne 0 ]; then
    echo "Python script failed to run." >> $error_log
else
    echo "Obtained parameter from $lake_file"
fi
