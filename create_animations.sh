#!/bin/bash

# Size of simulated region
A=$(echo "40*40*1e6" | bc) # m2

# Nr. of ensemble runs
e_nr=1

# Time span
T=1000

# time step
dt=1

# folder with lake data
folder_lakes="output/"

#output and error log files
current_date=$(date +"%Y%m%d")
error_log="animations_${current_date}_error.log"
echo "Error Log - $(date)" > $error_log

# Run script
python3 scripts/animations.py "$A" "$e_nr" "$T" "$dt" "$folder_lakes" 2>> $error_log

