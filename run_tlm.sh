#!/bin/bash

# variant (1: , 2:)
variant=1

# Size of simulated region
A=$(echo "40*40*1e6" | bc) # m2

# water / depression fraction limit
frac_lim=0.3

# Time span
T=119 

# time step
dt=1

# Nr. of ensemble runs
e_nr=10

# parameterization script
par_script="parameter/clim_param_func"

# OPTIONAL: initial lake data (if empty, simulation starts with no lakes)
file_ini_lakes="input/UTM54_cleaned.nc"

# OPTIONAL: subset of lakes (if empty, all lakes from file_ini_lakes are used)
subset_lakes="input/UTM54_North_ini_40x40.shp"

# forcing data
file_forcing="forcing/tdd_forcing.txt"

#output and error log files
current_date=$(date +"%Y%m%d")
output_log="tlm_${current_date}_output.log"
error_log="tlm_${current_date}_error.log"
echo "Output Log - $(date)" > $output_log
echo "Error Log - $(date)" > $error_log

# Run scripts to simulate and plot
python3 scripts/tlm.py "$variant" "$A" "$frac_lim" "$T" "$dt" "$e_nr" "$par_script" "$file_ini_lakes" "$subset_lakes" "$file_forcing" >> $output_log 2>> $error_log
python3 scripts/plotting.py "$A" "$e_nr" "$dt" >> $output_log 2>> $error_log

echo "Simulating $e_nr ensemble members for a region of $A m^2 and $T years."