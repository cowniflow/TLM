#!/bin/bash

# variant (1: , 2:)
variant=2

# Size of simulated region
A=$(echo "40*40*1e6" | bc) # m2

# water / depression fraction limit
frac_lim=0.5

# Time span
T=1000

# time step
dt=1

# Nr. of ensemble runs
e_nr=1

# parameterization script
par_script="parameter/idealized/clim_param_func_oscillating.py"

# OPTIONAL: initial lake data (if empty, simulation starts with no lakes)
file_ini_lakes="" #"input/UTM54_cleaned.nc"

# OPTIONAL: subset of lakes (if empty, all lakes from file_ini_lakes are used)
subset_lakes="" #"input/UTM54_North_ini_40x40.shp"

# forcing data
file_forcing="forcing/tdd_forcing.txt"

#output and error log files
current_date=$(date +"%Y%m%d")
error_log="tlm_${current_date}_error.log"
echo "Error Log - $(date)" > $error_log

# Run scripts to simulate and plot
python3 scripts/model.py "$variant" "$A" "$frac_lim" "$T" "$dt" "$e_nr" "$par_script" "$file_ini_lakes" "$subset_lakes" "$file_forcing" 2>> $error_log
python3 scripts/plotting.py "$A" "$e_nr" "$dt" 2>> $error_log

echo "Creating animations for "