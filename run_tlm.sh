#!/bin/bash

# variant (1: , 2:)
variant=2

# Size of simulated region
A=$(echo "40*40*1e6" | bc) # m2  

# water / depression fraction limit
frac_lim=1

# Time span
T=100

# time step
dt=1

# Nr. of ensemble runs
e_nr=5

# parameterization script
par_script="parameter/clim_param_func.py"

# OPTIONAL: initial lake data (if empty, simulation starts with no lakes)
file_ini_lakes="input/synthetic_lake_data.nc" 

# OPTIONAL: subset of lakes (if empty, all lakes from file_ini_lakes are used)
subset_lakes="" 

# forcing data
file_forcing="forcing/synthetic_tdd.txt"

#output and error log files
current_date=$(date +"%Y%m%d")
error_log="tlm_${current_date}_error.log"
echo "Error Log - $(date)" > $error_log

A_km2=$(echo "$A * 1e-6" | bc -l | awk '{printf "%g", $0}')
# Run scripts to simulate and plot
echo "Simulating $e_nr ensemble members for a region of $A_km2 sqkm and $T years. 
    Using parameter script: $par_script."



python3 scripts/model.py "$variant" "$A" "$frac_lim" "$T" "$dt" "$e_nr" "$par_script" "$file_ini_lakes" "$subset_lakes" "$file_forcing" 2>> $error_log
python3 scripts/plotting.py "$e_nr" 2>> $error_log
