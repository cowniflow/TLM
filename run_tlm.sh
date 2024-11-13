#!/bin/bash

# variant (1: , 2:)
variant = 1

# Size of simulated region
A = 40*40 * 1e6 # m2

# water / depression fraction limit
frac_lim = 0.3

# Time span
T=119 

# time step
dt=1

# Nr. of ensemble runs
e_nr=10

# parameterization script
par_script="paramter/clim_param_func"

# initial lake data
file_ini_lakes = "input/UTM54_cleaned.nc"

# OPTIONAL: subset of lakes
subset_lakes = "input/UTM54_North_ini_40x40.shp"

# forcing data
file_forcing = "forcing/tdd_forcing.txt"



# Run scripts to simulate and plot
python3 tlm.py "$variant" "$A" "$frac_lim" "$T" "$dt" "$e_nr" "$par_script" "$file_ini_lakes" "$subset_lakes" "$file_forcing"
python3 plotting.py "$A" "$e_nr"

echo "Simulating $e_nr ensemble members for a region of $A m^2 and $T years."