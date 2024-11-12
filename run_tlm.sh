#!/bin/bash

# Size of simulated region
A_cell=10000e6 # m2

# ground ice content
gr_ice = 0.3

# Time span and time step
T=119 # years
dt=1

# Nr. of ensemble runs
e_nr=10

# Parameterization script
par_script="clim_param_func"

# Initial lake data
file_ini_lakes = ""



# Run scripts to simulate and plot
python3 tlm.py "$A_cell" "$gr_ice" "$T" "$dt" "$par_script" "$e_nr" "$file_ini_lakes" 
python3 plotting.py "$A_cell" "$e_nr"

echo "Simulating $e_nr ensemble members for region of $A_cell m^2 for $T years."