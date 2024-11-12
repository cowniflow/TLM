#!/bin/bash

# Size of simulated region
A_cell= (40x40 * 1e6) # m2

# lake area data
lake_file = "input/UTM_54_cleaned.nc"

# cimate data
climate_data = "focing/tdd_forcing.txt"

# subset of IDs (OPTIONAL. If left empty, all IDs from lake_file will be used)
subset_file = "input/UTM54_North_40x40_m.shp"

# drainage event file in the style of Chen et al 2023 (OPTIONAL. If left empty, drainage rate will be calculated from lake area data)
drainage_file = "input/Drainage_events_UTM54_North40x40.shp"

# Run scripts to simulate and plot
python3 parameterization.py "$A_cell" "$lake_file" "$climate_data" "$subset_file" "$drainage_file"

echo "Obtaining parameter from $lake_file"