#!/bin/bash

# Size of simulated region
A_cell= 40x40 * 1e6 # m2

# climate data
climate_data = "forcing/tdd_frocing.txt"

# lake data
lake_file = "input/UTM54_cleaned.nc"

# subset ID file (OPTIONAL: leave blank and use all lakes from lake_file)
subset_file = "input/UTM54_North_ini_40x40.shp"

# drainage event data in style of Chen et al 2023 (OPTIONAL: leave blank and only use lake change data)
drainage_file = "inputDrainage_events_UTM54_North_40x40.shp"

# Run scripts to simulate and plot
python3 paramterization.py "$A_cell" "$lake_file" "$climate_data" "$subset_file" "$drainage_file"

echo "Obtaining parameterization data for "