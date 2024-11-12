# Thermokarst Lake Model (TLM)

This code was written by Constanze Reinken.

## Scripts


###  parameterization.py

This script calculates a timeseries of stochastic parameter (drift, volatility, formation rate, abrupt draianage rate) from the parameterization dataset. It finds the best fitting regression function between the parameter and the climate variables (thaw degree days, precipitation) out of a collection of common functions and outputs the function and fitted parameter values in file clim_param_func.py.

### clim_param_func.py

This script contains functions and parameter describing the relationship between stochastic parameter (drift, volatility, formation rate, abrupt draianage rate) and cliamte variables (thaw degree days, precipitation). Obtained via parameterization.py.

### model.py

This script contains the model code and simulates changes in thermokarst lake distributions using the parameterization contained in clim_param_func.py. The number of ensemble runs can be defined when executing the script. The script creates netcdf files for each ensemble run containing timeseries of the individual lakes and timeseries of drained and lake area fraction. For the latter, the model also outputs the ensemble mean. It can also be used to create a dataset that can be perturbed and turned into a synthetic parameterization dataset via synth_data.py.

### plotting.py

This script creates timeseries plot of lake and drained area fractions; as well as a gif of one ensemble run. Which one can be defined when executing the script.

## Folder structure

### input

Observational / remote sensing data on lake areas can be stored here and used directly for parameterization.py and as an initialization dataset in model.py. 

### parameter

The .py files that can be created by parameterization.py are stored here, as well as txt files of the calculate parameter timeseries. 

### forcing

Files of climate variables are stored here as txt files.

### output

The output from tlm.py and plotting.py is stored here. These include:
- lakes.nc: A netcdf file containing permanent water and land (i.e. drained) area, ages and type of the lake and DLB objects, as well as their id and coordinates at every time step.
- area_water_frac.txt: A timeseries of the water area fraction.
- area_drained_frac.tx: A timeseries of the drained area fraction.
- lake_nr.txt: A timeseries of the number of lakes.
- area_water_frac.png: A plot of the water area fraction of all ensemble runs over time.
- area_frained_frac.png: A plot of the drained area fraction of all ensemble runs over time.
- number.png: A plot of the lake number of all ensemble runs over time.
- circles: A folder with all .png files created for each ensemble.
- run_{}.gif: A gif for each ensemble run, created from .png files from circles.

## Documentation
Some documentation has been set up for this project which you should be able to find hosted online
here:
### https://



## Contributors
- Constanze Reinken

## Acknowledgements
- Victor Brovkin
- Ingmar Nitze
- Philipp deVrese
