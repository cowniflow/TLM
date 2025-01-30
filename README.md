# Thermokarst Lake Model (TLM)

This code was written by Constanze Reinken.

## Scripts


###  parameterization.py

This script calculates a timeseries of stochastic parameter (drift, volatility, formation rate, abrupt drainage rate) from the parameterization dataset. It finds the best fitting regression function between the parameter and a climate variable (e.g thaw degree days) out of a collection of common functions. It outputs the function and fitted parameter values in file clim_param_func.py.


### model.py

This script contains the model code and simulates changes in thermokarst lake distributions using the parameterization contained in clim_param_func.py. The number of ensemble runs can be defined when executing the script. The script creates netcdf files for each ensemble run containing timeseries of the individual lakes and timeseries of drained and lake area fraction. For the latter, the model also outputs the ensemble mean. 

### plotting.py

This script creates timeseries plot of lake and drained area fractions for the ensemble, including the ensemble mean and standard deviation. 

### animations.py

This script creates a folder with spatial plots (as .png files) of the lakes for each ensemble run and combines them into a gif and an mp4 file. 

## Running the scripts

The python scripts can be executed using shell scripts that contain timestep, variant,  paths to initialization, forcing and parameter files, and other parameter. If the python scripts shall be executed from an IDE, the paths / parameter from the shell scripts need to be put into the code directly. 

### run_parametarization.sh

Shell script to execute parameterization.py. Needs following input: 
- A: size of the study region
- lake_file: path and file name for the data file with lake areas (.nc)
- climate_data: path and file name for the file with data for one climate varialbe (.txt)
- subset_file (OPTIONAL): path and file name for file with lake IDs for an area within lake_file area (.shp)
- drainage_file (OPTIONAL): path and file name for file with drainage events in form of Chen et al 2023 (.shp)

### run_tlm.sh

Shell script to execute model.py and plotting.py. Needs following input:
- variant: variant of the model; Options: 1, 2 
- A: size of the simulated region
- frac_lim: maximum possible water / depression fraction limit
- T: time span of simulation in years
- dt: time step of simulation in years
- e_nr: number of ensemble runs
- par_script: path and file name for the python script with parameter functions (e.g. clim_param_func.py)
- file_ini_lakes (OPTIONAL): path and file name for file with initialization data (.nc); if left empty, the model will start with no lakes
- subset_lakes (OPTIONAL): path and file name for file with IDs for an area within file_ini_lakes (.shp); if left empty, all lakes from file_ini_lakes will be used
- file_forcing: path and file name with forcing data, i.e. one climate variable 

### create_animations.sh

Shell script to execute animations.py. Needs following input:
- A: size of the simulated region
- dt: time step of simulation in years
- e_nr: number of ensemble runs

## Folder structure

### input

Observational / remote sensing data on lake areas can be stored here and used directly for parameterization.py and as an initialization dataset in model.py. The data needs to be a netcdf file. If a subset from the netcdf file shall be extracted, this can be done using a shapefile containing the corresponding object names or ids. 

### parameter

The folder stores txt files of parameter timeseries as well as the file clim_param_func.py, that can be created via parameterization.py. The python script contains functions and parameter describing the relationship between stochastic parameter (drift, volatility, formation rate, abrupt draianage rate) and a cliamte variable (e.g. thaw degree days). It can be imported into model.py as a module. 
 

### forcing

Files of climate variables are stored here as txt files.

### output

The output from tlm.py are stored here. These include:
- lakes.nc: A netcdf file containing permanent water and land (i.e. drained) area, ages and type of the lake and DLB objects, as well as their id and coordinates at every time step.
- area_water_frac.txt: A timeseries of the water area fraction.
- area_drained_frac.tx: A timeseries of the drained area fraction.
- lake_nr.txt: A timeseries of the number of lakes.

### plots
All plots that were created with plotting.py or animations.py are stored here. These include:
- ensemble_plot.png: Timeseries of water area and drained fraction of the ensemble, including ensemble mean and standard deviation. 
- circles: A folder with all .png files created for each ensemble.
- run_{}.gif: A gif for each ensemble run, created from .png files from circles.
- run_{}.mp4: A video file for each ensemble run, created from .png files from circles

##  Zenodo

The code is published via Zenodo under doi: 

## Paper
The accompanying scientific paper is in preparation. 


## Contributors
- Constanze Reinken

## Acknowledgements
- Victor Brovkin
- Ingmar Nitze
- Philipp deVrese
- Helena Bergstedt
- Tobias Stacke
