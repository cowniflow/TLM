# Thermokarst Lake Model (TLM)

This code was written by Constanze Reinken.

## Installation

TLM requires a Python 3.x environment with several packages. The most convenient way would be to set up an Anaconda environment using environment.yml.

```
conda env create -f environment.yml
```
## Overview

The Thermokarst Lake Model (TLM) is designed to simulate changes in thermokarst lake distributions in thermokarst-affected regions. 

The model code is written in Python and can be executed using shell scripts that contain the necessary parameter values. Besides the model code that performs the simulations (model.py), there are also scripts to extract parameter values from observational data (parameterization.py), plot timeseries of simulated lake number as well as water and drained area (plotting.py), and create animations of a spatial representation of lake distribution (animations.py). 

The model represents lakes as circular objects and uses stochastic processes to represent the main processes behind thermokarst lake dynamics: 

* Formation: A Poisson process supplies the number of new lakes for each timestep (usually a year). More specifically, the number of new lakes is drawn at each timestep from a Poisson distribution with parameter $\lambda_f$, which is the formation rate. Each new lake gets two random centre coordinates. The probability of $k$ thermokarst depressions appearing during one year in $A_f$, which is the area available for formation, is

$$
P_f (k,A_f) = \frac{(\lambda_f A_f)^{k}}{k!} e^{- \lambda_f A_f} .
$$
    
* Expansion: The surface area of individual lakes change according to Geometric Brownian Motion. All lakes across a simulated area follow Geometric Brownian Motion with the same parameter drift $\mu$ and volatility $\sigma$, where $\sigma$ represents the random component and variety of lake behaviour. The individual lake areas $a_i(t)$ at time $t$ are calculated with the following euqation, where $B(t)$ is regular Brownian Motion: 

$$
a_i(t) = a_i(t-1)e^{(\mu - \frac{1}{2}\sigma^2)t+\sigma B(t)} 
$$
    
* Gradual Drainage: Geometric Brownian Motion can also lead to decreasing lake areas, either due to $\sigma$ or due to a negative drift value $\mu$. 

* Abrupt Drainage: Abrupt drainage is represented using another Poisson process. A number of abruptly draining lakes is drawn from a Poisson process at each time step, where the parameter $\lambda_d$ is the abrupt drainage rate. The abruptly draining lakes are then randomly selected from the active lakes and reach a surface of zero within one timestep. The probability of $k$ drainage events during one year in $A_d$, which is the disturbed or water area (see 'Model Variants') within the system, is 

$$
P_d (k,A_d) = \frac{(\lambda_d A_d)^{k}}{k!} e^{- \lambda_d A_d} .
$$

Lakes merge as soon as they start to overlap. A merging algorithm checks for overlapping lakes at each timestep and in case of overlap transfers the surface area of the smaller one to the larger one. It also determines the new centre coordinates of the resulting bigger lake by calculating the new centre of mass. 

### Model Variants

There are two model variants. Which variant is used, can be determined in run_tlm.sh. 

* In Variant 1, both the sum of water area $A_{water}$ and the sum of drained area $A_{drained}$ constitute $A_d$ and are subtracted from the area of the simulated region to obtain $A_f$.

* In Variant 2, only the water area $A_{water}$ is considered for the scaling of formation and drainage probability. 


## Scripts

###  parameterization.py

This script calculates a timeseries of the stochastic parameter (drift, volatility, formation rate, abrupt drainage rate) from the parameterization dataset. It finds the best fitting regression function between the parameter and a climate variable (e.g thaw degree days) out of a collection of common functions. It outputs the function and fitted parameter values in file clim_param_func.py. The parameter estimates represents how lake area or number have changed from the previous year to the current year. As the default, this value is compared to the climate variable in the previous year. This can be changed in the script by indexing the climate variable differently. Make sure that the datasets span the necessary years. 


### model.py

This script contains the model code and simulates changes in thermokarst lake distributions using the parameterization contained in clim_param_func.py. The number of ensemble runs can be defined when executing the script. The script creates netcdf files for each ensemble run containing timeseries of the individual lakes and timeseries of drained and lake area fraction. For the latter, the model also outputs the ensemble mean. 

### plotting.py

This script creates timeseries plots of lake and drained area fractions for the ensemble, including the ensemble mean and standard deviation. 

### animations.py

This script creates a folder with spatial plots (as .png files) of the lakes for each ensemble run and combines them into a gif and an mp4 file. 

## Prerequisites 
 
All scripts require the packages *numpy* and *os* to be installed. Additionally, each script needs the packages as listed in the table below. All requirements (packages and dependencies) are contained in environment.yml. 


|                     | model.py   | parameterization.py | plotting.py | animations.py |
|---------------------|------------|---------------------|-------------|---------------|
| *numpy*      | :heavy_check_mark: | :heavy_check_mark:         | :heavy_check_mark:  | :heavy_check_mark:    |
| *os*         | :heavy_check_mark: | :heavy_check_mark:          | :heavy_check_mark:  | :heavy_check_mark:    |
| *geopandas*  | :heavy_check_mark: | :heavy_check_mark:          |             |               |
| *imageio*    |            |                     |             | :heavy_check_mark:    |
| *math*       | :heavy_check_mark: |                     |             |               |
| *matplotlib* |            |                     | :heavy_check_mark:  | :heavy_check_mark:    |
| *moviepy*    |            |                     |             | :heavy_check_mark:    |
| *netCDF4*    |            | :heavy_check_mark:          |             | :heavy_check_mark:    |
| *pandas*     |            | :heavy_check_mark:          |             |               |
| *scipy*      |            | :heavy_check_mark:          |             |               |
| *sklearn*    |            | :heavy_check_mark:          |             |               |
| *sys*        | :heavy_check_mark: |                     |             |               |
| *tqdm*       | :heavy_check_mark: |                     |             | :heavy_check_mark:    |
| *xarray*     | :heavy_check_mark: | :heavy_check_mark:          |             |               |

## Running the scripts

The python scripts can be executed using shell scripts that contain timestep, variant, paths to initialization data, forcing and parameter files, and other parameter. If the python scripts shall be executed from an IDE, the paths / parameter from the shell scripts need to be put into the code directly. 

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
- e_nr: number of ensemble runs
- T: time span
- dt: time step of simulation in years
- folder_name: path to the lake data to be plotted

## Folder structure

### input

Observational / remote sensing data on lake areas can be stored here and used directly for parameterization.py and as an initialization dataset in model.py. The data needs to be in form of a netcdf file. If a subset from the netcdf file shall be extracted, this can be done using a shapefile containing the corresponding object names or ids. 

### parameter

The folder stores txt files of parameter timeseries as well as the file clim_param_func.py, that can be created via parameterization.py. The python script contains functions and parameter describing the relationship between stochastic parameter (drift, volatility, formation rate, abrupt draianage rate) and a climate variable (e.g. thaw degree days). It can be imported into model.py as a module. 
 
### forcing

Files of climate variables are stored here as txt files.

### output

The output from tlm.py is stored here. This includes:
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

## Test Case Data

A synthetic lake dataset and forcing data were created, which the model can be tested with. The repository also contains the corresponsing parameter, model output and plots. 

## TODO Zenodo

## Paper

The accompanying scientific paper is in preparation. 

## Contributors
- Constanze Reinken (constanze.reinken@mpimet.mpg.de)

## Acknowledgements
This work was supported by the European Research Council project Q-Arctic (grant no. 951288). It used resources of the Deutsches Klimarechenzentrum (DKRZ) granted by its Scientific Steering Committee (WLA) under project ID bm1236. Special thanks goes to Victor Brovkin, Philipp deVrese, Ingmar Nitze and Helena Bergstedt for providing data and scientific expertise that contributed to the model development, as well as to Tobias Stacke for his input on code structure and style. 
