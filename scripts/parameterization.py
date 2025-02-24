#%%
'''
FILENAME:
    parameterization.py

DESCRIPTION:
    This is a script to extract parameter from observational lake area data for
    simulations with the Thermokarst Lake Model (TLM).  

AUTHOR:
    Constanze Reinken

Copyright (C):
    2025 Max-Planck Institute for Meteorology, Hamburg

LICENSE:
    Redistribution and use in source and binary forms, with or without modification, 
    are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, 
        this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice, 
        this list of conditions and the following disclaimer in the documentation 
        and/or other materials provided with the distribution.

        3. Neither the name of the copyright holder nor the names of its contributors 
        may be used to endorse or promote products derived from this software without 
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
    IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
    POSSIBILITY OF SUCH DAMAGE.

'''
#%% import packages

import os
import sys
import inspect
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

#%% directories

# set working directory
os.chdir(os.path.join( os.path.dirname( __file__ ), '..' ))

#%% temp params

A = 40*40*1e6
lake_file = 'input/synthetic_lake_data.nc'
climate_data = 'forcing/synthetic_tdd.txt'
subset_file = None
drainage_file = None

# #%% Area of study region

# # Check if the correct number of arguments is provided (5 arguments including
# # the script name)
# if len(sys.argv) != 6:
#     print(f"Error: Expected 5 arguments, but got {len(sys.argv) - 1}.",
#           file=sys.stderr)
#     sys.exit(1)  # Exit the script if the number of arguments is incorrect

# # Assign each command-line argument to a variable, converting to the
# # appropriate type
# try:
#     A = float(sys.argv[1])
#     lake_file = sys.argv[2]
#     climate_data = sys.argv[3]
#     subset_file = sys.argv[4]
#     drainage_file = sys.argv[5]
# except ValueError as e:
#     print(f"Error: {e}", file=sys.stderr)
#     sys.exit(1)

#%% import timeseries dataset

lakes = xr.open_dataset(lake_file)
lakes_df = lakes.to_dataframe()

# Remove duplicates
lakes_df = lakes_df.loc[~lakes_df.index.duplicated(keep='first')]

# Convert the DataFrame back to a Dataset
lakes = lakes_df.to_xarray()
years = lakes.date.values

#%% OPTIONAL: import IDs for subset of the dataset via shapefile

# import shapefile for area of interest, with lakes as polygons
if subset_file:
    subset = gpd.read_file(subset_file)

    ID_list = list(subset['id_geohash'])

    ID_list_cleaned = []
    for i in ID_list:
        if any(ids == i for ids in lakes.id_geohash.values):
            ID_list_cleaned.append(i)

    lakes = lakes.sel(id_geohash = ID_list_cleaned, date = years)

#%% OPTIONAL: convert ha to m^2

lakes['area_water_permanent'] *= 10000
lakes['area_water_seasonal'] *= 10000
lakes['area_land'] *= 10000
lakes['area_nodata'] *= 10000


#%% clean dataset and extract parameter estimates

f_rate = np.zeros(len(years))
d_rate = np.zeros(len(years))
lake_frac = np.zeros(len(years))
drained_frac = np.zeros(len(years))

log_returns = np.full((len(years), len(lakes.id_geohash)), np.nan)
mu = np.zeros(len(years))
mu[:] = np.nan
sigma = np.zeros(len(years))
sigma[:] = np.nan

formation_events = []

for t in range(1,len(years)):

    excluded_ids = set()

    for i in lakes.id_geohash.values:

        # if more than 10% of the data is missing, set to nan
        polygon_area = lakes.area_water_permanent.sel(
            id_geohash=i,date=years[t]) + lakes.area_water_seasonal.sel(
                id_geohash=i,date=years[t]) + lakes.area_nodata.sel(
                    id_geohash=i,date=years[t]) + lakes.area_land.sel(
                        id_geohash=i,date=years[t])
        if lakes.sel(date=years[t-1],id_geohash = i).area_nodata > 0:
            lakes['area_water_permanent'][lakes.id_geohash.values.tolist().
                                          index(i), t-1] = np.nan
            lakes['area_water_seasonal'][lakes.id_geohash.values.tolist().
                                         index(i), t-1] = np.nan
            lakes['area_land'][lakes.id_geohash.values.tolist().index(i),
                               t-1] = np.nan

        # extract formation and drainage rate
        if all(x == 0 for x in lakes.sel(id_geohash=i).area_water_permanent.
               values[:t]) and (lakes.area_water_permanent.sel(
                   id_geohash=i,date=years[t]) > 0) :
            f_rate[t] += 1
            excluded_ids.add(i)
        if t != len(years) and all(x == 0 for x in lakes.sel(id_geohash=i).
                                   area_water_permanent.values[t:]) and \
                                    (lakes.area_water_permanent.sel(
                                        id_geohash=i,date=years[t-1]) > \
                                            polygon_area * 0.5):
            d_rate[t] += 1
            excluded_ids.add(i)

        if i not in excluded_ids and lakes.area_water_permanent.sel(
            id_geohash=i,date=years[t-1]).values > 0:
            log_returns[t, lakes.id_geohash.values.tolist().index(i)] = \
                np.log(lakes.area_water_permanent.sel(
                    id_geohash=i,date=years[t]).values /
                    lakes.area_water_permanent.sel(id_geohash=i,
                                                   date=years[t-1]).values)

    lake_frac[t] = np.nansum(lakes.area_water_permanent.sel(date=years[t]).
                             values) / A
    drained_frac[t] = (np.nansum(lakes.area_land.sel(date=years[t-1]).values) /
                       A) + max(np.nansum((lakes.area_land.sel(
                           id_geohash=i,date=years[t]).values - lakes.area_land.
                           sel(id_geohash=i,date=years[t-1]).values)) / A ,0)
    log_returns[t] = np.nan_to_num(log_returns[t], nan=np.nan, posinf=np.nan,
                                   neginf=np.nan)
    mu[t] = np.nanmean(log_returns[t]) + (0.5*np.nanstd(log_returns[t]) **2)
    sigma[t] = np.nanstd(log_returns[t])

lake_frac[0] = np.nansum(lakes.area_water_permanent.sel(date=years[0]).
                         values) / A

# scale f_ and d_rate rates by lake and disturbed area
d_rate_sLake = d_rate[1:] / (A * lake_frac[:-1])
d_rate_sDist = d_rate[1:] / (A*(drained_frac[:-1] + lake_frac[:-1]))

f_rate_sLake = f_rate[1:] / (A-(A*lake_frac[:-1]))
f_rate_sDist = f_rate[1:] / (A-(A*(drained_frac[:-1] + lake_frac[:-1])))

# Add np.nan as the first entry to f_rate_sDist
f_rate_sDist = np.insert(f_rate_sDist, 0, np.nan)
f_rate_sLake = np.insert(f_rate_sLake, 0, np.nan)
d_rate_sDist = np.insert(d_rate_sDist, 0, np.nan)
d_rate_sLake = np.insert(d_rate_sLake, 0, np.nan)

#%% OPTIONAL: import drainage event data (Chen et al 2023)

if drainage_file:
    drainage_events = gpd.read_file(drainage_file)
    drainage_events = drainage_events.where(drainage_events.Drain_pct > 0.9)

    time_period = len(drainage_events.DrainYear.unique())

    drain_timeseries = np.zeros(len(years))
    for i in range(0, time_period):
        drain_timeseries[i] = len(drainage_events.loc[drainage_events.
                                                      DrainYear == i])

    # scale d_rate by lake and disturbed area

    d_rate_sDist = drain_timeseries[1:] / (A*(drained_frac[:-1] +
                                              lake_frac[:-1]))
    d_rate_sLake = drain_timeseries[1:] / (A * lake_frac[:-1])


#%% save parameter timeseries in txt files

np.savetxt('parameter/mu.txt', mu)
np.savetxt('parameter/sigma.txt', sigma)
np.savetxt('parameter/f_rate_sDist.txt', f_rate_sDist)
np.savetxt('parameter/d_rate_sDist.txt', d_rate_sDist)
np.savetxt('parameter/f_rate_sLake.txt', f_rate_sLake)
np.savetxt('parameter/d_rate_sLake.txt', d_rate_sLake)

#%% import climate data

climvar = np.loadtxt(climate_data)

#%% create nan masks for mu and sigma

# mu
mask = ~np.isnan(mu) & ~np.isinf(mu)
# sigma
mask = ~np.isnan(sigma)


#%% Define functions to fit

def linear(x, a,b):
    return a * x + b

def exponential(x, a,b):
    return a * np.exp(b * x)

def logarithmic(x, a,b):
    return a * np.log(x) + b

functions = [exponential, logarithmic, linear]


#%% fit different functions

best_fit = {}
params = {'f_rate_sDist': f_rate_sDist, 'f_rate_sLake': f_rate_sLake,
          'd_rate_sDist': d_rate_sDist, 'd_rate_sLake': d_rate_sLake, 
          "mu": mu, "sigma": sigma}

for param_name, param in params.items():
    best_func = None
    best_popt = None
    best_r2 = -np.inf

    for func in functions:
        # Create a mask for non-nan values
        mask = ~np.isnan(param) & ~np.isinf(param)
        # Perform the regression
        popt, pcov = curve_fit(func, climvar[mask][:-1], param[mask][1:])
        # Calculate the y values of the fitted function
        y_fit = func(climvar[mask][:-1], *popt)
        # Calculate the R-squared value
        r2 = r2_score(param[mask][1:], y_fit)

        non_zero_mask = mask & (param != 0)
        if r2 < 0.5 or np.sum(non_zero_mask) < 3:
            continue

        elif r2 > best_r2:
            best_r2 = r2
            best_func = func
            best_popt = popt

    best_fit[param_name] = [best_func, best_popt, best_r2]


#%% save parameter and function in file

with open('parameter/clim_param_func.py', 'w', encoding="utf-8") as f:

    for param_name, param in params.items():

        # Extract the body of the first function, skipping the 'def' line and
        # the 'return' statement
        if best_fit[param_name][0] is not None:
            func1_body_lines = inspect.getsource(
                best_fit[param_name][0]).split('\n')[1:]
             # Remove 'return ' from the last line
            func1_return_expression = func1_body_lines[0][11:]
            # Replace the parameter names with the actual values
            func1_return_expression = func1_return_expression.replace(
                'a', (str(best_fit[param_name][1][0]))) 
            func1_return_expression = func1_return_expression.replace(
                'b', (str(best_fit[param_name][1][1])))
            if len(best_fit[param_name][1]) > 2:
                func1_return_expression = func1_return_expression.replace(
                    'c', (str(best_fit[param_name][1][2])))
            func1_return_expression = func1_return_expression.replace('x',
                                                                      'clim')
            COMBINED_RETURN_EXPRESSION = 'return ' + func1_return_expression
        # if no fitting function was found, use a constant the parameter
        else:

            log_returns_df = pd.DataFrame(log_returns)
            WINDOW_SIZE = 3  # Define the rolling window size
            rolling_mu_list = []
            rolling_sigma_list = []
            for column in log_returns_df:
                rolling_mu = log_returns_df[column].rolling(
                    window=WINDOW_SIZE).mean()
                rolling_sigma = log_returns_df[column].rolling(
                    window=WINDOW_SIZE).std()
                rolling_mu_list.append(rolling_mu)
                rolling_sigma_list.append(rolling_sigma)

            # Step 2: Aggregate rolling mu and sigma
            all_rolling_mu = pd.concat(rolling_mu_list)
            all_rolling_sigma = pd.concat(rolling_sigma_list)

            # Step 3: Calculate mean of aggregated rolling mu and sigma
            constant_sigma = all_rolling_sigma.mean()
            constant_mu = all_rolling_mu.mean() + (0.5*constant_sigma **2)

            VALUE = None
            if param_name == 'mu':
                mask = ~np.isnan(param) & ~np.isinf(param)
                VALUE = constant_mu
            elif param_name == 'sigma':
                VALUE = constant_sigma
            elif param_name == 'f_rate_sDist':
                VALUE = np.nansum(f_rate_sDist) / len(years)
            elif param_name == 'f_rate_sLake':
                VALUE = np.nansum(f_rate_sLake) / len(years)
            elif param_name == 'd_rate_sDist':
                VALUE = np.nansum(d_rate_sDist) / len(years)
            elif param_name == 'd_rate_sLake':
                VALUE = np.nansum(d_rate_sLake) / len(years)
            COMBINED_RETURN_EXPRESSION = 'return ' + str(VALUE)

        # Write the combined function definition to the file
        f.write(f'def func_{param_name}(clim):\n')
        # Write the combined return expression
        f.write(f'    {COMBINED_RETURN_EXPRESSION}\n\n')

f.close()

#%%

print('Parameterization completed.')

# %%
