# Parameterization using remote sensing or synthetic data
# by Constanze Reinken 2024


#%% import packages

import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os


#%% directories

path = '/Users/constanzereinken/Data/drive-download-20240119T112608Z-001/'

#%% Area of study region

# artificial dataset
#A = 48623e6  #m2
A = 40*40 * 1e6

#%% import timeseries dataset

#file = 'lakes_1.nc'
file = 'UTM54_cleaned.nc'
lakes = xr.open_dataset(path+file)


lakes_df = lakes.to_dataframe()

# Remove duplicates
lakes_df = lakes_df.loc[~lakes_df.index.duplicated(keep='first')]

# Convert the DataFrame back to a Dataset
lakes = lakes_df.to_xarray()

years = lakes.date.values
years = years[16:]

# #%% import IDs

# mu_mean = []
# sigma_mean = []

# # import shapefile for area of interest, with lakes as polygons
# #path_grid = '/Users/constanzereinken/Data/Gridded_UTM54/Grid1/Lake_change/'
# path_grid = '/Users/constanzereinken/Data/drive-download-20240119T112608Z-001'
# for file in os.listdir(path_grid):
#     if file.endswith('.shp') and file == 'UTM54_North_40x40.shp':
#         region = gpd.read_file(path_grid + file)
#         if len(region['id_geohash'].values) == 0:
#             continue
#         ID_list = list(region['id_geohash'])
#         ID_list_cleaned = []

#         for i in ID_list:
#             if any(ids == i for ids in lakes.id_geohash.values):
#                 ID_list_cleaned.append(i)

#         lakes_subset = lakes.sel(id_geohash = ID_list_cleaned, date = years)

#         lakes_subset['area_water_permanent'] *= 10000
#         lakes_subset['area_water_seasonal'] *= 10000
#         lakes_subset['area_land'] *= 10000
#         lakes_subset['area_nodata'] *= 10000


#         # clean and compute f_ and d_rate

#         f_rate = np.zeros(len(years))

#         d_rate_proto = np.zeros(len(years))

#         lake_frac = np.zeros(len(years))

#         drained_frac = np.zeros(len(years))

#         log_returns = np.full((len(years), len(lakes.id_geohash)), np.nan)
#         mu = np.zeros(len(years))
#         mu[:] = np.nan
#         sigma = np.zeros(len(years))
#         sigma[:] = np.nan

#         formation_events = []

#         for t in range(1,len(years)):

#             excluded_ids = set()

#             for i in lakes_subset.id_geohash.values:

#                 # if more than 10% of the data is missing, set to nan
#                 polygon_area = lakes.area_water_permanent.sel(id_geohash=i,date=years[t]) + lakes.area_water_seasonal.sel(id_geohash=i,date=years[t]) + lakes.area_nodata.sel(id_geohash=i,date=years[t]) + lakes.area_land.sel(id_geohash=i,date=years[t])
#                 if lakes.sel(date=years[t-1],id_geohash = i).area_nodata > 0:
#                     lakes['area_water_permanent'][lakes.id_geohash.values.tolist().index(i),t-1] = np.nan
#                     lakes['area_water_seasonal'][lakes.id_geohash.values.tolist().index(i),t-1] = np.nan
#                     lakes['area_land'][lakes.id_geohash.values.tolist().index(i),t-1] = np.nan

#                 # extract formation and drainage rate
#                 if all(x == 0 for x in lakes.sel(id_geohash=i).area_water_permanent.values[:t]) and (lakes.area_water_permanent.sel(id_geohash=i,date=years[t]) > 0) :
#                     f_rate[t] += 1
#                     excluded_ids.add(i)
#                 if t != len(years) and all(x == 0 for x in lakes.sel(id_geohash=i).area_water_permanent.values[t:]) and (lakes.area_water_permanent.sel(id_geohash=i,date=years[t-1]) > polygon_area * 0.5):
#                     d_rate_proto[t] += 1
#                     excluded_ids.add(i)

#                 if i not in excluded_ids and lakes.area_water_permanent.sel(id_geohash=i,date=years[t-1]).values > 0:
#                     log_returns[t, lakes.id_geohash.values.tolist().index(i)] = np.log(lakes.area_water_permanent.sel(id_geohash=i,date=years[t]).values / lakes.area_water_permanent.sel(id_geohash=i,date=years[t-1]).values)

#             lake_frac[t] = np.nansum(lakes.area_water_permanent.sel(date=years[t]).values) / A
#             drained_frac[t] = (np.nansum(lakes.area_land.sel(date=years[t-1]).values) / A) + max(np.nansum((lakes.area_land.sel(id_geohash=i,date=years[t]).values - lakes.area_land.sel(id_geohash=i,date=years[t-1]).values)) / A ,0)
#             log_returns[t] = np.nan_to_num(log_returns[t], nan=np.nan, posinf=np.nan, neginf=np.nan)
#             mu[t] = np.nanmean(log_returns[t]) + (np.nanstd(log_returns[t]) **2 / 2)
#             sigma[t] = np.nanstd(log_returns[t])

#         lake_frac[0] = np.nansum(lakes.area_water_permanent.sel(date=years[0]).values) / A

#         f_rate_sLake = f_rate[1:] / (A-(A*lake_frac[:-1]))
#         f_rate_sDist = f_rate[1:] / (A-(A*(drained_frac[:-1] + lake_frac[:-1])))

#         mu_mean.append(np.nanmean(mu))
#         sigma_mean.append(np.nanmean(sigma))

# mu_avg = np.nanmean(mu_mean)
# sigma_avg = np.nanmean(sigma_mean)

#%% import IDs

# import shapefile for area of interest, with lakes as polygons
file = path + 'UTM54_North_ini_40x40.shp'
region = gpd.read_file(file)

ID_list = list(region['id_geohash'])

ID_list_cleaned = []

for i in ID_list:
    if any(ids == i for ids in lakes.id_geohash.values):
        ID_list_cleaned.append(i)

lakes = lakes.sel(id_geohash = ID_list_cleaned, date = years)

lakes['area_water_permanent'] *= 10000
lakes['area_water_seasonal'] *= 10000
lakes['area_land'] *= 10000
lakes['area_nodata'] *= 10000


#%% clean and compute f_ and d_rate

f_rate = np.zeros(len(years))

d_rate_proto = np.zeros(len(years))

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
        polygon_area = lakes.area_water_permanent.sel(id_geohash=i,date=years[t]) + lakes.area_water_seasonal.sel(id_geohash=i,date=years[t]) + lakes.area_nodata.sel(id_geohash=i,date=years[t]) + lakes.area_land.sel(id_geohash=i,date=years[t])
        if lakes.sel(date=years[t-1],id_geohash = i).area_nodata > 0:
            lakes['area_water_permanent'][lakes.id_geohash.values.tolist().index(i),t-1] = np.nan
            lakes['area_water_seasonal'][lakes.id_geohash.values.tolist().index(i),t-1] = np.nan
            lakes['area_land'][lakes.id_geohash.values.tolist().index(i),t-1] = np.nan

        # extract formation and drainage rate
        if all(x == 0 for x in lakes.sel(id_geohash=i).area_water_permanent.values[:t]) and (lakes.area_water_permanent.sel(id_geohash=i,date=years[t]) > 0) :
            f_rate[t] += 1
            excluded_ids.add(i)
        if t != len(years) and all(x == 0 for x in lakes.sel(id_geohash=i).area_water_permanent.values[t:]) and (lakes.area_water_permanent.sel(id_geohash=i,date=years[t-1]) > polygon_area * 0.5):
            d_rate_proto[t] += 1
            excluded_ids.add(i)

        if i not in excluded_ids and lakes.area_water_permanent.sel(id_geohash=i,date=years[t-1]).values > 0:
            log_returns[t, lakes.id_geohash.values.tolist().index(i)] = np.log(lakes.area_water_permanent.sel(id_geohash=i,date=years[t]).values / lakes.area_water_permanent.sel(id_geohash=i,date=years[t-1]).values)

    lake_frac[t] = np.nansum(lakes.area_water_permanent.sel(date=years[t]).values) / A
    drained_frac[t] = (np.nansum(lakes.area_land.sel(date=years[t-1]).values) / A) + max(np.nansum((lakes.area_land.sel(id_geohash=i,date=years[t]).values - lakes.area_land.sel(id_geohash=i,date=years[t-1]).values)) / A ,0)
    log_returns[t] = np.nan_to_num(log_returns[t], nan=np.nan, posinf=np.nan, neginf=np.nan)
    mu[t] = np.nanmean(log_returns[t]) + (0.5*np.nanstd(log_returns[t]) **2)
    sigma[t] = np.nanstd(log_returns[t])

lake_frac[0] = np.nansum(lakes.area_water_permanent.sel(date=years[0]).values) / A


#%% import shapefile for area of interest, with drainage events
file = '/Users/constanzereinken/Data/Arctic_lake_drainage_events/Drainage_events_UTM54_North_40x40.shp'
events = gpd.read_file(file)
events = events.where(events.Drain_pct > 0.9)

drain_timeseries = np.zeros(len(years))
for i in range(2000, 2021):
    drain_timeseries[i - 2000] = len(events.loc[events.DrainYear == i])

#%%
d_rate_sDist = drain_timeseries[1:] / (A*(drained_frac[:-1] + lake_frac[:-1]))
d_rate_sLake = drain_timeseries[1:] / (A * lake_frac[:-1])

#%% scale rates by lake and disturbed area
d_rate_sLake_proto = d_rate_proto[1:] / (A * lake_frac[:-1])
d_rate_sDist_proto = d_rate_proto[1:] / (A*(drained_frac[:-1] + lake_frac[:-1]))

f_rate_sLake = f_rate[1:] / (A-(A*lake_frac[:-1]))
f_rate_sDist = f_rate[1:] / (A-(A*(drained_frac[:-1] + lake_frac[:-1])))

#%% save parameters

np.savetxt('data/UTM54_North_40x40_params/mu.txt', mu)
np.savetxt('data/UTM54_North_40x40_params/sigma.txt', sigma)
np.savetxt('data/UTM54_North_40x40_params/f_rate_sDist.txt', f_rate_sDist)
np.savetxt('data/UTM54_North_40x40_params/d_rate_sDist.txt', d_rate_sDist)
np.savetxt('data/UTM54_North_40x40_params/f_rate_sLake.txt', f_rate_sLake)
np.savetxt('data/UTM54_North_40x40_params/d_rate_sLake.txt', d_rate_sLake)

#%% get climate data

# precipitation
precip = np.loadtxt('data/UTM54_North/precip_forcing.txt')[15:-1]
# thaw degree days
tdd = np.loadtxt('data/UTM54_North/tdd_forcing.txt')[16:]
tdd_running_mean = np.convolve(tdd[1:], np.ones(3)/3, mode='valid')

tdd = precip

#%% Define the different types of functions
def linear(x, a,b):
    return a * x + b

def quadratic(x, a,b,c):
    return a * x**2 + b * x + c

def exponential(x, a,b):
    return a * np.exp(b * x)

def logarithmic(x, a,b):
    return a * np.log(x) + b

# List of functions to try
functions = [exponential, logarithmic, linear]

#%% kernel smoothing for f_ and d_rate

from statsmodels.nonparametric.kernel_regression import KernelReg

# f_rate
f_rate_running_mean = np.convolve(f_rate_sDist[:-2], np.ones(3)/3, mode='valid')

# Plot the results
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.scatter(tdd_running_mean, f_rate_running_mean, label='Original Data', color='blue')
ax.set_xlabel('Thaw degree days')
ax.set_ylabel('f_rate')
ax.set_title('Kernel Smoothing of f_rate vs. Climate Forcing')
plt.legend()
plt.show()

# d_rate

d_rate_running_mean = np.convolve(d_rate_sDist[:-2], np.ones(3)/3, mode='valid')

# Plot the results
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.scatter(tdd_running_mean, d_rate_running_mean, label='Original Data', color='blue')
ax.set_xlabel('Thaw degree days')
ax.set_ylabel('d_rate')
ax.set_title('Kernel Smoothing of d_rate vs. Climate Forcing')
plt.legend()
plt.show()

# mu

mask = ~np.isnan(mu[1:-2]) & ~np.isinf(mu[1:-2])

# Plot the results
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.scatter(tdd[1:][mask], mu[1:-2][mask], label='Original Data', color='blue')
ax.set_xlabel('Thaw degree days')
ax.set_ylabel('mu')
ax.set_title('Kernel Smoothing of mu vs. Climate Forcing')
plt.legend()
plt.show()

# # sigma

mask = ~np.isnan(sigma[1:-2])

# Plot the results
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.scatter(tdd[1:][mask], sigma[1:-2][mask], label='Original Data', color='blue')
ax.set_xlabel('Thaw degree days')
ax.set_ylabel('sigma')
ax.set_title('Kernel Smoothing of sigma vs. Climate Forcing')
plt.legend()
plt.show()


#%% test different functions

best_fit = {}
#params = {'mu': mu[1:], 'sigma': sigma[1:], 'f_rate_sDist': f_rate_sDist[1:]} #, 'f_rate_sLake': f_rate_sLake[1:], 'd_rate_sDist': d_rate_sDist[1:], 'd_rate_sLake': d_rate_sLake[1:]}
params = {'f_rate_sDist': f_rate_sDist[:-2], 'f_rate_sLake': f_rate_sLake[:-2], 'd_rate_sDist': d_rate_sDist[:-2], 'd_rate_sLake': d_rate_sLake[:-2], "mu": mu[1:-2], "sigma": sigma[1:-2]}

for param_name, param in params.items():
    best_func_tdd = None
    best_popt_tdd = None
    best_r2_tdd = -np.inf

    for func in functions:
        # Create a mask for non-nan values
        mask = ~np.isnan(param) & ~np.isinf(param)
        # Perform the regression
        popt, pcov = curve_fit(func, tdd[1:][mask], param[mask])
        # Calculate the y values of the fitted function
        y_fit = func(tdd[1:][mask], *popt)
        # Calculate the R-squared value
        r2 = r2_score(param[mask], y_fit)

        if r2 < 0.5:
            continue

        elif r2 > best_r2_tdd:
            best_r2_tdd = r2
            best_func_tdd = func
            best_popt_tdd = popt

    best_fit[param_name] = [best_func_tdd, best_popt_tdd, best_r2_tdd]


#%% save parameter and function in file

shp_areas = gpd.read_file(path + 'UTM54_North_ini_40x40.shp')
shp_areas_start = shp_areas['Area_start'] *10000
shp_areas_end = shp_areas['Area_end_h'] *10000

import inspect

with open('clim_param_func_utm54_40x40.py', 'w') as f:

    for param_name, param in params.items():

        # Extract the body of the first function, skipping the 'def' line and the 'return' statement
        if best_fit[param_name][0] is not None:
            func1_body_lines = inspect.getsource(best_fit[param_name][0]).split('\n')[1:]
            func1_return_expression = func1_body_lines[0][11:]  # Remove 'return ' from the last line
            func1_return_expression = func1_return_expression.replace('a', (str(best_fit[param_name][1][0])))  # Replace the parameter names with the actual values
            func1_return_expression = func1_return_expression.replace('b', (str(best_fit[param_name][1][1])))
            if len(best_fit[param_name][1]) > 2:
                func1_return_expression = func1_return_expression.replace('c', (str(best_fit[param_name][1][2])))
            func1_return_expression = func1_return_expression.replace('x', 'tdd')
            combined_return_expression = func1_return_expression
        # if no function was found, use a constant the parameter
        else:

            log_returns_df = pd.DataFrame(log_returns)
            window_size = 3  # Define the rolling window size
            rolling_mu_list = []
            rolling_sigma_list = []
            for column in log_returns_df:
                rolling_mu = log_returns_df[column].rolling(window=window_size).mean()
                rolling_sigma = log_returns_df[column].rolling(window=window_size).std()
                rolling_mu_list.append(rolling_mu)
                rolling_sigma_list.append(rolling_sigma)

            # Step 2: Aggregate rolling mu and sigma
            all_rolling_mu = pd.concat(rolling_mu_list)
            all_rolling_sigma = pd.concat(rolling_sigma_list)

            # Step 3: Calculate mean of aggregated rolling mu and sigma
            constant_sigma = all_rolling_sigma.mean()
            constant_mu = all_rolling_mu.mean() + (0.5*constant_sigma **2)
            #constant_sigma = np.nanstd(np.log(shp_areas_end / shp_areas_start) / np.sqrt(len(years)))
            #constant_mu = np.nanmean(np.log(shp_areas_end / shp_areas_start)) + (0.5*constant_sigma **2)

            if param_name == 'mu':
                mask = ~np.isnan(param) & ~np.isinf(param)
                value = constant_mu
            elif param_name == 'sigma':
                value = constant_sigma
            elif param_name == 'f_rate_sDist':
                value = np.nansum(f_rate_sDist) / len(years)
            elif param_name == 'f_rate_sLake':
                value = np.nansum(f_rate_sLake) / len(years)
            elif param_name == 'd_rate_sDist':
                value = np.nansum(d_rate_sDist) / len(years)
            elif param_name == 'd_rate_sLake':
                value = np.nansum(d_rate_sLake) / len(years)
            combined_return_expression = 'return ' + str(value)

        # Write the combined function definition to the file
        f.write(f'def func_{param_name}(tdd):\n')
        # Write the combined return expression
        f.write(f'    {combined_return_expression}\n\n')

f.close()

# %%
