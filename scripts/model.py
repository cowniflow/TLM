# Thermokarst Lake Model
# by Constanze Reinken 2024

#%% import packages
import sys
import os
import math
import importlib.util
import numpy as np
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

#%% directories

# set working directory
os.chdir(os.path.join( os.path.dirname( __file__ ), '..' ))

#%% define functions

# Geometric Brownian Motion
def geometric_brownian(x0, drift, volatility, timestep):
    x = x0 * np.exp((drift - ((volatility ** 2) / 2)) * timestep
                + volatility * np.random.normal(0, np.sqrt(timestep)))
    return x


# Merging Algorithm
def merge(area, x, y):

    area_new = area.copy()
    x_new = x.copy()
    y_new = y.copy()

    # select all circle objects with an area
    idx_circles = [i for i in range(len(area)) if area[i] != 0 
                   and not math.isnan(area[i])]

    # loop through all circles
    if len(idx_circles) != 0:
        for i in idx_circles:
            for j in idx_circles:

                if i != j and area_new[i] != 0 and area_new[j] != 0:
                    # calculate distance between centers
                    dx = abs(x_new[i] - x_new[j])
                    dy = abs(y_new[i] - y_new[j])
                    dist = (dx**2 + dy**2)**0.5

                    # check for overlapping circles
                    if (dist + 0.01) < np.sqrt(area_new[i]/np.pi) + \
                        np.sqrt(area_new[j]/np.pi):

                        # calculate new centre (centre of mass) and transfer area to the bigger one of the circles
                        if area[j] >= area[i]:
                            x_new[j] = (1 / (area_new[j] + area_new[i])) * \
                                np.sum((area_new[j] * x_new[j], area_new[i] * x_new[i]))
                            y_new[j] = (1 / (area_new[j] + area_new[i])) * \
                                np.sum((area_new[j] * y_new[j], area_new[i] * y_new[i]))
                            area_new[j] += area_new[i]
                            area_new[i] = 0

                        else:
                            x_new[i] = (1 / (area_new[i] + area_new[j])) * \
                                np.sum((area_new[i] * x_new[i], area_new[j] * x_new[j]))
                            y_new[i] = (1 / (area_new[i] + area_new[j])) * \
                                np.sum((area_new[i] * y_new[i], area_new[j] * y_new[j]))
                            area_new[i] += area_new[j]
                            area_new[j] = 0

    return area_new, x_new, y_new


#%% Import the necessary parameter, initialization data and forcing

# Check if the correct number of arguments is provided (11 arguments including the script name)
if len(sys.argv) != 11:
    print(f"Error: Expected 10 arguments, but got {len(sys.argv) - 1}.", file=sys.stderr)
    sys.exit(1)  # Exit the script if the number of arguments is incorrect

# Assign each command-line argument to a variable, converting to the appropriate type
variant = sys.argv[1]  
A = float(sys.argv[2])
frac_lim = float(sys.argv[3])
T = int(sys.argv[4])
dt =  float(sys.argv[5])
e_nr = int(sys.argv[6])
clim_param_func = sys.argv[7]
ini_lakes = sys.argv[8]
subset_lakes = sys.argv[9]
forcing = sys.argv[10]

#%% import parameter values and / or functions

spec = importlib.util.spec_from_file_location("clim_param_func", clim_param_func)
param_module = importlib.util.module_from_spec(spec)
sys.modules["clim_param_func"] = param_module
spec.loader.exec_module(param_module)

if variant == '1':
    f_rate = param_module.func_f_rate_sDist
    d_rate = param_module.func_d_rate_sDist
elif variant == '2':
    f_rate = param_module.func_f_rate_sLake
    d_rate = param_module.func_d_rate_sLake
func_sigma = param_module.func_sigma
func_mu = param_module.func_mu

#%% define spatial domain, temporal domain & time step

n = int(T/dt) # nr of steps
years = np.arange(1900, 1900 + T, dt) # time points

#%% define maximum potential disturbed area

A_lim = A * frac_lim # maximum potential disturbed area in m^2

#%% load climate data and extrapolate to the desired time points

climvar = np.loadtxt(forcing)[:-1]
original_time_points = np.linspace(0, len(climvar) - 1, len(climvar))
new_time_points = np.linspace(0, len(climvar) - 1, n)
climvar = np.interp(new_time_points, original_time_points, climvar)

#%% initialization

max_f = np.max(f_rate(climvar)) # maximum formation rate
# generate max. number of potential new lakes
form_n = math.ceil(A * T * max_f)

if ini_lakes:
    # import / create initialization data (lake sizes, DLB sizes)
    lake_dataset = xr.open_dataset(ini_lakes)
    lakes_df = lake_dataset.to_dataframe()
    # Remove duplicates
    lakes_df = lakes_df.loc[~lakes_df.index.duplicated(keep='first')]
    # Convert the DataFrame back to a Dataset
    lakes_nc = lakes_df.to_xarray()

    # OPTIONAL: get subset of lakes that are within the region
    if subset_lakes:
        region = gpd.read_file(subset_lakes)
        ID_list = list(region['id_geohash'])
        ID_list_cleaned = []
        for i in ID_list:
            if any(ids == i for ids in lakes_nc.id_geohash.values):
                ID_list_cleaned.append(i)
    else:
        ID_list_cleaned = lakes_nc.id_geohash.values

    # get initialization data
    x0_lake = lakes_nc['area_water_permanent'].sel(id_geohash=ID_list_cleaned
                                                   ,date='2000-01-01T00:00:00.000000000').values*10000
    x0_DLB = []
else:
    print('No initialization data provided. Initializing with zero lakes.')
    x0_lake = []
    x0_DLB = []

lake_nr = len(x0_lake) + len(x0_DLB) + int(form_n)
idx = np.arange(0,lake_nr)
idx = idx.astype(str)
area_water = np.zeros((n,lake_nr))
area_land = np.zeros((n,lake_nr))
age = np.zeros((n,lake_nr),dtype=object)
area_status = np.empty((n,lake_nr),dtype=object)
area_status[:] = np.nan

# initialize lake pool
idx_lake = np.empty(n,dtype=object)
idx_lake[0] =  list(range(0,len(x0_lake))) # first entry: all indices from x0_lake

# initialize DLB pool
idx_dlb = np.empty(n,dtype=object)
idx_dlb[0] =  list(range(len(x0_lake), len(x0_lake) + len(x0_DLB)))

# fill first entry (time step) with data
if len(x0_lake) != 0 or len(x0_DLB) != 0:
    area_water[0,:len(x0_lake)] = x0_lake
    area_land[0,:len(x0_lake)] = 0
    area_status[0,:len(x0_lake)] = 'L' # 'L' for lake

    area_land[:,len(x0_lake):(len(x0_lake) + len(x0_DLB))] = x0_DLB
    area_water[:,len(x0_lake):(len(x0_lake) + len(x0_DLB))] = 0
    area_status[:,len(x0_lake):(len(x0_lake) + len(x0_DLB))] = 'DLB' # 'DLB' for drained lake basin

    age[:,:len(x0_lake) + len(x0_DLB)] = 'nK' # 'nK' for not known

#%%  Simulation

# run ensemble simulation
for e in range(1,e_nr + 1):

    age = np.zeros((n,lake_nr),dtype=object)
    area_status = np.empty((n,lake_nr),dtype=object)
    area_status[:] = np.nan
    if len(x0_lake) != 0 or len(x0_DLB) != 0:
        area_status[:,len(x0_lake):(len(x0_lake) + len(x0_DLB))] = 'DLB' # 'DLB' for drained lake basin
        age[:,:len(x0_lake) + len(x0_DLB)] = 'nK' # 'nK' for not known

    # initialize lakes coordinates
    xcoord = np.zeros((n,lake_nr))
    ycoord = np.zeros((n,lake_nr))
    xcoord_ini = np.random.uniform(0,np.sqrt(A),lake_nr)
    ycoord_ini = np.random.uniform(0, np.sqrt(A),lake_nr)
    for i in range(lake_nr):
        xcoord[:,i] = xcoord_ini[i]
        ycoord[:,i] = ycoord_ini[i]

    A_drained = np.zeros(T)  # drained area
    A_drained[1:] = np.nan
    A_disturbed = np.zeros(T)  # undisturbed area
    A_disturbed[0] = A_drained[0] + np.nansum(area_water[0,:])
    A_disturbed[1:] = np.nan
    A_undisturbed = np.zeros(T)  # undisturbed area
    A_undisturbed[0] = A - min(A_disturbed[0],A)
    A_undisturbed[1:] = np.nan
    form_arr = np.zeros(n) # array with number of new lakes at each t
    drain_arr = np.zeros(n) # array with abruptly drained lakes at each t
    merged_lakes = np.zeros(n)  # counter of merged lakes
    lake_count = np.zeros(n)
    lake_count[0] = len(x0_lake)

    # loop over time steps
    for t in tqdm(range(1,n), desc="Simulating ensemble run " + str(e),file=sys.stdout):

        idx_lake[t] = idx_lake[t-1][:]
        idx_dlb[t] = idx_dlb[t-1][:]

        # merging lakes
        #area_water[t], xcoord[t], ycoord[t] = area_water[t-1], xcoord[t-1], ycoord[t-1]
        area_water[t], xcoord[t], ycoord[t] = merge(area_water[t-1], xcoord[t-1], ycoord[t-1])

        # Remove lakes with area_water of zero after merging
        for l in idx_lake[t][:]:
            if area_water[t, l] == 0 and area_water[t-1, l] != 0:
                area_water[t+1:, l] = area_water[t,l]
                area_land[t+1:, l] = area_land[t,l]
                area_status[t:, l] = "merged"
                age[t:, l] = age[t,l]
                idx_lake[t].remove(l)
                merged_lakes[t] += 1

        # calculate drift and volatility
        sigma = func_sigma(climvar[t-1])
        mu = func_mu(climvar[t-1])

        # expansion & gradual drainage
        for l in idx_lake[t]:
            # calculate new lake area for next time step
            area_water[t,l] = geometric_brownian(area_water[t,l], mu, sigma, dt)
            # Check if A_disturbed / area_water is below A_lim
            if variant == '2':
                if np.sum(area_water[t,:]) >= A_lim: 
                    area_water[t,l] = min(geometric_brownian(area_water[t,l], mu, sigma, dt), 
                                          area_water[t-1,l])
            if variant == '1':
                if (A_disturbed[t-1]+(area_water[t,l]-area_water[t-1,l])) >= A_lim:
                    area_water[t,l] = min(geometric_brownian(area_water[t,l], mu, sigma, dt), 
                                        area_water[t-1,l])
            # track age & type
            area_status[t,l] = 'L'
            if age[t-1,l] == 'nK':
                age[t,l] = 'nK'
            else:
                age[t,l] = int(age[t-1,l]) + 1

        # abrupt drainage
        if len(idx_lake[t]) != 0:
            if d_rate == 0:
                drain_nr = 0
            else:
                if variant == '1':
                    drain_nr = min(np.random.poisson(d_rate(climvar[t-1]) * A_disturbed[t-1]),
                                   len(idx_lake[t]))
                elif variant == '2':
                    drain_nr = min(np.random.poisson(((d_rate(climvar[t-1])) * dt *
                                                    np.nansum(area_water[t-1,:]))),len(idx_lake[t]))
            drain_arr[t] = drain_nr
            drain_idx = np.random.choice(idx_lake[t], drain_nr, replace=False).tolist()
            if len(drain_idx) != 0:
                for l in drain_idx:
                    area_water[t,l] = 0 # new lake area for next time step
                    idx_lake[t].remove(l)
                    idx_dlb[t].append(l)
                    # track age & type
                    area_status[t,l] = 'DLB'
                    if area_status[t-1,l] != 'DLB':
                        age[t,l] = 0
                    else: age[t,l] = age[t-1,l] + 1

        # update area_land
        for l in (idx_dlb[t] + idx_lake[t]):
            area_land[t,l] = max(area_land[t-1,l] + (area_water[t-1,l] - area_water[t,l]),0)

        # lake formation
        possible_idx = [idx for idx in range(0,lake_nr)
                if idx not in [*idx_lake[t],*idx_dlb[t]]]
        new_idx = []
        if f_rate == 0:
            form_nr = 0
        else:
            if variant == '1':
                form_nr = int(np.random.poisson((f_rate(climvar[t-1]) * A_undisturbed[t-1] * dt)))
            elif variant == '2':
                form_nr = int(np.random.poisson((f_rate(climvar[t-1]) * \
                                                max((A - np.nansum(area_water[t-1,:])),0) * dt)))
        form_arr[t] = form_nr
        for l in possible_idx[:int(form_nr)]:
            area_water[t,l] = 1
            new_idx.append(l)
        idx_lake[t] = [*idx_lake[t], *new_idx]

        # update A_drained, A_disturbed, A_undisturbed and lake_count
        A_drained[t] = max(A_drained[t-1] + (np.nansum(area_water[t-1,:]) - np.nansum(area_water[t,:])),0)
        A_disturbed[t] = min(A_drained[t] + np.nansum(area_water[t,:]),A)
        A_undisturbed[t] = A - min(A_disturbed[t],A)
        lake_count[t] = len(idx_lake[t])

    # create the xarray Dataset with latitude and longitude included
    age = age.astype(str)
    area_status = area_status.astype(str)
    lakes = xr.Dataset(
        {
            "area_water_permanent": (("date", "id_geohash"), area_water),
            "area_water_seasonal": (("date", "id_geohash"), np.zeros((n,lake_nr))),
            "area_land": (("date", "id_geohash"), area_land),
            "area_nodata": (("date", "id_geohash"), np.zeros((n,lake_nr))),
            "age": (("date", "id_geohash"), age),
            "type": (("date", "id_geohash"), area_status),
            "xcoord": (("date", "id_geohash"), xcoord),
            "ycoord": (("date", "id_geohash"), ycoord)
        },
        coords={
            "date": years,
            "id_geohash": idx
        }
    )

    # save output as netcdf file
    lakes.to_netcdf('output/lakes_' + str(e) + '.nc')
    # close netcdf file
    lakes.close()

    # calculate timeseries of lake number, lake area fraction and drained area fraction
    time_points = np.arange(0, T, dt)
    area_water_frac = np.zeros(n)
    area_drained_frac = np.zeros(n)
    for t in range(n):
        area_water_frac[t] = np.nansum(area_water[t,:]) / A
        area_drained_frac[t] = A_drained[t] / A

    # save numbers and area fractions as txt files
    np.savetxt('output/area_water_frac_' + str(e) + '.txt', area_water_frac)
    np.savetxt('output/area_drained_frac_' + str(e) + '.txt', area_drained_frac)
    np.savetxt('output/lake_nr_' + str(e) + '.txt', lake_count)

    # save arrays with number of new, drained and merged lakes
    np.savetxt('output/drain_arr_' + str(e) + '.txt', drain_arr)
    np.savetxt('output/form_arr_' + str(e) + '.txt', form_arr)
    np.savetxt('output/merged_lakes_' + str(e) + '.txt', merged_lakes)

#%%
print('Simulations finished!')
# %%
