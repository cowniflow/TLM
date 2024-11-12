# Thermokarst Lake Model
# by Constanze Reinken 2024

#%% import packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import math
import pandas as pd

#%% define functions

# Geometric Brownian Motion
def geometric_brownian(x0, mu, sigma, dt):
    x = x0 * np.exp((mu - ((sigma ** 2) / 2)) * dt
                + sigma * np.random.normal(0, np.sqrt(dt)))
    return x


# Merging Algorithm
def merge(area_water, xcoord, ycoord):

    area_water_new = area_water.copy()
    xcoord_new = xcoord.copy()
    ycoord_new = ycoord.copy()

    # select all circle objects with an area
    idx_circles = [i for i in range(len(area_water)) if area_water[i] != 0 and not math.isnan(area_water[i])]

    # loop through all circles
    if len(idx_circles) != 0:
        for i in idx_circles: 
            for j in idx_circles:

                if i != j and area_water_new[i] != 0 and area_water_new[j] != 0:     
                    # calculate distance between centers
                    dx = abs(xcoord_new[i] - xcoord_new[j])
                    dy = abs(ycoord_new[i] - ycoord_new[j])
                    dist = (dx**2 + dy**2)**0.5 
                            
                    # check for overlapping circles 
                    if (dist + 0.01) < np.sqrt(area_water_new[i]/np.pi) + np.sqrt(area_water_new[j]/np.pi):

                        # calculate new centre (centre of mass) and transfer area to the bigger one of the circles
                        if area_water[j] >= area_water[i]:
                            xcoord_new[j] = (1 / (area_water_new[j] + area_water_new[i])) * np.sum((area_water_new[j] * xcoord_new[j], area_water_new[i] * xcoord_new[i]))
                            ycoord_new[j] = (1 / (area_water_new[j] + area_water_new[i])) * np.sum((area_water_new[j] * ycoord_new[j], area_water_new[i] * ycoord_new[i]))
                            area_water_new[j] += area_water_new[i]
                            area_water_new[i] = 0

                        else:
                            xcoord_new[i] = (1 / (area_water_new[i] + area_water_new[j])) * np.sum((area_water_new[i] * xcoord_new[i], area_water_new[j] * xcoord_new[j]))
                            ycoord_new[i] = (1 / (area_water_new[i] + area_water_new[j])) * np.sum((area_water_new[i] * ycoord_new[i], area_water_new[j] * ycoord_new[j]))
                            area_water_new[i] += area_water_new[j]
                            area_water_new[j] = 0
                        
    return area_water_new, xcoord_new, ycoord_new


#%% Import the necessary parameter, initialization data and forcing

import sys  # Import sys to access command-line arguments

# Check if the correct number of arguments is provided (5 arguments including the script name)
if len(sys.argv) != 8:
    sys.exit(1)  # Exit the script if the number of arguments is incorrect

# Assign each command-line argument to a variable, converting to the appropriate type
A_cell = float(sys.argv[1])  
gr_ice = float(sys.argv[2])  
T = int(sys.argv[3]) 
dt = float(sys.argv[4])  
clim_param_func = sys.argv[5]  
e_nr = int(sys.argv[6]) 
ini_lakes = sys.argv[7]  
forcing = sys.argv[8]  


#%% import parameter values and / or functions

import importlib

module = importlib.import_module(clim_param_func)

# Update the global namespace with the imported module's contents
globals().update(module.__dict__)

#%% define spatial domain, temporal domain & time step

n = int(T/dt) # nr of steps
years = np.arange(1900, 1900 + T, dt) # time points

#%% define maximum potential disturbed area

A_lim = A_cell * gr_ice # maximum potential disturbed area (depends on ground ice content)

#%% load climate data

tdd = np.loadtxt(forcing)[:-1]
original_time_points = np.linspace(0, len(tdd) - 1, len(tdd))
new_time_points = np.linspace(0, len(tdd) - 1, n)
tdd = np.interp(new_time_points, original_time_points, tdd)

#%% initialization

#f_rate = np.max(func_f_rate_sLake(tdd)) # maximum formation rate
f_rate = np.max(func_f_rate_sDist(tdd)) # maximum formation rate
# generate max. number of potential new lakes
form_n = math.ceil(A_cell*1e6 * T * f_rate)

# import / create initialization data (lake sizes, DLB sizes)
import geopandas as gpd
lake_dataset = xr.open_dataset(path+file)
lakes_df = lake_dataset.to_dataframe()
# Remove duplicates
lakes_df = lakes_df.loc[~lakes_df.index.duplicated(keep='first')]
# Convert the DataFrame back to a Dataset
lakes_nc = lakes_df.to_xarray()
region = gpd.read_file(ini_lakes)
ID_list = list(region['id_geohash'])
ID_list_cleaned = []
for i in ID_list:
    if any(ids == i for ids in lakes_nc.id_geohash.values):
        ID_list_cleaned.append(i)
# Concatenate lake_dataset with itself
#x0_lake =  (region['Area_start']*0.01).tolist()
x0_lake = lakes_nc['area_water_permanent'].sel(id_geohash=ID_list_cleaned,date='2000-01-01T00:00:00.000000000').values*0.01
x0_DLB = [] # initial DLB sizes



#%%
# initialize
lake_nr = len(x0_lake) + len(x0_DLB) + int(form_n)
id = np.arange(0,lake_nr)
id = id.astype(str)

area_water = np.zeros((n,lake_nr))
area_land = np.zeros((n,lake_nr))
#area_merge = np.zeros((n,lake_nr),dtype=object)
age = np.zeros((n,lake_nr),dtype=object)
type = np.empty((n,lake_nr),dtype=object)
type[:] = np.nan

# initialize lake pool
idx_lake = np.empty((n),dtype=object)
idx_lake[0] =  list(range(0,len(x0_lake))) # first entry: all indices from x0_lake

# initialize DLB pool
idx_dlb = np.empty((n),dtype=object)
idx_dlb[0] =  list(range(len(x0_lake), len(x0_lake) + len(x0_DLB)))

# initialize first time step with data
if len(x0_lake) != 0 or len(x0_DLB) != 0:
    area_water[0,:len(x0_lake)] = x0_lake
    area_land[0,:len(x0_lake)] = 0
    type[0,:len(x0_lake)] = 'L' # 'L' for lake

    area_land[:,len(x0_lake):(len(x0_lake) + len(x0_DLB))] = x0_DLB
    area_water[:,len(x0_lake):(len(x0_lake) + len(x0_DLB))] = 0
    type[:,len(x0_lake):(len(x0_lake) + len(x0_DLB))] = 'DLB' # 'DLB' for drained lake basin

    age[:,:len(x0_lake) + len(x0_DLB)] = 'nK' # 'nK' for not known

    #area_merge[:] = 0


#%% spatial distribution of initial lakes

# xcoord = np.zeros((n,lake_nr))
# ycoord = np.zeros((n,lake_nr))
# for i in range(lake_nr,len(x0_lake)+lake_nr):
#     xcoord[:], ycoord[:] = distribute_circles(x0_lake, np.sqrt(A_cell))

# # get centroids of initial lakes
# latitudes = lake_dataset['geometry'].centroid.x
# longitudes = lake_dataset['geometry'].centroid.y

# min_longitude = min(longitudes)
# max_longitude = max(longitudes)
# min_latitude = min(latitudes)
# max_latitude = max(latitudes)

# normalized_longitudes = [((lon - min_longitude) / (max_longitude - min_longitude))*np.sqrt(A_cell) for lon in longitudes]
# normalized_latitudes = [((lat - min_latitude) / (max_latitude - min_latitude))*np.sqrt(A_cell) for lat in latitudes]


# import matplotlib.patches as mpatches

# # Plot the data
# fig, ax = plt.subplots(figsize=(10,10))
# plt.xlim(0, np.sqrt(A_cell))
# plt.ylim(0, np.sqrt(A_cell))
# l_circles = []
# blue_patch = mpatches.Patch(color='blue', label='water')
# for i in range(len(x0_lake)):
#     l_circles.append(plt.Circle((xcoord[0,i], y_coord[:,i]), np.sqrt(x0_lake[i])/np.pi, color = 'blue'))
#     ax.add_patch(l_circles[i])

# plt.xlabel('Normalized Longitude')
# plt.ylabel('Normalized Latitude')
# plt.title('Normalized Coordinates on Square Plot')
# plt.show()



#%%  Simulation

# run ensemble simulation
for e in range(1,e_nr + 1):


    age = np.zeros((n,lake_nr),dtype=object)
    type = np.empty((n,lake_nr),dtype=object)
    type[:] = np.nan

    if len(x0_lake) != 0 or len(x0_DLB) != 0:
        type[:,len(x0_lake):(len(x0_lake) + len(x0_DLB))] = 'DLB' # 'DLB' for drained lake basin
        age[:,:len(x0_lake) + len(x0_DLB)] = 'nK' # 'nK' for not known

    # initialize lakes coordinates

    xcoord = np.zeros((n,lake_nr))
    ycoord = np.zeros((n,lake_nr))

    xcoord_ini = np.random.uniform(0,np.sqrt(A_cell),(lake_nr))
    ycoord_ini = np.random.uniform(0, np.sqrt(A_cell),(lake_nr))

    for i in range(lake_nr):
        xcoord[:,i] = xcoord_ini[i]
        ycoord[:,i] = ycoord_ini[i]

    A_drained = np.zeros(T)  # drained area
    A_drained[1:] = np.nan

    A_disturbed = np.zeros(T)  # undisturbed area
    A_disturbed[0] = A_drained[0] + np.nansum(area_water[0,:])
    A_disturbed[1:] = np.nan

    A_undisturbed = np.zeros(T)  # undisturbed area
    A_undisturbed[0] = A_cell - min(A_disturbed[0],A_cell)
    A_undisturbed[1:] = np.nan

    form_arr = np.zeros(n) # array with number of new lakes at each t
    drain_arr = np.zeros(n) # array with abruptly drained lakes at each t

    merged_lakes = np.zeros(n)  # counter of merged lakes

    lake_count = np.zeros(n)
    lake_count[0] = len(x0_lake)

    # loop over time steps
    for t in range(1,n):

        # time step tracker
        print("t = " + str(t))

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
                type[t:, l] = "merged"
                age[t:, l] = age[t,l]
                idx_lake[t].remove(l)
                merged_lakes[t] += 1
                #area_merge[t:,l] = max((area_water[t-1,l] - area_water[t,l]),0)

        # calculate drift and volatility
        sigma = func_sigma(tdd[t-1])
        mu = func_mu(tdd[t-1])

        # expansion & gradual drainage
        for l in idx_lake[t]:
            area_water[t,l] = geometric_brownian(area_water[t,l], mu, sigma, dt) # calculate new lake area for next time step
            if (A_disturbed[t-1]+(area_water[t,l]-area_water[t-1,l])) >= A_lim:
            #if np.sum(area_water[t,:]) >= A_lim:  # Check if A_disturbed is below A_lim
                area_water[t,l] = min(geometric_brownian(area_water[t,l], mu, sigma, dt), area_water[t-1,l]) # calculate new lake area for next time step
            # track age & type
            type[t,l] = 'L'
            if age[t-1,l] == 'nK':
                age[t,l] = 'nK'
            else:
                age[t,l] = int(age[t-1,l]) + 1

        # calculate d_rate
        #d_rate = func_d_rate_sLake(tdd[t-1])
        d_rate = func_d_rate_sDist(tdd[t-1])

        # abrupt drainage
        if len(idx_lake[t]) != 0:
            if d_rate == 0:
                drain_nr = 0
            else:
                drain_nr = min(np.random.poisson(d_rate * A_disturbed[t-1]*1e-6),len(idx_lake[t]))
                #drain_nr = min(np.random.poisson(((d_rate)*dt*np.nansum(area_water[t-1,:])*1e6)),len(idx_lake[t]))
            drain_arr[t] = drain_nr
            drain_idx = np.random.choice(idx_lake[t], drain_nr, replace=False).tolist()
            if len(drain_idx) != 0:
                for l in drain_idx:
                    area_water[t,l] = 0 # new lake area for next time step
                    idx_lake[t].remove(l)
                    idx_dlb[t].append(l)
                    # track age & type
                    type[t,l] = 'DLB'
                    if type[t-1,l] != 'DLB':
                        age[t,l] = 0
                    else: age[t,l] = age[t-1,l] + 1

        for l in (idx_dlb[t] + idx_lake[t]):
            area_land[t,l] = max(area_land[t-1,l] + (area_water[t-1,l] - area_water[t,l]),0)

        # calculate f_rate
        #f_rate = func_f_rate_sLake(tdd[t-1])
        f_rate = func_f_rate_sDist(tdd[t-1])

        # add new (expanding) lakes via idx_exp
        possible_idx = [idx for idx in range(0,lake_nr)
                if idx not in [*idx_lake[t],*idx_dlb[t]]]
        new_idx = []
        if f_rate == 0:
            form_nr = 0
        else:
            form_nr = int(np.random.poisson((f_rate*A_undisturbed[t-1]*1e6*dt)))
            #form_nr = int(np.random.poisson((f_rate*max((A_cell - np.nansum(area_water[t-1,:]))*1e6,0)*dt)))
        form_arr[t] = form_nr
        for l in possible_idx[:int(form_nr)]:
            area_water[t,l] = 1
            new_idx.append(l)
        idx_lake[t] = [*idx_lake[t], *new_idx]

        A_drained[t] = max(A_drained[t-1] + (np.nansum(area_water[t-1,:]) - np.nansum(area_water[t,:])),0)
        #A_drained[t] = total_area(xcoord[t], ycoord[t], area_water[t], area_land[t])
        A_disturbed[t] = min(A_drained[t] + np.nansum(area_water[t,:]),A_cell)
        A_undisturbed[t] = A_cell - min(A_disturbed[t],A_cell)

        lake_count[t] = len(idx_lake[t])

    # Creating the xarray Dataset with latitude and longitude included

    age = age.astype(str)
    type = type.astype(str)

    lakes = xr.Dataset(
        {
            "area_water_permanent": (("date", "id_geohash"), area_water),
            "area_water_seasonal": (("date", "id_geohash"), np.zeros((n,lake_nr))),
            "area_land": (("date", "id_geohash"), area_land),
            "area_nodata": (("date", "id_geohash"), np.zeros((n,lake_nr))),
            "age": (("date", "id_geohash"), age),
            "type": (("date", "id_geohash"), type),
            "xcoord": (("date", "id_geohash"), xcoord),
            "ycoord": (("date", "id_geohash"), ycoord)
        },
        coords={
            "date": years,
            "id_geohash": id
        }
    )

    # save output as netcdf file
    lakes.to_netcdf('data/lakes_' + str(e) + '.nc')

    # close netcdf file
    lakes.close()

    # calculate timeseries of lake number, lake area fraction and drained area fraction

    # Generate the time points for each step
    time_points = np.arange(0, T, dt)

    area_water_frac = np.zeros(n)
    area_drained_frac = np.zeros(n)

    for t in range(n):
        area_water_frac[t] = np.nansum(area_water[t,:]) / A_cell
        area_drained_frac[t] = A_drained[t] / A_cell


    # save numbers and area fractions as txt files
    np.savetxt('data/area_water_frac_' + str(e) + '.txt', area_water_frac)
    np.savetxt('data/area_drained_frac_' + str(e) + '.txt', area_drained_frac)
    np.savetxt('data/lake_nr_' + str(e) + '.txt', lake_count)


    # plot results

    # Create figure and primary axis
    fig, ax1 = plt.subplots()

    # Plot area fractions on primary axis
    ax1.plot(area_water_frac, label='Water', color='blue')
    ax1.plot(area_drained_frac, label='Drained', color='brown')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Area fractions')

    # Create secondary axis for 'number'
    ax2 = ax1.twinx()
    ax2.plot(lake_count, label='Number', color='black')
    ax2.set_ylabel('Number of lakes')

    # Add legends and title
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    # Show plot
    plt.show()
    plt.show(block=False)
    plt.pause(0.1)  # Pause for a short period to ensure the plot is rendered

    np.savetxt('data/drain_arr_' + str(e) + '.txt', drain_arr)
    np.savetxt('data/form_arr_' + str(e) + '.txt', form_arr)
    np.savetxt('data/merged_lakes_' + str(e) + '.txt', merged_lakes)

#%%
print('Simulation finished!')
# %%
