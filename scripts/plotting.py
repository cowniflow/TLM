# Plot lake gifs and timeseries

#%% import packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import imageio
import netCDF4 as nc
import os
import moviepy.video.io.ImageSequenceClip
from matplotlib.ticker import FuncFormatter


# #%% Import the necessary parameter
# import sys  # Import sys to access command-line arguments

# # Check if the correct number of arguments is provided (5 arguments including the script name)
# if len(sys.argv) != 2:
#     sys.exit(1)  # Exit the script if the number of arguments is incorrect

# # Assign each command-line argument to a variable, converting to the appropriate type
# A_cell = float(sys.argv[1])  # Convert A_cell to float
# e_nr = int(sys.argv[2])  # Convert e_nr to integer

#%%  temporary parameter values for testing (Usually would be imported from command line arguments)

A_cell = 40*40 * 1e6 #km²
e_nr = 5

#%% Initialize dictionary with all ensemble members

ensemble = {
    'date': None,
    'water_frac': [],
    'drained_frac': [],
    'lake_nr': []
}



#%% import lake dataset and plot

dt = 20

for e in range(1,e_nr+1):

    #ds = nc.Dataset('observation_based_Variant2_06Sep/lakes_' + str(e) + '.nc')

    area_water_frac = np.loadtxt('/Users/constanzereinken/Scripts/Lake_Model/object_model/final/Plot1/Variant1/merge/area_water_frac_' + str(e) + '.txt')
    area_drained_frac = np.loadtxt('/Users/constanzereinken/Scripts/Lake_Model/object_model/final/Plot1/Variant1/merge/area_drained_frac_' + str(e) + '.txt')
    lake_nr = np.loadtxt('/Users/constanzereinken/Scripts/Lake_Model/object_model/final/Plot1/Variant1/merge/lake_nr_' + str(e) + '.txt')

    N = len(lake_nr)

    # fill ensemble dictionary with data from the current ensemble member
    if ensemble['date'] is None:
        ensemble['date'] = np.arange(2000, 2000 + N)

    # Compute the mean across id_geohash and append to the ensemble dictionary
    ensemble['water_frac'].append(area_water_frac)
    ensemble['drained_frac'].append(area_drained_frac)
    ensemble['lake_nr'].append(lake_nr)

    #for i in np.random.choice(e_nr, max(int(e_nr*0.3),1)):
    for i in range(1,e_nr+1):

        path = "plots/run_" + str(e)
        os.makedirs("plots/run_" + str(e), exist_ok=True)

        # create circle plots
        os.makedirs(path + "/circles", exist_ok=True)

        # turn x axis label from m into km
        # def yr(x,pos):
        #     return (x/1000)
        # formatter = FuncFormatter(yr)

        # for n in range(0,N,dt):
        #     fig, ax = plt.subplots(figsize=(10,10))
        #     plt.rcParams.update({'font.size': 18})
        #     ax.set_aspect('equal')
        #     ax.set_xlim(0,(np.sqrt(A_cell)))
        #     ax.set_ylim(0,(np.sqrt(A_cell)))
        #     #ax.xaxis.set_major_formatter(formatter)
        #     #ax.yaxis.set_major_formatter(formatter)
        #     d_rings = []
        #     l_circles = []
        #     blue_patch = mpatches.Patch(color='blue', label='water')
        #     brown_patch = mpatches.Patch(color='brown', label='drained area since start of simulations')
        #     for i in range(len(ds['id_geohash'])):
        #         d_rings.append(plt.Circle((ds['xcoord'][n,i], ds['ycoord'][n,i]), np.sqrt(((ds['area_land'][n,i]) + ds['area_water_permanent'][n,i])/np.pi), color='brown',zorder=1))
        #         l_circles.append(plt.Circle((ds['xcoord'][n,i], ds['ycoord'][n,i]), np.sqrt((ds['area_water_permanent'][n,i])/np.pi), color='blue', zorder=2))
        #         ax.add_patch(d_rings[i])
        #         ax.add_patch(l_circles[i])
        #     if n == 0:
        #         plt.legend(handles=[blue_patch], loc=1).set_zorder(102)
        #     else:
        #         plt.legend(handles=[blue_patch, brown_patch], loc=1).set_zorder(102)
        #     plt.title("Lakes at year " + str(2000 + n),fontsize=18)
        #     plt.xlabel("Distance / km",fontsize=18)
        #     plt.ylabel("Distance / km",fontsize=18)
        #     plt.savefig(path + "/circles/lakes_" + str(n) + ".png", dpi = 200,bbox_inches="tight")
        #     plt.show()
        #     plt.close()

        # # create gif of circles
        # images = []
        # for n in range(0,N,dt):
        #     images.append(imageio.imread(path + "/circles/lakes_" + str(n) + ".png"))
        # imageio.mimsave(path + '/lake_evolution.gif', images, duration = 1, loop=1)

        # # create video of circles
        # images = []
        # for n in range(0,N,dt):
        #     images.append(imageio.imread(path + "/circles/lakes_" + str(n) + ".png"))

        # fps = 10
        # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
        # clip.write_videofile(path + '/lake_evolution_slow.mp4')


#%% ensemble plot

import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

colors = list(mcolors.CSS4_COLORS.keys())
#random_colors = np.random.choice(colors, e_nr)
random_colors = ['red', 'blue', 'green', 'orange', 'purple']

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors

# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent,
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height,
                           facecolor=c,
                           edgecolor='none'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch


# Create figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 6.5),sharex=True)


for n in range(e_nr):
    axs[0].plot(ensemble['date'], ensemble['water_frac'][n], alpha=0.5, color = random_colors[n])
    axs[1].plot(ensemble['date'], ensemble['drained_frac'][n], alpha=0.5, color = random_colors[n])

ensemble_mean_area_water_permanent = np.mean(ensemble['water_frac'], axis=0)
ensemble_std = np.std(ensemble['water_frac'], axis = 0)
axs[0].fill_between(ensemble['date'], ensemble_mean_area_water_permanent - ensemble_std, ensemble_mean_area_water_permanent + ensemble_std, color='black', alpha=0.2, label ='Standard Deviation')
axs[0].plot(ensemble['date'], ensemble_mean_area_water_permanent, color='black', linewidth=2, label='Ensemble Mean')
axs[0].set_ylabel('Water area fraction')
axs[0].set_ylim(0, 1)

ensemble_mean_area_land = np.mean(ensemble['drained_frac'], axis=0)
ensemble_std_drained = np.std(ensemble['drained_frac'], axis = 0)
axs[1].fill_between(ensemble['date'], ensemble_mean_area_land - ensemble_std_drained, ensemble_mean_area_land + ensemble_std_drained, color='black', alpha=0.2, label ='Standard Deviation')
axs[1].plot(ensemble['date'], ensemble_mean_area_land, color='black', linewidth=2, label='Ensemble Mean')
axs[1].set_ylabel('Drained area fraction')
axs[1].set_ylim(0, 1)
axs[1].set_xlabel('Year')

# Add legends to each subplot
for ax in axs:
    handles, labels = ax.get_legend_handles_labels()
    handles.append(MulticolorPatch(random_colors))
    labels.append("Ensemble Members")
    ax.legend(handles, labels, loc='upper left', handler_map={MulticolorPatch: MulticolorPatchHandler()})

plt.tight_layout()
plt.savefig("plots/ensemble_plot.png", dpi = 200,bbox_inches="tight")
plt.show()
# %% plot with obs data

#%% directories

path = '/Users/constanzereinken/Data/drive-download-20240119T112608Z-001/'

#%% Area of study region

# artificial dataset
#A = 48623e6  #m2
A = 40*40 * 1e6

#%% import timeseries dataset

import xarray as xr
import geopandas as gpd

path = '/Users/constanzereinken/Data/drive-download-20240119T112608Z-001/'

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

# import shapefile for area of interest, with lakes as polygons

region = gpd.read_file(path + 'UTM54_North_ini_40x40.shp')

ID_list = list(region['id_geohash'])
ID_list_cleaned = []

for i in ID_list:
    if any(ids == i for ids in lakes.id_geohash.values):
        ID_list_cleaned.append(i)

lakes_subset = lakes.sel(id_geohash = ID_list_cleaned, date = years)

water_frac_obs = (lakes_subset['area_water_permanent'].sum(dim='id_geohash')*10000) / A_cell
# %%


ensemble_nomerge = {
    'date': None,
    'water_frac': [],
    'drained_frac': [],
    'lake_nr': []
}


for e in range(1,e_nr+1):

    area_water_frac_nomerge = np.loadtxt('/Users/constanzereinken/Scripts/Lake_Model/object_model/final/Plot1/Variant1/no_merge/area_water_frac_' + str(e) + '.txt')
    area_drained_frac_nomerge = np.loadtxt('/Users/constanzereinken/Scripts/Lake_Model/object_model/final/Plot1/Variant1/no_merge/area_drained_frac_' + str(e) + '.txt')
    lake_nr_nomerge = np.loadtxt('/Users/constanzereinken/Scripts/Lake_Model/object_model/final/Plot1/Variant1/no_merge/lake_nr_' + str(e) + '.txt')

    # fill ensemble dictionary with data from the current ensemble member
    if ensemble_nomerge['date'] is None:
        ensemble_nomerge['date'] = np.arange(2000, 2000 + N)

    # Compute the mean across id_geohash and append to the ensemble dictionary
    ensemble_nomerge['water_frac'].append(area_water_frac_nomerge)
    ensemble_nomerge['drained_frac'].append(area_drained_frac_nomerge)
    ensemble_nomerge['lake_nr'].append(lake_nr_nomerge)

#%%
fig = plt.figure(figsize=(10, 3.25))

ensemble_mean_area_water_permanent = np.mean(ensemble['water_frac'], axis=0)
ensemble_std = np.std(ensemble['water_frac'], axis = 0)
ensemble_mean_area_water_permanent_nomerge = np.mean(ensemble_nomerge['water_frac'], axis=0)
ensemble_std_nomerge = np.std(ensemble_nomerge['water_frac'], axis = 0)
plt.fill_between(ensemble['date'][:22], ensemble_mean_area_water_permanent[:22] - ensemble_std[:22], ensemble_mean_area_water_permanent[:22] + ensemble_std[:22], color='green', alpha=0.2,)
plt.fill_between(ensemble['date'][:22], ensemble_mean_area_water_permanent_nomerge[:22] - ensemble_std_nomerge[:22], ensemble_mean_area_water_permanent_nomerge[:22] + ensemble_std_nomerge[:22], color='blue', alpha=0.2)
plt.plot(ensemble['date'][:22], ensemble_mean_area_water_permanent[:22], color='green', linewidth=2, label='Ensemble Mean with merging')
plt.plot(ensemble['date'][:22], ensemble_mean_area_water_permanent_nomerge[:22], color='blue', linewidth=2, label='Ensemble Mean without merging')
plt.ylabel('Water area fraction')
plt.ylim(0, 0.5)
plt.plot(ensemble['date'][:22], water_frac_obs[:22], color='red', label='Observations',linewidth=2)
plt.xlabel('Year')

# Add legends to each subplot
handles, labels = ax.get_legend_handles_labels()
handles.append(MulticolorPatch(random_colors))
labels.append("Ensemble Members")
plt.legend()

plt.tight_layout()
plt.savefig("plots/ensemble_plot2022.png", dpi = 200,bbox_inches="tight")
plt.show()


# %%