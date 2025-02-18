#%%
'''
FILENAME:
    plotting.py

DESCRIPTION:
    This is a script for plotting timeseries of lake and drained area as well as
    lake number from output files of the Thermokarst Lake Model (TLM).  

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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import netCDF4 as nc
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

#%% directories

# set working directory
os.chdir(os.path.join( os.path.dirname( __file__ ), '..' ))

#%% Import the necessary parameter

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print(f"Error: Expected 1 argument, but got {len(sys.argv) - 1}.", file=sys.stderr)
    sys.exit(1)  # Exit the script if the number of arguments is incorrect

# Assign each command-line argument to a variable, converting to the appropriate type
e_nr = int(sys.argv[1])  # Convert e_nr to integer


#%% Initialize dictionary with all ensemble members

ensemble = {
    'date': None,
    'water_frac': [],
    'drained_frac': [],
    'lake_nr': []
}

#%% import lake dataset and plot

for e in range(1,e_nr+1):

    ds = nc.Dataset('output/lakes_' + str(e) + '.nc')

    area_water_frac = np.loadtxt('output/area_water_frac_' + str(e) + '.txt')
    area_drained_frac = np.loadtxt('output/area_drained_frac_' + str(e) + '.txt')
    lake_nr = np.loadtxt('output/lake_nr_' + str(e) + '.txt')

    N = len(lake_nr)

    # fill ensemble dictionary with data from the current ensemble member
    if ensemble['date'] is None:
        ensemble['date'] = np.arange(2000, 2000 + N)

    # Compute the mean across id_geohash and append to the ensemble dictionary
    ensemble['water_frac'].append(area_water_frac)
    ensemble['drained_frac'].append(area_drained_frac)
    ensemble['lake_nr'].append(lake_nr)

    for e in range(1,e_nr+1):

        PATH = "plots/run_" + str(e)
        os.makedirs("plots/run_" + str(e), exist_ok=True)

        # create circle plots
        os.makedirs(PATH + "/circles", exist_ok=True)

        def yr(x,pos):
            """turn x axis value from m to km"""
            return x/1000
        formatter = FuncFormatter(yr)

        plt.rcParams.update({'font.size': 18})
        blue_patch = mpatches.Patch(color='blue', label='water')
        brown_patch = mpatches.Patch(color='brown', label='drained')


#%% ensemble plot

color_list = list(mcolors.CSS4_COLORS.keys())
random_colors = np.random.choice(color_list, e_nr)
#random_colors = ['red', 'blue', 'green', 'orange', 'purple']

class MulticolorPatch(object):
    """Create a legend with colored patches for each ensemble member"""
    def __init__(self, colors):
        self.colors = colors

class MulticolorPatchHandler(object):
    """Handler for the MulticolorPatch class"""
    def legend_artist(self, orig_handle, handlebox):
        """Create legend patch with colored patches for each ensemble member"""
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i -
                                          handlebox.xdescent, -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height,
                           facecolor=c,
                           edgecolor='none'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch


# Create figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 6.5),sharex=True)

if e_nr > 1:

    for n in range(e_nr):
        axs[0].plot(ensemble['date'], ensemble['water_frac'][n], alpha=0.5,
                    color = random_colors[n])
        axs[1].plot(ensemble['date'], ensemble['drained_frac'][n], alpha=0.5,
                    color = random_colors[n])

    ensemble_mean_area_water_permanent = np.mean(ensemble['water_frac'], axis=0)
    ensemble_std = np.std(ensemble['water_frac'], axis = 0)
    axs[0].fill_between(ensemble['date'], ensemble_mean_area_water_permanent -
                        ensemble_std, ensemble_mean_area_water_permanent +
                        ensemble_std, color='black', alpha=0.2, label ='Standard Deviation')
    axs[0].plot(ensemble['date'], ensemble_mean_area_water_permanent, color='black',
                linewidth=2, label='Ensemble Mean')
    axs[0].set_ylabel('Water area fraction')

    ensemble_mean_area_land = np.mean(ensemble['drained_frac'], axis=0)
    ensemble_std_drained = np.std(ensemble['drained_frac'], axis = 0)
    axs[1].fill_between(ensemble['date'], ensemble_mean_area_land - ensemble_std_drained,
                        ensemble_mean_area_land + ensemble_std_drained, color='black',
                        alpha=0.2, label ='Standard Deviation')
    axs[1].plot(ensemble['date'], ensemble_mean_area_land, color='black',
                linewidth=2, label='Ensemble Mean')
    axs[1].set_ylabel('Drained area fraction')
    axs[1].set_xlabel('Year')

    # Add legends to each subplot
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(MulticolorPatch(random_colors))
        labels.append("Ensemble Members")
        ax.legend(handles, labels, loc='upper left',
                  handler_map={MulticolorPatch: MulticolorPatchHandler()})

elif e_nr == 1:
    axs[0].plot(ensemble['date'], ensemble['water_frac'][0], color = 'black',
                linewidth=2, label='Water area fraction')
    axs[1].plot(ensemble['date'], ensemble['drained_frac'][0], color = 'black',
                linewidth=2, label='Drained area fraction')
    axs[0].set_ylabel('Water area fraction')
    axs[1].set_ylabel('Drained area fraction')
    axs[1].set_xlabel('Year')

plt.tight_layout()
plt.savefig("plots/ensemble_plot.png", dpi = 200,bbox_inches="tight")

#%%
print("Plots created successfully!")
