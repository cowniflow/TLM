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
from tqdm import tqdm

#%% directories

# set working directory
import os
os.chdir(os.path.join( os.path.dirname( __file__ ), '..' ))

#%% Import the necessary parameter
import sys  # Import sys to access command-line arguments

# Check if the correct number of arguments is provided
if len(sys.argv) != 4:
    print(f"Error: Expected 3 arguments, but got {len(sys.argv) - 1}.", file=sys.stderr)
    sys.exit(1)  # Exit the script if the number of arguments is incorrect
    
# Assign each command-line argument to a variable, converting to the appropriate type
A_cell = float(sys.argv[1])  # Convert A_cell to float
e_nr = int(sys.argv[2])  # Convert e_nr to integer
dt = int(sys.argv[3])  # Convert dt to integer


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

    #for i in np.random.choice(e_nr, max(int(e_nr*0.3),1)):
    for i in range(1,e_nr+1):

        path = "plots/run_" + str(e)
        os.makedirs("plots/run_" + str(e), exist_ok=True)

        # create circle plots
        os.makedirs(path + "/circles", exist_ok=True)

        # turn x axis label from m into km
        def yr(x,pos):
            return (x/1000)
        formatter = FuncFormatter(yr)

        for n in tqdm(range(0,N,dt), desc="Plotting ensemble " + str(e),file=sys.stdout):
            fig, ax = plt.subplots(figsize=(10,10), dpi=200)
            plt.rcParams.update({'font.size': 18})
            ax.set_aspect('equal')
            ax.set_xlim(0,(np.sqrt(A_cell)))
            ax.set_ylim(0,(np.sqrt(A_cell)))
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            d_rings = []
            l_circles = []
            blue_patch = mpatches.Patch(color='blue', label='water')
            brown_patch = mpatches.Patch(color='brown', label='drained area since start of simulations')
            for i in range(len(ds['id_geohash'])):
                d_rings.append(plt.Circle((ds['xcoord'][n,i], ds['ycoord'][n,i]), np.sqrt(((ds['area_land'][n,i]) + ds['area_water_permanent'][n,i])/np.pi), color='brown',zorder=1))
                l_circles.append(plt.Circle((ds['xcoord'][n,i], ds['ycoord'][n,i]), np.sqrt((ds['area_water_permanent'][n,i])/np.pi), color='blue', zorder=2))
                ax.add_patch(d_rings[i])
                ax.add_patch(l_circles[i])
            if n == 0:
                plt.legend(handles=[blue_patch], loc=1).set_zorder(102)
            else:
                plt.legend(handles=[blue_patch, brown_patch], loc=1).set_zorder(102)
            plt.title("Lakes at year " + str(2000 + n),fontsize=18)
            plt.xlabel("Distance / km",fontsize=18)
            plt.ylabel("Distance / km",fontsize=18)
            plt.savefig(path + "/circles/lakes_" + str(n) + ".png", dpi = 200,bbox_inches="tight")
            plt.close()

        # create gif of circles
        images = []
        for n in range(0,N,dt):
            img = imageio.imread(path + "/circles/lakes_" + str(n) + ".png")
            if n == 0:
                img_shape = img.shape
            elif img.shape != img_shape:
                raise ValueError(f"Image at index {n} has a different shape: {img.shape} compared to {img_shape}")
            images.append(img)
        imageio.mimsave(path + '/lake_evolution.gif', images, duration = 1, loop=1)

        # create video of circles
        images = []
        for n in range(0, N, dt):
            img = imageio.imread(path + "/circles/lakes_" + str(n) + ".png")
            if img.shape != img_shape:
                raise ValueError(f"Image at index {n} has a different shape: {img.shape} compared to {img_shape}")
            images.append(img)

        fps = 10
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
        clip.write_videofile(path + '/lake_evolution_slow.mp4')


#%% ensemble plot

import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

colors = list(mcolors.CSS4_COLORS.keys())
random_colors = np.random.choice(colors, e_nr)
#random_colors = ['red', 'blue', 'green', 'orange', 'purple']

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

if e_nr > 1:

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

elif e_nr == 1:
    axs[0].plot(ensemble['date'], ensemble['water_frac'][0], color = 'black', linewidth=2, label='Water area fraction')
    axs[1].plot(ensemble['date'], ensemble['drained_frac'][0], color = 'black', linewidth=2, label='Drained area fraction')
    axs[0].set_ylabel('Water area fraction')
    axs[0].set_ylim(0, 1)
    axs[1].set_ylabel('Drained area fraction')
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('Year')

plt.tight_layout()
plt.savefig("plots/ensemble_plot.png", dpi = 200,bbox_inches="tight")

#%%
print("Plots created successfully!")
#%%