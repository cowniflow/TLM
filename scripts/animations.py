#%%
'''
FILENAME:
    animations.py

DESCRIPTION:
    This is a script to create animated spatial representations of output from 
    Thermokarst Lake Model (TLM).  

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
import imageio.v2 as imageio
import netCDF4 as nc
import moviepy.video.io.ImageSequenceClip
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

#%% directories

# set working directory
os.chdir(os.path.join( os.path.dirname( __file__ ), '..' ))

#%% Import the necessary parameter

# Check if the correct number of arguments is provided
if len(sys.argv) != 6:
    print(f"Error: Expected 3 arguments, but got {len(sys.argv) - 1}.",
        file=sys.stderr)
    # Exit the script if the number of arguments is incorrect
    sys.exit(1)

# Assign each command-line argument to a variable, converting to the appropriate type
A_cell = float(sys.argv[1])  # Convert A_cell to float
e_nr = int(sys.argv[2])  # Convert e_nr to integer
T = int(sys.argv[3])  # Number of years
print(T)
dt = int(sys.argv[4])  # Convert dt to integer
folder = sys.argv[5]  # Folder name


#%% Initialize dictionary with all ensemble members

ensemble = {
    'date': None,
    'water_frac': [],
    'drained_frac': [],
    'lake_nr': []
}

#%% import lake dataset and plot

for e in range(1,e_nr+1):

    ds = nc.Dataset(folder + 'lakes_' + str(e) + '.nc')

    PATH = "plots/run_" + str(e)
    os.makedirs("plots/run_" + str(e), exist_ok=True)

    # create circle plots
    os.makedirs(PATH + "/circles", exist_ok=True)

    def yr(x, pos):
        """turn x axis value from m to km"""
        return x/1000
    formatter = FuncFormatter(yr)

    plt.rcParams.update({'font.size': 18})
    blue_patch = mpatches.Patch(color='blue', label='water')
    brown_patch = mpatches.Patch(color='brown', label='drained')

    for n in tqdm(range(0,T,dt), desc="Create animations for ensemble run " +
                  str(e),file=sys.stdout):

        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.set_aspect('equal')
        ax.set_xlim(0, np.sqrt(A_cell))
        ax.set_ylim(0, np.sqrt(A_cell))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        for i in range(len(ds['id_geohash'])):
            area_land = ds['area_land'][n, i]
            area_water = ds['area_water_permanent'][n, i]
            if area_land > 0 or area_water > 0:
                xcoord = ds['xcoord'][n, i]
                ycoord = ds['ycoord'][n, i]
                if area_land > 0:
                    d_ring = plt.Circle((xcoord, ycoord), np.sqrt(area_land / np.pi),
                                        facecolor='peru', edgecolor='dimgrey',
                                        linewidth=1, zorder=1)
                    ax.add_patch(d_ring)
                if area_water > 0:
                    l_circle = plt.Circle((xcoord, ycoord), np.sqrt(area_water / np.pi),
                                          color='blue', zorder=2)
                    ax.add_patch(l_circle)

        if n == 0:
            plt.legend(handles=[blue_patch], loc=1).set_zorder(102)
        else:
            plt.legend(handles=[blue_patch, brown_patch], loc=1).set_zorder(102)

        plt.title(f"Lakes at year {n}", fontsize=18)
        plt.xlabel("Distance / km", fontsize=18)
        plt.ylabel("Distance / km", fontsize=18)
        plt.savefig(PATH + f"/circles/lakes_{n}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # create gif of circles
    images = []
    for n in range(0,T,dt):
        img = imageio.imread(PATH + "/circles/lakes_" + str(n) + ".png")
        if n == 0:
            img_shape = img.shape
        elif img.shape != img_shape:
            raise ValueError(f"Image at index {n} has a different shape: \
                             {img.shape} compared to {img_shape}")
        images.append(img)
    imageio.mimsave(PATH + f'/run_{e}.gif', images, duration = 1, loop=1)

    # create video of circles
    images = []
    for n in range(0, T, dt):
        img = imageio.imread(PATH + "/circles/lakes_" + str(n) + ".png")
        if img.shape != img_shape:
            raise ValueError(f"Image at index {n} has a different shape: \
                             {img.shape} compared to {img_shape}")
        images.append(img)

    FPS = 10
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=FPS)
    clip.write_videofile(PATH + f'/run_{e}.mp4')
    