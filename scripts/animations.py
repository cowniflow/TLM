# Plot timeseries

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

        plt.rcParams.update({'font.size': 18})
        blue_patch = mpatches.Patch(color='blue', label='water')
        brown_patch = mpatches.Patch(color='brown', label='drained')

        for n in tqdm(range(0,N,dt), desc="Create animations for ensemble run " + str(e),file=sys.stdout):

            fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
            ax.set_aspect('equal')
            ax.set_xlim(0, np.sqrt(A_cell))
            ax.set_ylim(0, np.sqrt(A_cell))
            #ax.xaxis.set_major_formatter(formatter)
            #ax.yaxis.set_major_formatter(formatter)

            for i in range(len(ds['id_geohash'])):
                area_land = ds['area_land'][n, i]
                area_water = ds['area_water_permanent'][n, i]
                if area_land > 0 or area_water > 0:
                    xcoord = ds['xcoord'][n, i]
                    ycoord = ds['ycoord'][n, i]
                    if area_land > 0:
                        d_ring = plt.Circle((xcoord, ycoord), np.sqrt(area_land / np.pi), color='brown', zorder=1)
                        ax.add_patch(d_ring)
                    if area_water > 0:
                        l_circle = plt.Circle((xcoord, ycoord), np.sqrt(area_water / np.pi), color='blue', zorder=2)
                        ax.add_patch(l_circle)

            if n == 0:
                plt.legend(handles=[blue_patch], loc=1).set_zorder(102)
            else:
                plt.legend(handles=[blue_patch, brown_patch], loc=1).set_zorder(102)

            plt.title(f"Lakes at year {n}", fontsize=18)
            plt.xlabel("Distance / km", fontsize=18)
            plt.ylabel("Distance / km", fontsize=18)
            plt.savefig(path + f"/lakes_{n}.png", dpi=200, bbox_inches="tight")
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
            imageio.mimsave(path + f'/run_{n}.gif', images, duration = 1, loop=1)

            # create video of circles
            images = []
            for n in range(0, N, dt):
                img = imageio.imread(path + "/circles/lakes_" + str(n) + ".png")
                if img.shape != img_shape:
                    raise ValueError(f"Image at index {n} has a different shape: {img.shape} compared to {img_shape}")
                images.append(img)

            fps = 10
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
            clip.write_videofile(paht + f'/run_{n}.mp4')