""" Make a png files plotting the field of view of the camera.
    Uses Az, and Alt from either platepar files or a override.cvs file"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from RMS.Routines.FOVArea2 import fovArea2

def plotFOVSkyMap(platepars, out_dir, north_up=False,  masks=None):
    """ Plot all given platepar files on an Alt/Az sky map. 
    

    Arguments:
        platepars: [dict] A dictionary of Platepar objects where keys are station codes.
        out_dir: [str] Path to where the graph will be saved.

    Keyword arguments:
        masks: [dict] A dictionary of mask objects where keys are station codes.

    """

    # Change plotting style
    plt.style.use('ggplot')

    # Init an alt/az polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    if north_up:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    # Set up elevation limits
    ax.set_rlim(bottom=90, top=0)

    for station_code in platepars:

        pp = platepars[station_code]

        if station_code in masks:
            mask = masks[station_code]
        else:
            mask = None

        print("Computing FOV for {:s}".format(pp.station_code))

        # Compute the edges of the
        side_points_AzAlt = fovArea2(pp, mask=mask, side_points=50, elev_limit=0)
        azims = []
        alts = []
        for side in side_points_AzAlt:

            for azim, alt in side:
                azims.append(azim)
                alts.append(alt)


        # If the polygon is not closed, close it
        if (azims[0] != azims[-1]) or (alts[0] != alts[-1]):
            azims.append(azims[0])
            alts.append(alts[0])


        # Plot the FOV alt/az
        line_handle, = ax.plot(np.radians(azims), alts, alpha=0.75)

        # Fill the FOV
        ax.fill(np.radians(azims), alts, color='0.5', alpha=0.3)


        # Plot the station name at the middle of the FOV
        ax.text(np.radians(pp.az_centre), pp.alt_centre, pp.station_code, va='center', ha='center', 
            color=line_handle.get_color(), weight='bold', size=8)


    ax.grid(True, color='0.98')
    ax.set_xlabel("Azimuth (deg)")
    ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
    ax.tick_params(axis='y', which='major', labelsize=8, direction='out')

    plt.tight_layout()


    # Save the plot to disk
    plot_file_name = "fov_sky_map2.png"
    plot_path = os.path.join(out_dir, plot_file_name)
    plt.savefig(plot_path, dpi=150)
    print("FOV sky map saved to: {:s}".format(plot_path))




if __name__ == "__main__":

    from RMS.ConfigReader import Config
    from RMS.Formats.Platepar import Platepar
    from RMS.Routines.MaskImage import loadMask
    import csv

    import os

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the FOV area given the platepars and mask files. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
                    help="Path to the directory with platepar files. All platepar files will be found recursively.")

    arg_parser.add_argument('-n', '--northup', dest='north_up', default=False, action="store_true", \
                    help="Plot the chart with north up, azimuth increasing clockwise.")



    ###

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Init the default config file
    config = Config()

    # Define the override file to look for
    override_file_name = "override.csv"
    override_file_path = os.path.join(cml_args.dir_path, override_file_name)

    # Initializing array
    platepars = {}
    masks = {}

    # Find all platepar files
    for entry in os.walk(cml_args.dir_path):

        dir_path, _, file_list = entry

        # Add platepar to the list if found
        if config.platepar_name in file_list:

            pp_path = os.path.join(dir_path, config.platepar_name)

            # Load the platepar file
            pp = Platepar()
            pp.read(pp_path)

            # If the station code already exists, skip it
            if pp.station_code in platepars:
                print("Skipping already added station: {:s}".format(pp_path))
                continue

            print()
            print("Loaded platepar for {:s}: {:s}".format(pp.station_code, pp_path))

            print(pp.station_code)
            platepars[pp.station_code] = pp
            # platepars[pp.station_code].az_centre = 10.0


            # Also add a mask if it's available
            if config.mask_file in file_list:
                masks[pp.station_code] = loadMask(os.path.join(dir_path, config.mask_file))
                print("Loaded the mask too!")

    # Look for override.csv
    platepars_override = {}

    if os.path.exists(override_file_path):
        print(f"Override file {override_file_name} found - OVERRIDING platepars!")

        with open(override_file_path, 'r') as file:
            reader = csv.DictReader(file, delimiter=',')

            for row in reader:
                station_code = row['station_code']

                if station_code in platepars:
                    # Validate the data
                    az_centre = float(row['az_centre'])
                    alt_centre = float(row['alt_centre'])
                    rotation_from_horiz = float(row['rotation_from_horiz'])

                    if 0 <= az_centre <= 360 and 0 <= alt_centre <= 90 and -360 <= rotation_from_horiz <= 360:

                        # Make a copy of the original platepar for this station
                        new_platepar = platepars[station_code].copy()
                        combined_code = f"{row['station_code']}-{row['station_code_override']}"
                        new_platepar.station_code = combined_code
                        new_platepar.az_centre = az_centre
                        new_platepar.alt_centre = alt_centre
                        new_platepar.rotation_from_horiz = rotation_from_horiz

                        # Add the new platepar to the override dictionary with the combined code as key
                        platepars_override[combined_code] = new_platepar
                        print(f"Added overridden values for {station_code}: {combined_code}")
                    
                    else:
                        print(f"Invalid data in {override_file_name}")
                        print(f"Station {combined_code}: az_centre: {az_centre}, alt_centre: {alt_centre}, rotation_from_horiz: {rotation_from_horiz}")
                        print('Must be 0 <= az_centre <= 360 and 0 <= alt_centre <= 90 and -360 <= rotation_from_horiz <= 360')
                        exit("Exiting due to invalid data.")

        # Replace the original platepars with the overridden ones
        platepars = platepars_override

    # Plot all plateaprs on an alt/az sky map
    plotFOVSkyMap(platepars, cml_args.dir_path, cml_args.north_up, masks=masks)


    
