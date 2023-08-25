""" Make KML files outlining the field of view of the camera.
    Uses Az, Alt, and Ht data from either platepar files or a override.cvs file"""


from __future__ import print_function, division, absolute_import, unicode_literals

import os
import sys

import numpy as np

import RMS.ConfigReader as cr
from RMS.Formats.Platepar import Platepar
from RMS.Routines.MaskImage import loadMask
from RMS.Routines.FOVArea import fovArea


def fovKML(dir_path, platepars, masks=None, area_ht=100000, side_points=10, plot_station=True):
    """ Given all platepar files and a mask files, make KML files outlining the camera FOV.

    Arguments:
        dir_path: [str] Path where the KML file will be saved.
        platepars: [Platepar array object]

    Keyword arguments:
        mask: [Mask object] Mask object, None by default.
        area_ht: [float] Height in meters of the computed area.
        side_points: [int] How many points to use to evaluate the FOV on seach side of the image. Normalized
            to the longest side.
        plot_station: [bool] Plot the location of the station. True by default.

    Return:
        kml_path: [str] Path to the saved KML file.

    """

    for station_code in platepars:

        platepar = platepars[station_code]

        if station_code in masks:
            mask = masks[station_code]
        else:
            mask = None

        print("Computing KML for {:s}".format(platepar.station_code))

        # Find lat/lon/elev describing the view area
        polygon_sides = fovArea(platepar, mask, area_ht, side_points, elev_limit=5)

        # Make longitued in the same wrap region
        lon_list = []
        for side in polygon_sides:
            side = np.array(side)
            lat, lon, elev = side.T
            lon_list += lon.tolist()
        
        # Unwrap longitudes
        lon_list = np.degrees(np.unwrap(2*np.radians(lon_list))/2)

        # Assign longitudes back to the proper sides
        prev_len = 0
        polygon_sides_lon_wrapped = []
        for side in polygon_sides:
            side = np.array(side)
            lat, _, elev = side.T

            # Extract the wrapped longitude
            lon = lon_list[prev_len:prev_len + len(lat)]

            side = np.c_[lat, lon, elev]
            polygon_sides_lon_wrapped.append(side.tolist())

            prev_len += len(lat)

        polygon_sides = polygon_sides_lon_wrapped


        # List of polygons to plot
        polygon_list = []


        # Join points from all sides to create the collection area at the given height
        area_vertices = []
        for side in polygon_sides:
            for side_p in side:
                area_vertices.append(side_p)

        polygon_list.append(area_vertices)


            
        # If the station is plotted, connect every side to the station
        if plot_station:
            
            for side in polygon_sides:

                side_vertices = []

                # Add coordinates of the station (first point)
                side_vertices.append([platepar.lat, platepar.lon, platepar.elev])

                for side_p in side:
                    side_vertices.append(side_p)        

                # Add coordinates of the station (last point)
                side_vertices.append([platepar.lat, platepar.lon, platepar.elev])

                polygon_list.append(list(side_vertices))



        ### MAKE A KML ###

        kml = "<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://earth.google.com/kml/2.1'><Folder><name>{:s}</name><open>1</open><Placemark id='{:s}'>".format(platepar.station_code, platepar.station_code) \
            + """
                    <Style id='camera'>
                    <LineStyle>
                    <width>1.5</width>
                    </LineStyle>
                    <PolyStyle>
                    <color>40000800</color>
                    </PolyStyle>
                    </Style>
                    <styleUrl>#camera</styleUrl>\n""" \
            + "<name>{:s}</name>\n".format(platepar.station_code) \
            + "                <description>Area height: {:d} km\n".format(int(area_ht/1000))

        # Only add station info if the station is plotted
        if plot_station:
            kml += "Longitude: {:10.6f} deg\n".format(platepar.lat) \
                + "Latitude:  {:11.6f} deg\n".format(platepar.lon) \
                + "Altitude: {:.2f} m\n".format(platepar.elev) \

        kml += """
        </description>
        
        <MultiGeometry>"""


        ### Plot all polygons ###
        for polygon_points in polygon_list:
            kml += \
    """    <Polygon>
            <extrude>0</extrude>
            <altitudeMode>absolute</altitudeMode>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>\n"""

            # Add the polygon points to the KML
            for p_lat, p_lon, p_elev in polygon_points:
                kml += "                    {:.6f},{:.6f},{:.0f}\n".format(p_lon, p_lat, p_elev)

            kml += \
    """                </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>"""
        ### ###


        kml += \
    """    </MultiGeometry>
        </Placemark>
        </Folder>
        </kml> """
        ###


        # Save the KML file to the directory with the platepar
        kml_path = os.path.join(dir_path, "{:s}-{:d}km.kml".format(platepar.station_code, int(area_ht/1000)))
        with open(kml_path, 'w') as f:
            f.write(kml)

        print("KML saved to:", kml_path)


    return kml_path



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

    arg_parser.add_argument('-e', '--elev', metavar='ELEVATION', type=float, \
        help="Height of area polygon (km). 100 km by default.", default=100)

    arg_parser.add_argument('-p', '--pts', metavar='SIDE_POINT', type=int, \
        help="Number of points to evaluate on the longest side. 10 by default.", default=50)

    arg_parser.add_argument('-s', '--station', action="store_true", \
        help="""Plot the location of the station.""")

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
                    # Make a copy of the original platepar for this station
                    new_platepar = platepars[station_code].copy()
                    combined_code = f"{row['station_code']}-{row['station_code_override']}"
                    new_platepar.station_code = combined_code
                    new_platepar.az_centre = float(row['az_centre'])
                    new_platepar.alt_centre = float(row['alt_centre'])
                    new_platepar.rotation_from_horiz = float(row['rotation_from_horiz'])
                    new_platepar.lat = float(row['lat'])
                    new_platepar.lon = float(row['lon'])
                    new_platepar.elev = float(row['elev'])

                    # Add the new platepar to the override dictionary with the combined code as key
                    platepars_override[combined_code] = new_platepar
                    print(f"Added overridden values for {station_code}: {combined_code}")

        # Replace the original platepars with the overridden ones
        platepars = platepars_override

    # Generate a KML file from the platepar
    fovKML(cml_args.dir_path, platepars, masks=masks, area_ht=1000*cml_args.elev, side_points=cml_args.pts, \
        plot_station=cml_args.station)
