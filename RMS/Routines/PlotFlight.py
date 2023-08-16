""" Functions for computing the pixel coordiantes on the image. """

from __future__ import print_function, division, absolute_import, unicode_literals

# Debug statements
import sys
sys.path.append('x:/Projects/Contrails/RMS-Contrail')

import numpy as np

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Astrometry.Conversions import jd2Date, J2000_JD, latLonAlt2ECEF, ECEF2AltAz, altAz2RADec
from RMS.Formats.Platepar import Platepar
from RMS.Routines.MaskImage import loadMask, MaskStructure

def GeoHt2xy (platepar, lat, lon, h):
    """ Given geo coordiantes of the point and a height above sea level, compute pixel coordiantes on the image.

    Arguments:
        platepar: [Platepar object]
        lat: [float] latitude in radians (+north)
        lon: [float] longitude in radians (+east)
        h: [float] elevation in meters (WGS84)

    Return:
        (x, y): [tuple of floats] Image X coordinate, Image Y coordinate

    """
    # Compute the ECEF location of the point and the station
    p_vector = latLonAlt2ECEF(lat, lon, h)
    print(p_vector)
    s_vector = latLonAlt2ECEF(np.radians(platepar.lat), np.radians(platepar.lon), platepar.elev)
    print(s_vector)

    azim, elev = ECEF2AltAz(s_vector, p_vector)

    ra, dec = altAz2RADec(azim, elev, J2000_JD.days, platepar.lat, platepar.lon)

    x, y = raDecToXYPP(np.array([ra]), np.array([dec]), J2000_JD.days, platepar)

    return x, y

if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the Image X, Y given the platepar and the point lat, lon, ht. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('platepar', metavar='PLATEPAR', type=str, \
                    help="Path to the platepar file.")
    
    arg_parser.add_argument('lat', metavar='LAT', type=float, \
                    help="latitude in degrees (+north).")
    
    arg_parser.add_argument('lon', metavar='LON', type=float, \
                    help="longitude in degrees (+east).")
    
    arg_parser.add_argument('height', metavar='HEIGHT', type=float, \
                    help="elevation in meters (WGS84).")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################
    

    # Load the platepar file
    pp = Platepar()
    pp.read(cml_args.platepar)

    # Convert cml args to radians
    lat_rad = np.radians(cml_args.lat)
    lon_rad = np.radians(cml_args.lon)

    # Compute the FOV geo points
    XY = GeoHt2xy(pp, lat_rad, lon_rad, cml_args.height)

    print(XY)