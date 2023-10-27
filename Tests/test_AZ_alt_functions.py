# debug code
import sys
sys.path.append('/Users/lucbusquin/Projects/RMS-Contrail')
# end of debug code

from RMS.Astrometry.Conversions import J2000_JD, date2JD, jd2Date, raDec2AltAz, latLonAlt2ECEF, altAz2RADec
from RMS.Astrometry.CheckFit import rotationWrtHorizon
from RMS.Formats.Platepar import Platepar
import numpy as np
# Import Cython functions
import pyximport
#import RMS.Formats.Platepar
import scipy.optimize
from RMS.Astrometry.AtmosphericExtinction import \
    atmosphericExtinctionCorrection
from RMS.Astrometry.Conversions import J2000_JD, date2JD, jd2Date, raDec2AltAz, latLonAlt2ECEF, ECEF2AltAz, altAz2RADec 
from RMS.Astrometry.ApplyAstrometry import azAltToXYPP, rotationWrtStandard
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import refractionTrueToApparent

def GeoHt2xy_enu (platepar, lat, lon, h):
    """ Given geo coordinates of the point and a height above sea level, compute pixel coordinates on the image.

    Arguments:
        platepar: [Platepar object]
        lat: [float] latitude in degrees (+north)
        lon: [float] longitude in degrees (+east)
        h: [float] elevation in meters (WGS84)

    Return:
        (x, y): [tuple of floats] Image X coordinate, Image Y coordinate

    """
    # Compute the ECEF location of the point and the station
    p_vector = latLonAlt2ECEF(np.radians(lat), np.radians(lon), h)
    s_vector = latLonAlt2ECEF(np.radians(platepar.lat), np.radians(platepar.lon), platepar.elev)

    # Compute the ENU coordinates of the point
    enu_vector = ECEF2ENU(s_vector, p_vector, platepar.lat, platepar.lon)
    
    azim, alt = ENU2AltAz(enu_vector)

    x, y = azAltToXYPP(np.array([azim]), np.array([alt]), platepar)

    return x, y



pp = Platepar()
pp.read('/Users/lucbusquin/Projects/US9999_20230917_021948_930969/platepar_cmn2010.cal')

print(rotationWrtStandard(pp))


# s_lat = 33
# s_lon = -112
# s_h = 445

# lat = 35
# lon= -111
# h = 10000

# print(GeoHt2xy_enu(pp, lat, lon, h))
