# test AZ alt functions
from RMS.Astrometry.Conversions import J2000_JD, date2JD, jd2Date, raDec2AltAz, latLonAlt2ECEF, altAz2RADec
import numpy as np

def ECEF2AltAz_ellipsoidal(s_vect, p_vect):
    """ 
    Given two sets of ECEF coordinates, compute alt/az which point from the point S (observer) to the point P (target),
    considering the ellipsoidal shape of the Earth.

    Arguments:
        s_vect: [ndarray] sx, sy, sz - S point (observer) ECEF coordinates
        p_vect: [ndarray] px, py, pz - P point (target) ECEF coordinates

    Return:
        (azim, alt): Horizontal coordinates in degrees.

    """

    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis in meters
    b = 6356752.3142  # semi-minor axis in meters

    sx, sy, sz = s_vect
    px, py, pz = p_vect

    # Compute the ellipsoidal normal for the observer's location
    mag_s = np.sqrt(sx**2 + sy**2 + sz**2)
    nx = a**2 * sx / (mag_s * (a**2 - (a**2 - b**2) * (sz/mag_s)**2))
    ny = a**2 * sy / (mag_s * (a**2 - (a**2 - b**2) * (sz/mag_s)**2))
    nz = b**2 * sz / (mag_s * (b**2 + (a**2 - b**2) * (sz/mag_s)**2))
    n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= n_mag
    ny /= n_mag
    nz /= n_mag

    # Compute the line-of-sight vector from observer to target
    dx = px - sx
    dy = py - sy
    dz = pz - sz
    d_mag = np.sqrt(dx**2 + dy**2 + dz**2)

    # Compute the angle between the ellipsoidal normal and the line-of-sight vector
    cos_zenith = (nx*dx + ny*dy + nz*dz) / d_mag
    zenith = np.arccos(cos_zenith)
    alt = 90 - np.degrees(zenith)

    # Compute the azimuth relative to the ellipsoidal normal
    north_x, north_y, north_z = -sy, sx, 0  # Define the north direction in ECEF
    east_x = -sx * nz
    east_y = -sy * nz
    east_z = sx**2 + sy**2
    east_mag = np.sqrt(east_x**2 + east_y**2 + east_z**2)
    east_x /= east_mag
    east_y /= east_mag
    east_z /= east_mag

    cos_az = (east_x*dx + east_y*dy + east_z*dz) / d_mag
    sin_az = (north_x*dx + north_y*dy + north_z*dz) / d_mag

    azim = np.degrees(np.arctan2(sin_az, cos_az)) % 360

    return azim, alt

# Test with sample ECEF coordinates
s_vect = [1000, 2000, 3000]
p_vect = [4000, 5000, 6000]

ECEF2AltAz_ellipsoidal(s_vect, p_vect)




def ECEF2AltAz(s_vect, p_vect):
    """ Given two sets of ECEF coordinates, compute alt/az which point from the point S to the point P.

    Source: https://gis.stackexchange.com/a/58926
    
    Arguments:
        s_vect: [ndarray] sx, sy, sz - S point ECEF coordiantes
        p_vect: [ndarray] px, py, pz - P point ECEF coordiantes

    Return:
        (azim, alt): Horizontal coordiantes in degrees.

    """


    sx, sy, sz = s_vect
    px, py, pz = p_vect

    # Compute the pointing vector from S to P
    dx = px - sx
    dy = py - sy
    dz = pz - sz

    # Compute the elevation
    alt = np.degrees(
        np.pi/2 - np.arccos((sx*dx + sy*dy + sz*dz)/np.sqrt((sx**2 + sy**2 + sz**2)*(dx**2 + dy**2 + dz**2)))
        )

    # Compute the azimuth
    
    cos_az = (-sz*sx*dx - sz*sy*dy + (sx**2 + sy**2)*dz)/np.sqrt(
                                            (sx**2 + sy**2)*(sx**2 + sy**2 + sz**2)*(dx**2 + dy**2 + dz**2)
                                            )
    
    sin_az = (-sy*dx + sx*dy)/np.sqrt((sx**2 + sy**2)*(dx**2 + dy**2 + dz**2))

    azim = np.degrees(np.arctan2(sin_az, cos_az))%360


    return azim, alt




def GeoHt2xy (s_lat, s_lon, s_h, lat, lon, h):
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
    s_vector = latLonAlt2ECEF(np.radians(s_lat), np.radians(s_lon), s_h)

    azim, alt = ECEF2AltAz_ellipsoidal(s_vector, p_vector)

    alt = refractionTrueToApparent(np.radians(alt))

    ra, dec = altAz2RADec(azim, np.degrees(alt), J2000_JD.days, platepar.lat, platepar.lon)

    x, y = raDecToXYPP(np.array([ra]), np.array([dec]), J2000_JD.days, platepar)

    return x, y

s_lat = 34
s_lon = -112
s_h = 445

lat = 35
lon= -111
h = 1000

print(GeoHt2xy(s_lat, s_lon, s_h, lat, lon, h))

