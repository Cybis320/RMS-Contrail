import sys
sys.path.append('/Users/lucbusquin/Projects/RMS-Contrail')

import RMS.Astrometry.ApplyAstrometry as aa
import RMS.Formats.Platepar as pp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


platepar = pp.Platepar()
platepar.read("/Users/lucbusquin/Projects/RMS_data/ArchivedFiles/US9999_20240203_013216_212627_detected/platepar_cmn2011.cal")

from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Astrometry.Conversions import date2JD, jd2Date
time = getMiddleTimeFF('FF_HR000A_20181215_015724_739_0802560.fits', 25)


margin = 10
x_grid, y_grid = np.linspace(margin, platepar.X_res-margin, 190), np.linspace(margin, platepar.Y_res-margin, 100)
xx, yy = np.meshgrid(x_grid, y_grid)
xx, yy = xx.flatten(), yy.flatten()


# Compute celestial coordinates for each grid point
_, ra_arr, dec_arr, _ = aa.xyToRaDecPP(len(xx)*[platepar.JD], xx, yy, len(xx)*[1], platepar, extinction_correction=False, jd_time=True)

# Convert back to pixel coordinates
x_star, y_star = aa.raDecToXYPP(ra_arr, dec_arr, platepar.JD, platepar)

# Calculate error for each point and populate the errors matrix
errors = np.sqrt((x_star - xx)**2 + (y_star - yy)**2)

errors_reshaped = errors.reshape(len(y_grid), len(x_grid))
xx_reshaped = xx.reshape(len(y_grid), len(x_grid))
yy_reshaped = yy.reshape(len(y_grid), len(x_grid))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a surface plot
surf = ax.plot_surface(xx_reshaped, yy_reshaped, errors_reshaped, cmap='coolwarm',  vmin=0, vmax=.25)


fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_zlim(0, .25)
ax.set_xlim(0, platepar.X_res)
ax.set_ylim(0, platepar.Y_res)
ax.set_title('XY to RaDEC Roundtrip Error\nUsing Different Parameters in Each Directions')
ax.set_xlabel('X Coordinate (px)')
ax.set_ylabel('Y Coordinate(px)')
ax.set_zlabel('Error(px)')

plt.show()