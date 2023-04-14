#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#                       神兽保佑
#                      代码无BUG!

"""

    Code to generate ceres cloud data for further analysis.
        
    Owner: Mu Qingyu
    version 1.0
    
    Created: 2023-02-19
    
    Including the following parts:

        1) Read in basic Cirrus data (include cirrus morphology and microphysics)
                
        2) Generate CERES cloud data for further analysis.
        
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

# Read CERES files using xarray
CERES_file = sorted(glob.glob("/RAID01/data/CERES_og/CERES_*.nc"))

# concatenate all cld nc files together
ds_CERES = xr.open_mfdataset(CERES_file, combine="by_coords")

# Sort by time
ds_CERES = ds_CERES.sortby("time")

# Select every 28 days of each month
ds_CERES_28day = ds_CERES.isel(time=(ds_CERES.time.dt.day >= 1) & (ds_CERES.time.dt.day <= 28))

variables = [
    "cldarea_high_1h",
    "cldtau_high_1h",
    "cldtau_lin_high_1h",
    "cldemissir_high_1h",
    "iwp_high_1h",
    "cldpress_top_high_1h",
    "cldtemp_top_high_1h",
    "cldhght_top_high_1h",
    "cldpress_base_high_1h",
    "cldtemp_base_high_1h",
    "cldicerad_high_1h",
    "cldpress_high_1h",
    "cldtemp_high_1h",
    "cldhght_high_1h",
]

# Extract and reshape all variables into 4D arrays
data_arrays = {
    "data_" + var: np.reshape(ds_CERES_28day[var].values, (132, 28 * 24, 180, 360)) for var in variables
}

def save_Cld_data_as_netcdf(data_arrays):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    data_vars = {
        var: (("Month", "Hour", "Latitude", "Longitude"), data_arrays[var])
        for var in data_arrays
    }

    ds_CERES = xr.Dataset(
        data_vars,
        coords={
            "lat": ("Latitude", lat),
            "lon": ("Longitude", lon),
            "month": ("Month", np.linspace(0, 131, 132)),
            "hour": ("Hour", np.linspace(0, 28 * 24 - 1, 28 * 24)),
        },
    )

    ds_CERES.to_netcdf(
        "/RAID01/data/Cld_data/2010_2020_CERES_high_cloud_data_hourly.nc"
    )

save_Cld_data_as_netcdf(data_arrays)