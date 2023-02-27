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
CERES_file = sorted(
    glob.glob("/RAID01/data/Pre_Data_PCA/CERES_og/CERES_*.nc")
)

# concatenate all cld nc files together
ds_CERES = xr.open_mfdataset(CERES_file, combine="by_coords")

# Sort by time
ds_CERES = ds_CERES.sortby("time")

# Select every 28 days of each month
# ！！！！！！！
ds_CERES_28day = ds_CERES.isel(
    time=(ds_CERES.time.dt.day >= 1) & (ds_CERES.time.dt.day <= 28)
)

cldarea = ds_CERES_28day.cldarea_high_daily
cldtau = ds_CERES_28day.cldtau_high_daily
cldtauL = ds_CERES_28day.cldtau_lin_high_daily
cldemissir = ds_CERES_28day.cldemissir_high_daily
iwp = ds_CERES_28day.iwp_high_daily
cldpress_top = ds_CERES_28day.cldpress_top_high_daily
cldtemp_top = ds_CERES_28day.cldtemp_top_high_daily
cldhgth_top = ds_CERES_28day.cldhght_top_high_daily
cldpress_base = ds_CERES_28day.cldpress_base_high_daily
cldtemp_base = ds_CERES_28day.cldtemp_base_high_daily
cldicerad = ds_CERES_28day.cldicerad_high_daily
cldphase = ds_CERES_28day.cldphase_high_daily

cldeff_press = ds_CERES_28day.cldpress_high_daily
cldeff_temp = ds_CERES_28day.cldtemp_high_daily
cldeff_hgth = ds_CERES_28day.cldhght_high_daily


# Save file as netcdf
def save_Cld_data_as_netcdf(
    cldarea,
    cldtau,
    cldtauL,
    cldemissir,
    iwp,
    cldpress_top,
    cldtemp_top,
    cldhgth_top,
    cldpress_base,
    cldtemp_base,
    cldicerad,
    cldphase,
    cldeff_press,
    cldeff_temp,
    cldeff_hgth,
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    cldarea = np.reshape(cldarea.values, (132, 28, 180, 360))
    cldtau = np.reshape(cldtau.values, (132, 28, 180, 360))
    cldtauL = np.reshape(cldtauL.values, (132, 28, 180, 360))
    cldemissir = np.reshape(cldemissir.values, (132, 28, 180, 360))
    cldicerad = np.reshape(cldicerad.values, (132, 28, 180, 360))
    cldphase = np.reshape(cldphase.values, (132, 28, 180, 360))
    cldpress_top = np.reshape(
        cldpress_top.values, (132, 28, 180, 360)
    )
    cldtemp_top = np.reshape(
        cldtemp_top.values, (132, 28, 180, 360)
    )
    cldhgth_top = np.reshape(
        cldhgth_top.values, (132, 28, 180, 360)
    )
    cldpress_base = np.reshape(
        cldpress_base.values, (132, 28, 180, 360)
    )
    cldtemp_base = np.reshape(
        cldtemp_base.values, (132, 28, 180, 360)
    )
    cldeff_press = np.reshape(
        cldeff_press.values, (132, 28, 180, 360)
    )
    cldeff_temp = np.reshape(
        cldeff_temp.values, (132, 28, 180, 360)
    )
    cldeff_hgth = np.reshape(
        cldeff_hgth.values, (132, 28, 180, 360)
    )
    iwp = np.reshape(iwp.values, (132, 28, 180, 360))

    ds_CERES = xr.Dataset(
        {
            "Cldarea": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldarea[:, :, :, :],
            ),
            "Cldtau": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldtau[:, :, :, :],
            ),
            "CldtauL": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldtauL[:, :, :, :],
            ),
            "Cldemissir": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldemissir[:, :, :, :],
            ),
            "Cldicerad": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldicerad[:, :, :, :],
            ),
            "Cldphase": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldphase[:, :, :, :],
            ),
            "Cldpress_top": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldpress_top[:, :, :, :],
            ),
            "Cldtemp_top": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldtemp_top[:, :, :, :],
            ),
            "Cldhgth_top": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldhgth_top[:, :, :, :],
            ),
            "Cldpress_base": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldpress_base[:, :, :, :],
            ),
            "Cldtemp_base": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldtemp_base[:, :, :, :],
            ),
            "Cldeff_press": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldeff_press[:, :, :, :],
            ),
            "Cldeff_temp": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldeff_temp[:, :, :, :],
            ),
            "Cldeff_hgth": (
                ("Month", "Day", "Latitude", "Longitude"),
                cldeff_hgth[:, :, :, :],
            ),
            "IWP": (
                ("Month", "Day", "Latitude", "Longitude"),
                iwp[:, :, :, :],
            ),
        },
        coords={
            "lat": ("Latitude", np.linspace(-90, 89, 180)),
            "lon": ("Longitude", np.linspace(0, 359, 360)),
            "month": ("months", np.linspace(0, 131, 132)),
            "day": ("days", np.linspace(0, 27, 28)),
        },
    )

    os.makedirs("/RAID01/data/Cld_data/", exist_ok=True)
    ds_CERES.to_netcdf(
        "/RAID01/data/Cld_data/2010_2020_CERES_high_cloud_data.nc"
    )


save_Cld_data_as_netcdf(
    cldarea,
    cldtau,
    cldtauL,
    cldemissir,
    iwp,
    cldpress_top,
    cldtemp_top,
    cldhgth_top,
    cldpress_base,
    cldtemp_base,
    cldicerad,
    cldphase,
    cldeff_press,
    cldeff_temp,
    cldeff_hgth,
)
