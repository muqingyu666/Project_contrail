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

    Function for utilizing MERRA2 data
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2023-03-11
    
    Including the following parts:

        1) Read in MERRA2 data
        
        2) 

"""

import glob
import os
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr


def extract_merra2_data_and_interp(start_date, end_date):
    """
    Extract MERRA-2 data from the original dataset and interpolate the data to 1-degree
    resolution. Then, compute the daily mean of the data, and concatenate all daily mean
    data into one DataArray. Finally, store the data into one nc file.

    :return: None
    """

    # Define the start and end date of the data
    start_date = start_date
    end_date = end_date

    # Generate a range of dates with a frequency of 1 month start
    month_starts = pd.date_range(
        start=start_date, end=end_date, freq="MS"
    )

    # Initialize an empty list to store the resulting dates
    result_dates = []

    # Loop through the month starts and generate a range of dates with a frequency of 1 day for 28 days
    for start in month_starts:
        month_end = start + pd.DateOffset(days=27)
        month_dates = pd.date_range(
            start=start, end=month_end, freq="D"
        )
        result_dates.extend(month_dates)

    # Create a pandas DatetimeIndex from the resulting dates
    date_index = pd.DatetimeIndex(result_dates)

    # Loop through each year
    for year in range(2011, 2021):
        # Define the data directory for each year
        year_dir = f"/RAID01/data/merra2/{year}"

        # Define an empty list to store daily mean data of each year
        daily_mean_data_list = []

        # extract date index for each year
        dates = date_index[date_index.to_series().dt.year == year]

        # Loop through each date in the date range
        for date in dates:
            if year == 2010:
                file_name = f"MERRA2_300.tavg1_2d_aer_Nx.{date.strftime('%Y%m%d')}.nc4"
            else:
                if date.year == 2020 and date.month == 9:
                    file_name = f"MERRA2_401.tavg1_2d_aer_Nx.{date.strftime('%Y%m%d')}.nc4"
                else:
                    file_name = f"MERRA2_400.tavg1_2d_aer_Nx.{date.strftime('%Y%m%d')}.nc4"

            # Define the file path
            file_path = os.path.join(year_dir, file_name)
            # Check if the file exists
            if os.path.isfile(file_path):
                # process the file as needed
                # Open the file
                data = xr.open_dataset(file_path)

                # Extract the time coordinate
                time_coord = data["time"]
                # do something with the data
                # Define the new latitude and longitude range
                new_lat = xr.DataArray(
                    data=np.arange(-90, 90, 1),
                    dims=("lat",),
                    coords={"lat": np.arange(-90, 90, 1)},
                )
                new_lon = xr.DataArray(
                    data=np.arange(-180, 180, 1),
                    dims=("lon",),
                    coords={"lon": np.arange(-180, 180, 1)},
                )

                # Interpolate the data
                interp_data = data.interp(
                    lat=new_lat, lon=new_lon, method="cubic"
                )

                # Assign the original time coordinate to the new dataset
                interp_data["time"] = time_coord
                # Compute the daily mean
                daily_mean_data = interp_data.mean(
                    dim="time", skipna=True
                )

                # Append the daily mean data to the list
                daily_mean_data_list.append(daily_mean_data)

        # Concatenate all daily mean data into one DataArray
        if len(daily_mean_data_list) > 0:
            daily_mean_data_all = xr.concat(
                daily_mean_data_list, dim="time"
            )

            # Assign the original time coordinate to the new dataset
            daily_mean_data_all["time"] = dates
            # Store the data into one nc file
            output_file_path = (
                f"/RAID01/data/merra2/merra_2_daily_{year}.nc"
            )
            daily_mean_data_all.to_netcdf(output_file_path)


extract_merra2_data_and_interp(
    start_date="2010-01-01", end_date="2020-12-31"
)
