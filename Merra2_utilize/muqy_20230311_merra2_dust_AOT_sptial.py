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
from ast import List
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from pyparsing import Optional

# extract the data from 2010 to 2014 like above
data_merra2 = xr.open_mfdataset(
    [
        "/RAID01/data/merra2/merra_2_daily_2010.nc",
        "/RAID01/data/merra2/merra_2_daily_2011.nc",
        "/RAID01/data/merra2/merra_2_daily_2012.nc",
        "/RAID01/data/merra2/merra_2_daily_2013.nc",
        "/RAID01/data/merra2/merra_2_daily_2014.nc",
        "/RAID01/data/merra2/merra_2_daily_2015.nc",
        "/RAID01/data/merra2/merra_2_daily_2016.nc",
        "/RAID01/data/merra2/merra_2_daily_2017.nc",
        "/RAID01/data/merra2/merra_2_daily_2018.nc",
        "/RAID01/data/merra2/merra_2_daily_2019.nc",
        "/RAID01/data/merra2/merra_2_daily_2020.nc",
    ]
)


# -------------------------- Data preprocessing --------------------------
# !!!!
# Be aware! We just change time range from 2010-2020 to 2017-2020
# To be the same with the contrail signal extraction period


def lattitude_mean_anormally_generator(
    data_merra2, variable_name, lat_band
):
    """
    Calculate the anomaly (departure from the mean) of a given variable in a given latitude band (20N-50N)
    for each year from 2017 to 2020 using MERRA-2 data. The mean is calculated using data from 2017-2020,
    excluding the current year.

    Parameters
    ----------
    data_merra2 : xr.Dataset
        The MERRA-2 dataset containing the variable of interest.
    variable_name : str
        The name of the variable to calculate the anomaly for.
    lat_band : slice
        The latitude band of interest, specified as a slice object (e.g. slice(20,50)).

    Returns
    -------
    List[np.ndarray]
        A list of 4 NumPy arrays (one for each year from 2017 to 2020) containing the anomaly data
        for the specified latitude band.
    """
    # Calculate the anomaly data for the given latitude band and each year from 2017 to 2020
    lat_band_mean_anormally_list = []

    for year in range(2017, 2021):
        other_years = [y for y in range(2017, 2021) if y != year]

        mean_data = data_merra2.sel(
            time=(
                slice(
                    f"{other_years[0]}-01-01",
                    f"{other_years[-1]}-12-31",
                )
            )
        )[variable_name].mean(dim=["time"])

        anomaly_data = (
            data_merra2.sel(
                time=slice(f"{year}-01-01", f"{year}-12-31")
            )[variable_name].mean(dim=["time"])
            - mean_data
        )

        lat_band_mean_360 = (
            anomaly_data.sel(lat=lat_band).mean(dim=["lat"]).values
        )
        lat_band_mean_anormally_list.append(lat_band_mean_360)

    return lat_band_mean_anormally_list


def lattitude_mean_generator(data_merra2, variable_name, lat_band):
    """
    Calculate the mean value of a given variable in a given latitude band (20N-50N)
    for each year from 2010 to 2021 using MERRA-2 data.

    Parameters
    ----------
    data_merra2 : xr.Dataset
        The MERRA-2 dataset containing the variable of interest.
    variable_name : str
        The name of the variable to calculate the mean for.
    lat_band : slice
        The latitude band of interest, specified as a slice object (e.g. slice(20,50)).

    Returns
    -------
    List[np.ndarray]
        A list of 12 NumPy arrays (one for each year from 2010 to 2021) containing the mean data
        for the specified latitude band.
    """
    mean_data = data_merra2.sel(time=slice("2017", "2021"))[
        variable_name
    ]

    # Calculate the mean data for the given latitude band and each year from 2010 to 2021
    lat_band_mean_list = []

    for year in range(2017, 2021):
        lat_band_mean = mean_data.sel(
            lat=lat_band, time=f"{year}"
        ).mean(dim=["lat", "time"])
        lat_band_mean_360 = lat_band_mean.values
        lat_band_mean_list.append(lat_band_mean_360)

    return lat_band_mean_list


# Anomaly calculations
# Dust AOT
lat_band_mean_anormally_list_dust_AOT = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="DUEXTTAU",
        lat_band=slice(20, 50),
    )
)
lat_band_mean_anormally_list_dust_mass = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="DUCMASS",
        lat_band=slice(20, 50),
    )
)
# BC AOT
lat_band_mean_anormally_list_BC_AOT = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="BCEXTTAU",
        lat_band=slice(20, 50),
    )
)
lat_band_mean_anormally_list_BC_mass = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="BCCMASS",
        lat_band=slice(20, 50),
    )
)
# OC AOT
lat_band_mean_anormally_list_OC_AOT = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="OCEXTTAU",
        lat_band=slice(20, 50),
    )
)
lat_band_mean_anormally_list_OC_mass = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="OCCMASS",
        lat_band=slice(20, 50),
    )
)
# DMS AOT
lat_band_mean_anormally_list_SO4_AOT = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="SUEXTTAU",
        lat_band=slice(20, 50),
    )
)
lat_band_mean_anormally_list_SO4_mass = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="SO4SMASS",
        lat_band=slice(20, 50),
    )
)
# TOT AOT
lat_band_mean_anormally_list_TOT_AOT = (
    lattitude_mean_anormally_generator(
        data_merra2=data_merra2,
        variable_name="TOTEXTTAU",
        lat_band=slice(20, 50),
    )
)

# Mean calculations
# Dust AOT
lat_band_mean_list_dust_AOT = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="DUEXTTAU",
    lat_band=slice(20, 50),
)
lat_band_mean_list_dust_mass = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="DUCMASS",
    lat_band=slice(20, 50),
)
# BC AOT
lat_band_mean_list_BC_AOT = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="BCEXTTAU",
    lat_band=slice(20, 50),
)
lat_band_mean_list_BC_mass = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="BCCMASS",
    lat_band=slice(20, 50),
)
# OC AOT
lat_band_mean_list_OC_AOT = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="OCEXTTAU",
    lat_band=slice(20, 50),
)
lat_band_mean_list_OC_mass = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="OCCMASS",
    lat_band=slice(20, 50),
)
# DMS AOT
lat_band_mean_list_SO4_AOT = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="SUEXTTAU",
    lat_band=slice(20, 50),
)
lat_band_mean_list_SO4_mass = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="SO4SMASS",
    lat_band=slice(20, 50),
)
# TOT AOT
lat_band_mean_list_TOT_AOT = lattitude_mean_generator(
    data_merra2=data_merra2,
    variable_name="TOTEXTTAU",
    lat_band=slice(20, 50),
)

# concatenate the list into an array with shape:
# (aerosol type, year, longitude), the lat_band_mean_anormally_list_dust_AOT is a list with shape(year, longitude)
# anormally
lat_band_mean_anormally_array = np.array(
    [
        lat_band_mean_anormally_list_dust_AOT,
        lat_band_mean_anormally_list_BC_AOT,
        lat_band_mean_anormally_list_OC_AOT,
        lat_band_mean_anormally_list_SO4_AOT,
        lat_band_mean_anormally_list_TOT_AOT,
    ]
)
lat_band_mean_anormally_list_TOT_AOT = np.array(
    lat_band_mean_anormally_list_TOT_AOT
)

# mean
lat_band_mean_array = np.array(
    [
        lat_band_mean_list_dust_AOT,
        lat_band_mean_list_BC_AOT,
        lat_band_mean_list_OC_AOT,
        lat_band_mean_list_SO4_AOT,
    ]
)
lat_band_mean_list_TOT_AOT = np.array(lat_band_mean_list_TOT_AOT)

# aerosol type
aerosol_type = ["Dust", "BC", "OC", "SO4"]

mean_2017_2019 = (
    data_merra2["DUEXTTAU"]
    .sel(time=slice("2017", "2020"))
    .mean(dim="time")
    .values
)
mean_2018_2020 = (
    data_merra2["DUEXTTAU"]
    .sel(time=slice("2018", "2020"))
    .mean(dim="time")
    .values
)
mean_2017_2019_2020 = (
    data_merra2["DUEXTTAU"]
    .sel(time=["2017", "2019", "2020"])
    .mean(dim="time")
    .values
)
mean_2017_2018_2020 = (
    data_merra2["DUEXTTAU"]
    .sel(time=["2017", "2018", "2020"])
    .mean(dim="time")
    .values
)

mean_2017 = (
    data_merra2["DUEXTTAU"]
    .sel(time=("2017"))
    .mean(dim="time")
    .values
)
mean_2018 = (
    data_merra2["DUEXTTAU"]
    .sel(time=("2018"))
    .mean(dim="time")
    .values
)
mean_2019 = (
    data_merra2["DUEXTTAU"]
    .sel(time=("2019"))
    .mean(dim="time")
    .values
)
mean_2020 = (
    data_merra2["DUEXTTAU"]
    .sel(time=("2020"))
    .mean(dim="time")
    .values
)

anormally_2017 = mean_2017 - mean_2018_2020
anormally_2018 = mean_2018 - mean_2017_2019_2020
anormally_2019 = mean_2019 - mean_2017_2018_2020
anormally_2020 = mean_2020 - mean_2017_2019

# ------------------------------ Plotting functions ------------------------------


def plot_spatial_after_interp(
    data,
    var_name,
    vmin,
    vmax,
    title,
):
    mpl.style.use("seaborn-v0_8-ticks")
    mpl.rc("font", family="Times New Roman")
    lon = np.linspace(-180, 179, 360)
    lat = np.linspace(-90, 89, 180)
    lons, lats = np.meshgrid(lon, lat)
    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(11, 7),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})
    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.coastlines(resolution="50m", lw=0.9)
    ax1.set_title(title, fontsize=24)
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.65,
        extend="both",
    )
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}
    plt.show()


plot_spatial_after_interp(
    data=mean_2020 - mean_2017_2019,
    var_name="AOT Anormally",
    vmin=-0.04,
    vmax=0.04,
    title="Dust AOT at 550nm 2020 - (2017-2019)",
)

plot_spatial_after_interp(
    data=mean_2019 - mean_2017_2018_2020,
    var_name="AOT Anormally",
    vmin=-0.04,
    vmax=0.04,
    title="Dust AOT at 550nm 2019 - (2017-2019)",
)

plot_spatial_after_interp(
    data=mean_2018 - mean_2017_2019_2020,
    var_name="AOT Anormally",
    vmin=-0.04,
    vmax=0.04,
    title="Dust AOT at 550nm 2018 - (2017-2019)",
)

plot_spatial_after_interp(
    data=mean_2017 - mean_2018_2020,
    var_name="AOT Anormally",
    vmin=-0.04,
    vmax=0.04,
    title="Dust AOT at 550nm 2017 - (2017-2019)",
)


def plot_each_year_dust_lat_mean(lat_mean_dust_lst):
    """plot the lat_mean_dust_lst, means plot each line for each year,
    the x axis is the longitude, the y axis is the dust AOT at 550nm
    """
    mpl.style.use("seaborn-v0_8-deep")
    mpl.rc("font", family="Times New Roman")

    fig, ax = plt.subplots(figsize=(12, 3))
    plt.rcParams.update({"font.family": "Times New Roman"})

    for i in range(4):
        ax.plot(
            np.linspace(-180, 179, 360),
            lat_mean_dust_lst[i],
            label=f"{2017+i}",
        )

    ax.set_xlabel("Longitude", fontsize=22)
    ax.set_ylabel("Dust AOT at 550nm", fontsize=22)
    ax.tick_params(labelsize=20)

    # 创建新轴并放置图例
    ax.legend(
        loc="center",
        fontsize=14,
        bbox_to_anchor=(1.07, 0.5),
        bbox_transform=ax.transAxes,
    )

    plt.show()


plot_each_year_dust_lat_mean(lat_band_mean_anormally_array[0])


def plot_each_year_aerosol(
    lat_mean_anormally_AOT_array: np.ndarray,
    lat_mean_AOT_array: np.ndarray,
    aerosol_name: list,
    total_AOT_array: np.ndarray,
):
    """
    Plot the AOT anomaly and the proportion of each AOT to the total aerosol AOT for each year.

    Parameters
    ----------
    lat_mean_anormally_AOT_array : np.ndarray
        A 3D NumPy array with the latitude mean AOT anomaly for each aerosol type.
        Shape: (number_of_aerosols, number_of_years, number_of_longitudes)

    lat_mean_AOT_array : np.ndarray
        A 3D NumPy array with the latitude mean AOT for each aerosol type.
        Shape: (number_of_aerosols, number_of_years, number_of_longitudes)

    aerosol_name : list
        A list of aerosol names (strings) corresponding to the aerosol types in lat_mean_AOT_array.

    total_AOT_array : np.ndarray
        A 3D NumPy array with the total AOT for each year.
        Shape: (number_of_years, number_of_latitudes, number_of_longitudes)

    """
    mpl.style.use("seaborn-v0_8-ticks")
    mpl.rc("font", family="Times New Roman")

    # Create the figure and subplots
    fig, axs = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(14, 14),
        tight_layout=True,
        width_ratios=[1.9, 1],
        dpi=300,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    # Loop over each year (2017-2020)
    for i in range(4):
        # Left column: plot the lat_mean anormally for each aerosol type
        for j in range(lat_mean_anormally_AOT_array.shape[0]):
            aerosol_name_add_tot = aerosol_name + ["Total"]
            axs[i, 0].plot(
                np.linspace(-180, 179, 360),
                lat_mean_anormally_AOT_array[j, i],
                label=f"{aerosol_name_add_tot[j]}",
            )

        # Only set x labels and tick labels for the bottom row
        if i == 3:
            axs[i, 0].set_xlabel("Longitude", fontsize=18)

        else:
            axs[i, 0].set_xticklabels([])
            axs[i, 0].set_xticks([])
            axs[i, 0].set_xlabel("")

        axs[i, 0].set_ylabel("AOT", fontsize=18)
        axs[i, 0].tick_params(labelsize=16)
        axs[i, 0].legend()

        # Right column: create a pie chart for the proportion of each AOT to the total aerosol AOT
        # Calculate the total AOT and AOT for each aerosol type
        total_AOT = np.sum(total_AOT_array[i], axis=0)
        each_AOT = np.sum(lat_mean_AOT_array[:, i, :], axis=1)

        # Set the labels, sizes, and explode values for the pie chart
        labels = aerosol_name
        proportions = each_AOT / total_AOT

        # Create the pie chart and set the title
        axs[i, 1].pie(
            proportions,
            labels=labels,
            autopct="%1.1f%%",
            shadow=False,
            startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "k"},
            textprops={"fontsize": 15},
        )
        axs[i, 1].axis(
            "equal"
        )  # Equal aspect ratio ensures that pie is drawn as a circle.
        axs[i, 1].set_title(f"{2017+i}", fontsize=18)

    # Set the overall ylim for the left column
    plt.setp(axs[:, 0], ylim=(-0.052, 0.052))

    # Adjust the layout of the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(
        "AOT_latitude_and_proportion.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )

    plt.show()


plot_each_year_aerosol(
    lat_mean_anormally_AOT_array=lat_band_mean_anormally_array,
    lat_mean_AOT_array=lat_band_mean_array,
    aerosol_name=aerosol_type,
    total_AOT_array=lat_band_mean_list_TOT_AOT,
)
