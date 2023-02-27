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

    Useful functions for PCA-HCF analyze, including the 
    calculation of CV
    
    Owner: Mu Qingyu
    version 1.0
        version 2.0 : Filter_data_fit_PC1_gap_plot now must pass a Cld_Data object
            to automatically get the lat and lon information
    Created: 2022-06-28
    
    Including the following parts:
        
        1) Read the Pre-calculated PC1 and HCF data 
        
        2) Filter the data to fit the PC1 gap like -1.5 ~ 3.5
        
        3) Plot boxplot to show the distribution of HCF data
        
        4) Compare HCF of different years in the same PC1 condition
                
        5) All sort of test code
        
"""

import glob

# ------------ PCA analysis ------------
# ------------ Start import ------------
import os
import warnings
from statistics import mean

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

# from numba import jit
import pandas as pd
import seaborn as sns
import shapely.geometry as sgeom
import xarray as xr
from matplotlib import rcParams
from metpy.units import units

# ----------  importing dcmap from my util ----------#
from muqy_20220413_util_useful_functions import dcmap as dcmap
from scipy import stats
from scipy.stats import norm, zscore
from sklearn.decomposition import PCA

# ----------  done importing  ----------#


# ---------- Read PCA&CLD data from netcdf file --------
def read_PC1_CERES_from_netcdf(PC_path, CERES_Cld_dataset_num):
    """
    Read the PC1 and CERES data from the netcdf file
    you can choose the PC1 and CERES data you want to read
    PC1 data :
    [0] : 4-para PC1
    [1] : 5-para PC1

    Cld data :
    ["Cldarea",
        "Cldicerad",
        "Cldtau",
        "Cldtau_lin",
        "IWP",
        "Cldemissir"]
    using dataset_num

    Args:
        PC_para_num (int): the number of PC1 parameter
        0 : 4para PC1
        1 : 5para PC1
        2 : 4para PC1 each grid point

        CERES_Cld_dataset_num (int): 0-5, specify the CERES dataset
        0 - Cldarea
        1 - Cldicerad
        2 - Cldtau
        3 - Cldtau_lin
        4 - IWP
        5 - Cldemissir

    Returns:
        Specified CERES dataset and PC1 data
    """
    CERES_Cld_dataset = [
        "Cldarea",
        "Cldicerad",
        "Cldtau",
        "Cldtau_lin",
        "IWP",
        "Cldemissir",
    ]
    # Read data from netcdf file
    print("Reading data from netcdf file...")
    data0 = xr.open_dataset(
        "/RAID01/data/All_data/2010_2020_PC1_and_CLD.nc"
    )
    data1 = xr.open_dataset(PC_path)

    print("Done loading netcdf file.")

    PC_all = np.array(data1.PC1)

    # Arrange data from all years
    PC_all = PC_all.reshape(239500800)
    PC_all = PC_all.reshape(132, 28, 180, 360)

    # --------- Original data ---------#
    # region
    PC_2010_2019 = PC_all[0:120, :, :, :]  # 2010-2019
    PC_2017_2020 = PC_all[84:132, :, :, :]  # 2017-2020
    PC_2010 = PC_all[0:12, :, :, :]  # 2010
    PC_2011 = PC_all[12:24, :, :, :]  # 2011
    PC_2012 = PC_all[24:36, :, :, :]  # 2012
    PC_2013 = PC_all[36:48, :, :, :]  # 2013
    PC_2014 = PC_all[48:60, :, :, :]  # 2014
    PC_2015 = PC_all[60:72, :, :, :]  # 2015
    PC_2016 = PC_all[72:84, :, :, :]  # 2016
    PC_2017 = PC_all[84:96, :, :, :]  # 2017
    PC_2018 = PC_all[96:108, :, :, :]  # 2018
    PC_2019 = PC_all[108:120, :, :, :]  # 2019
    PC_2020 = PC_all[120:132, :, :, :]  # 2020
    # endregion

    # --------- Get 4-6 month data of each years ---------#
    PC_2010_2019_4_month = PC_2010_2019[3::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    PC_2010_2019_5_month = PC_2010_2019[4::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    PC_2010_2019_6_month = PC_2010_2019[5::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    # put them together
    PC_2010_2019_4_6_month = np.concatenate(
        [
            PC_2010_2019_4_month,
            PC_2010_2019_5_month,
            PC_2010_2019_6_month,
        ],
        axis=0,
    ).reshape(-1, 180, 360)
    PC_2017_2019_4_6_month = PC_2010_2019_4_6_month[588:, :, :]
    # --------- Get 4-6 month data of each years ---------#
    PC_2020_4_6_month = PC_all[123:126, :, :, :].reshape(
        -1, 180, 360
    )
    # PC_2019_4_6_month = PC_2010_2019_4_6_month[756:, :, :]
    # PC_2018_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]
    # PC_2017_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]
    # PC_2018_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]
    # PC_2018_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]
    # PC_2018_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]

    # region
    PC_all = PC_all.reshape(3696, 180, 360)
    PC_2010_2019 = PC_2010_2019.reshape(3360, 180, 360)
    PC_2010 = PC_2010.reshape(336, 180, 360)
    PC_2011 = PC_2011.reshape(336, 180, 360)
    PC_2012 = PC_2012.reshape(336, 180, 360)
    PC_2013 = PC_2013.reshape(336, 180, 360)
    PC_2014 = PC_2014.reshape(336, 180, 360)
    PC_2015 = PC_2015.reshape(336, 180, 360)
    PC_2016 = PC_2016.reshape(336, 180, 360)
    PC_2017 = PC_2017.reshape(336, 180, 360)
    PC_2018 = PC_2018.reshape(336, 180, 360)
    PC_2019 = PC_2019.reshape(336, 180, 360)
    PC_2020 = PC_2020.reshape(336, 180, 360)
    # endregion

    Cld_data = np.array(
        data0[CERES_Cld_dataset[CERES_Cld_dataset_num]]
    )

    # --------- Original data ---------#
    # region
    # Cld_all = Cldarea_data.reshape(
    #     132, 28, 180, 360
    # )  # Choose the variable used in the plot
    Cld_all = Cld_data.reshape(
        132, 28, 180, 360
    )  # Choose the variable used in the plot
    Cld_all[Cld_all == -999] = np.nan

    Cld_2010_2019 = Cld_all[0:120, :, :, :]  # 2010-2019
    Cld_2018_2020 = Cld_all[84:132, :, :, :]  # 2018-2020
    Cld_2010 = Cld_all[0:12, :, :, :]  # 2010
    Cld_2011 = Cld_all[12:24, :, :, :]  # 2011
    Cld_2012 = Cld_all[24:36, :, :, :]  # 2012
    Cld_2013 = Cld_all[36:48, :, :, :]  # 2013
    Cld_2014 = Cld_all[48:60, :, :, :]  # 2014
    Cld_2015 = Cld_all[60:72, :, :, :]  # 2015
    Cld_2016 = Cld_all[72:84, :, :, :]  # 2016
    Cld_2017 = Cld_all[84:96, :, :, :]  # 2017
    Cld_2018 = Cld_all[96:108, :, :, :]  # 2018
    Cld_2019 = Cld_all[108:120, :, :, :]  # 2019
    Cld_2020 = Cld_all[120:132, :, :, :]  # 2020
    # endregion

    # --------- Get 4-6 month data of each years ---------#
    Cld_2010_2019_4_month = Cld_2010_2019[3::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    Cld_2010_2019_5_month = Cld_2010_2019[4::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    Cld_2010_2019_6_month = Cld_2010_2019[5::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    # put them together
    Cld_2010_2019_4_6_month = np.concatenate(
        [
            Cld_2010_2019_4_month,
            Cld_2010_2019_5_month,
            Cld_2010_2019_6_month,
        ],
        axis=0,
    ).reshape(-1, 180, 360)

    # region
    Cld_all = Cld_all.reshape(3696, 180, 360)
    Cld_2010_2019 = Cld_2010_2019.reshape(3360, 180, 360)
    Cld_2010 = Cld_2010.reshape(336, 180, 360)
    Cld_2011 = Cld_2011.reshape(336, 180, 360)
    Cld_2012 = Cld_2012.reshape(336, 180, 360)
    Cld_2013 = Cld_2013.reshape(336, 180, 360)
    Cld_2014 = Cld_2014.reshape(336, 180, 360)
    Cld_2015 = Cld_2015.reshape(336, 180, 360)
    Cld_2016 = Cld_2016.reshape(336, 180, 360)
    Cld_2017 = Cld_2017.reshape(336, 180, 360)
    Cld_2018 = Cld_2018.reshape(336, 180, 360)
    Cld_2019 = Cld_2019.reshape(336, 180, 360)
    Cld_2020 = Cld_2020.reshape(336, 180, 360)
    # endregion

    return (
        PC_all,
        # PC_2010_2019_4_6_month,
        # PC_2017_2019_4_6_month,
        # PC_2020_4_6_month,
        # PC_2010_2019,
        PC_2010,
        PC_2011,
        PC_2012,
        PC_2013,
        PC_2014,
        PC_2015,
        PC_2016,
        PC_2017,
        PC_2018,
        PC_2019,
        PC_2020,
        Cld_all,
        # Cld_2010_2019_4_6_month,
        # Cld_2017_2019_4_6_month,
        # Cld_2020_4_6_month,
        # Cld_2010_2019,
        # Cld_2018_2020,
        Cld_2010,
        Cld_2011,
        Cld_2012,
        Cld_2013,
        Cld_2014,
        Cld_2015,
        Cld_2016,
        Cld_2017,
        Cld_2018,
        Cld_2019,
        Cld_2020,
    )


# ---------- Read atmos para from netcdf ----------#


def read_atmos_from_netcdf(atmos_path):
    atmos_data = xr.open_dataset(atmos_path)

    RelativeH_300 = atmos_data["RelativeH_300"].values
    Temperature_300 = atmos_data["Temperature_300"].values
    Wvelocity_300 = atmos_data["Wvelocity_300"].values
    Stability_300 = atmos_data["Stability_300"].values
    Uwind_300 = atmos_data["Uwind_300"].values

    return (
        RelativeH_300,
        Temperature_300,
        Wvelocity_300,
        Stability_300,
        Uwind_300,
    )


# ---------- Only read PC1 data ----------#
# compare 4 parameters pc1 and 5 parameters pc1
def read_PC1_only_from_netcdf(
    pc_data_name="2010_2020_5_parameters_300hPa_PC1.nc",
):
    """
    Read the PC1 data from the netcdf file

    Parameters:
        pc_data_name: the name of the netcdf file
        "2010_2020_5_parameters_300hPa_PC1.nc" by default
        can be "2010_2020_4_parameters_300hPa_PC1.nc"
        as 4 parameters pc1
    Returns:
        Specified PC1 data
    """

    # Read data from netcdf file
    print("Reading data from netcdf file...")
    data0 = xr.open_dataset(
        "/RAID01/data/All_data/2010_2020_PC1_and_CLD.nc"
    )
    data1 = xr.open_dataset("/RAID01/data/PCA_data/" + pc_data_name)

    print("Done loading netcdf file.")

    PC_all = np.array(data1.PC1)

    # Arrange data from all years
    PC_all = PC_all.reshape(239500800)
    PC_all = PC_all.reshape(132, 28, 180, 360)

    # --------- Original data ---------#
    # region
    PC_2010_2019 = PC_all[0:120, :, :, :]  # 2010-2019
    PC_2017_2020 = PC_all[84:132, :, :, :]  # 2017-2020
    PC_2010 = PC_all[0:12, :, :, :]  # 2010
    PC_2011 = PC_all[12:24, :, :, :]  # 2011
    PC_2012 = PC_all[24:36, :, :, :]  # 2012
    PC_2013 = PC_all[36:48, :, :, :]  # 2013
    PC_2014 = PC_all[48:60, :, :, :]  # 2014
    PC_2015 = PC_all[60:72, :, :, :]  # 2015
    PC_2016 = PC_all[72:84, :, :, :]  # 2016
    PC_2017 = PC_all[84:96, :, :, :]  # 2017
    PC_2018 = PC_all[96:108, :, :, :]  # 2018
    PC_2019 = PC_all[108:120, :, :, :]  # 2019
    PC_2020 = PC_all[120:132, :, :, :]  # 2020
    # endregion

    # region
    PC_all = PC_all.reshape(3696, 180, 360)
    PC_2010_2019 = PC_2010_2019.reshape(3360, 180, 360)
    PC_2010 = PC_2010.reshape(336, 180, 360)
    PC_2011 = PC_2011.reshape(336, 180, 360)
    PC_2012 = PC_2012.reshape(336, 180, 360)
    PC_2013 = PC_2013.reshape(336, 180, 360)
    PC_2014 = PC_2014.reshape(336, 180, 360)
    PC_2015 = PC_2015.reshape(336, 180, 360)
    PC_2016 = PC_2016.reshape(336, 180, 360)
    PC_2017 = PC_2017.reshape(336, 180, 360)
    PC_2018 = PC_2018.reshape(336, 180, 360)
    PC_2019 = PC_2019.reshape(336, 180, 360)
    PC_2020 = PC_2020.reshape(336, 180, 360)
    # endregion

    return (
        PC_all,
        PC_2010,
        PC_2011,
        PC_2012,
        PC_2013,
        PC_2014,
        PC_2015,
        PC_2016,
        PC_2017,
        PC_2018,
        PC_2019,
        PC_2020,
    )


# Use the font and apply the matplotlib style
mpl.rc("font", family="Times New Roman")
mpl.style.use("seaborn-v0_8-ticks")
# Reuse the same font to ensure that font set properly
# plt.rc("font", family="Times New Roman")

#########################################################
######### moving average function #######################
#########################################################


def np_move_avg(a, n, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode)


#########################################################
###### simple plot func ##################################
#########################################################


def plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    Cld_match_PC_gap,
    # p_value,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    lons, lats = np.meshgrid(lon, lat)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(11, 4),
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
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)

    # dot the significant area
    # dot_area = np.where(p_value < 0.00000005)
    # dot = ax1.scatter(
    #     lons[dot_area],
    #     lats[dot_area],
    #     color="k",
    #     s=3,
    #     linewidths=0,
    #     transform=ccrs.PlateCarree(),
    # )

    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


def plot_Cld_no_mean_simple_partial_self_cmap(
    Cld_match_PC_gap,
    # p_value,
    cld_min,
    cld_max,
    cld_name,
    lon=np.linspace(0, 359, 360),
    lat=np.linspace(-90, 89, 180),
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    lons, lats = np.meshgrid(lon, lat)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(11, 4),
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
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)

    # dot the significant area
    # dot_area = np.where(p_value < 0.00000005)
    # dot = ax1.scatter(
    #     lons[dot_area],
    #     lats[dot_area],
    #     color="k",
    #     s=3,
    #     linewidths=0,
    #     transform=ccrs.PlateCarree(),
    # )

    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


def plot_Cld_no_mean_simple_tropical_self_cmap(
    Cld_match_PC_gap,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(11, 4),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
    ax1.set_facecolor("silver")
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


def plot_Cld_no_mean_simple_north_polar_self_cmap(
    Cld_match_PC_gap,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    from cartopy.mpl.ticker import (
        LatitudeFormatter,
        LongitudeFormatter,
    )

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(14, 11),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.NorthPolarStereo(),
    )
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())
    ax1.coastlines(resolution="50m", lw=0.9)
    # gl = ax1.gridlines(
    #     linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    # )
    gl = ax1.gridlines(
        linestyle="--",
        ylocs=np.arange(65, 90, 5),
        xlocs=np.arange(-180, 180, 30),
        draw_labels=True,
    )
    gl.xlabels_top = True
    # gl.ylabels_right = True

    gl.xformatter = LongitudeFormatter()

    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1.set_boundary(circle, transform=ax1.transAxes)


def plot_Cld_no_mean_simple_sourth_polar_self_cmap(
    Cld_match_PC_gap,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    from cartopy.mpl.ticker import (
        LatitudeFormatter,
        LongitudeFormatter,
    )

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(14, 11),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.SouthPolarStereo(),
    )
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
    ax1.coastlines(resolution="50m", lw=0.9)
    # gl = ax1.gridlines(
    #     linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    # )
    gl = ax1.gridlines(
        linestyle="--",
        ylocs=np.arange(-90, -65, 5),
        xlocs=np.arange(-180, 180, 30),
        draw_labels=True,
    )
    gl.xlabels_top = True
    # gl.ylabels_right = True

    gl.xformatter = LongitudeFormatter()

    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1.set_boundary(circle, transform=ax1.transAxes)


def plot_Cld_no_mean_simple_sourth_polar_half_hemisphere(
    Cld_match_PC_gap,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    from cartopy.mpl.ticker import (
        LatitudeFormatter,
        LongitudeFormatter,
    )

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, -1, 90)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(14, 11),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.SouthPolarStereo(),
    )
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.coastlines(resolution="50m", lw=1.7)
    # gl = ax1.gridlines(
    #     linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    # )
    gl = ax1.gridlines(
        linestyle="--",
        ylocs=np.arange(-90, -50, 5),
        xlocs=np.arange(-180, 180, 30),
        draw_labels=True,
    )
    gl.xlabels_top = True
    # gl.ylabels_right = True

    gl.xformatter = LongitudeFormatter()

    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1.set_boundary(circle, transform=ax1.transAxes)


def compare_cld_between_PC_condition_by_Lat(
    Cld_data_PC_condition, Cld_data_aux
):
    """
    Plot mean cld anormaly from lat 30 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    Cld_data_PC_condition = np.concatenate(
        (
            Cld_data_PC_condition[:, 180:],
            Cld_data_PC_condition[:, :180],
        ),
        axis=1,
    )
    Cld_data_aux = np.concatenate(
        (Cld_data_aux[:, 180:], Cld_data_aux[:, :180]), axis=1
    )

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_PC_condition[120:150, :], axis=0),
        color="Blue",
        linewidth=2,
        label="With PC constraint",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[120:150, :], axis=0),
        color="Red",
        label="Without PC constraint",
        linewidth=2,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    ax.set_facecolor("white")
    ax.legend()
    # adjust the legend font size
    for text in ax.get_legend().get_texts():
        plt.setp(text, color="black", fontsize=20)

    # x_ticks_mark = [
    #     "60$^\circ$E",
    #     "120$^\circ$E",
    #     "180$^\circ$",
    #     "120$^\circ$W",
    #     "60$^\circ$W",
    # ]
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    plt.xlim([-180, 180])
    plt.xlabel("Longitude", size=23, weight="bold")
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20, weight="bold")
    plt.yticks(fontsize=20, weight="bold")
    plt.ylabel("HCF difference (%)", size=20, weight="bold")
    # plt.ylim(0, 100)
    plt.show()


def compare_cld_between_PC_condition_by_Lat_smoothed(
    Cld_data_PC_condition,
    Cld_data_aux,
    step,
):
    """
    Plot mean cld anormaly from lat 30 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    Cld_data_PC_condition = np.concatenate(
        (
            Cld_data_PC_condition[:, 180:],
            Cld_data_PC_condition[:, :180],
        ),
        axis=1,
    )
    Cld_data_aux = np.concatenate(
        (Cld_data_aux[:, 180:], Cld_data_aux[:, :180]), axis=1
    )

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition[120:150, :], axis=0),
            step,
            mode="same",
        ),
        color="Blue",
        linewidth=2,
        label="With PC constraint",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[120:150, :], axis=0),
        color="Red",
        label="Without PC constraint",
        linewidth=2,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    ax.set_facecolor("white")
    ax.legend()
    # adjust the legend font size
    for text in ax.get_legend().get_texts():
        plt.setp(text, color="black", fontsize=20)

    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    plt.xlim([-180, 180])
    plt.xlabel("Longitude", size=23, weight="bold")
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20, weight="bold")
    plt.yticks(fontsize=20, weight="bold")
    plt.ylabel("HCF difference (%)", size=20, weight="bold")
    # plt.ylim(0, 100)
    plt.show()


def compare_cld_between_PC_condition_by_each_Lat_smoothed(
    Cld_data_PC_condition_0,
    Cld_data_PC_condition_1,
    Cld_data_PC_condition_2,
    Cld_data_aux,
    step,
):
    """
    Plot mean cld anormaly from lat 30 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    Cld_data_PC_condition_0 = np.concatenate(
        (
            Cld_data_PC_condition_0[:, 180:],
            Cld_data_PC_condition_0[:, :180],
        ),
        axis=1,
    )
    Cld_data_PC_condition_1 = np.concatenate(
        (
            Cld_data_PC_condition_1[:, 180:],
            Cld_data_PC_condition_1[:, :180],
        ),
        axis=1,
    )
    Cld_data_PC_condition_2 = np.concatenate(
        (
            Cld_data_PC_condition_2[:, 180:],
            Cld_data_PC_condition_2[:, :180],
        ),
        axis=1,
    )
    Cld_data_aux = np.concatenate(
        (Cld_data_aux[:, 180:], Cld_data_aux[:, :180]), axis=1
    )

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_0[120:150, :], axis=0),
            step,
            mode="same",
        ),
        color="Indigo",
        linewidth=2,
        label="Bad Atmospheric Condition",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_1[120:150, :], axis=0),
            step,
            mode="same",
        ),
        color="Blue",
        linewidth=2,
        label="Moderate Atmospheric Condition",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_2[120:150, :], axis=0),
            step,
            mode="same",
        ),
        color="Green",
        linewidth=2,
        label="Good Atmospheric Condition",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[120:150, :], axis=0),
        color="Red",
        label="Without PC constraint",
        linewidth=2,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    ax.set_facecolor("white")
    ax.legend()
    # adjust the legend font size
    for text in ax.get_legend().get_texts():
        plt.setp(text, color="black", fontsize=20)

    # x_ticks_mark = [
    #     "60$^\circ$E",
    #     "120$^\circ$E",
    #     "180$^\circ$",
    #     "120$^\circ$W",
    #     "60$^\circ$W",
    # ]
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    plt.xlim([-180, 180])
    plt.xlabel("Longitude", size=23, weight="bold")
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20, weight="bold")
    plt.yticks(fontsize=20, weight="bold")
    plt.ylabel("HCF difference (%)", size=20, weight="bold")
    # plt.ylim(0, 100)
    plt.show()


#########################################################
########### Filter CLD data to fit PC1 gap #######################################
########################################################


class Filter_data_fit_PC1_gap_plot(object):
    def __init__(self, Cld_data, start, end, gap):
        self.start = start
        self.end = end
        self.gap = gap
        self.latitude = [i for i in range(0, Cld_data.shape[1], 1)]
        self.longitude = [i for i in range(0, Cld_data.shape[2], 1)]

    def Filter_data_fit_PC1_gap(self, Cld_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data
        PC_data : numpy.array
            PC data
        start : int
            Start PC value, like -1
        end : int
            End PC value, like 2
        gap : int
            Giving gap, like 0.2

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num, " ******")
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
            " ******",
        )
        PC_gap = [i for i in range(0, int(gap_num), 1)]

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        for lat in self.latitude:
            for lon in self.longitude:
                for gap_num in PC_gap:
                    # Filter Cld data with gap, start and end with giving gap
                    Cld_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        Cld_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )
                    # generate PC match PC gap as well to insure
                    PC_match_PC_gap[gap_num, lat, lon] = np.nanmean(
                        PC_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )

        return Cld_match_PC_gap, PC_match_PC_gap

    def Filter_data_fit_PC1_gap_each_day(self, Cld_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data
        PC_data : numpy.array
            PC data
        start : int
            Start PC value, like -1
        end : int
            End PC value, like 2
        gap : int
            Giving gap, like 0.2

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num)
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
        )
        PC_gap = [i for i in range(0, int(gap_num), 1)]

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        for lat in self.latitude:
            for lon in self.longitude:
                for gap_num in PC_gap:
                    # Filter Cld data with gap, start and end with giving gap
                    Cld_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        Cld_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )
                    # generate PC match PC gap as well to insure
                    PC_match_PC_gap[gap_num, lat, lon] = np.nanmean(
                        PC_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )

        return Cld_match_PC_gap, PC_match_PC_gap

    def numba_Filter_data_fit_PC1_gap(self, Cld_data, PC_data):
        """
        Call numba filter function

        Parameters
        ----------
        Cld_data : array in shape
            (PC_gap_len, latitude_len, longitude_len)
            CLD_data
        PC_data : array in shape
            (PC_gap_len, latitude_len, longitude_len)
            PC1_data
            Returns

        -------
        Same as the numba filter function

        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        PC_gap = [i for i in range(0, int(gap_num), 1)]

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num)
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
        )

        return NUMBA_FILTER_DATA_FIT_PC1_GAP(
            var1,
            var2,
            coef,
            gap_num,
            len(PC_gap),
            len(self.latitude),
            len(self.longitude),
            Cld_data,
            PC_data,
        )

    def give_loop_list_for_giving_gap(self):
        """
        Give the loop list for giving gap

        Parameters
        ----------
        start : int
            start of the loop
        end : int
            end of the loop
        gap : int
            gap

        Returns
        -------
        loop_list : list
            loop list
        """
        range = self.end - self.start
        loop_num = range / self.gap

        var1 = (
            self.start
            * (loop_num - 1)
            / ((self.end - self.gap) - self.start)
        )
        coefficient = self.start / var1
        var2 = self.gap / coefficient + var1

        return var1, var2, coefficient, loop_num

    def calc_correlation_PC1_Cld(self, PC_data, Cld_data):
        Correlation = np.zeros((180, 360))

        for i in range(180):
            for j in range(360):
                Correlation[i, j] = pd.Series(
                    PC_data[:, i, j]
                ).corr(
                    pd.Series(Cld_data[:, i, j]),
                    method="pearson",
                )

        return Correlation

    def plot_PC1_Cld(
        self, start, end, PC_match_PC_gap, Cld_match_PC_gap
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        # lat = np.linspace(-90, 89, 180)
        lat = np.linspace(-90, -1, 90)

        print("****** Start plot PC1 ******")
        fig, (ax1, ax2) = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(20, 20),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            211,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000", alpha=1)
        cmap.set_under("#191970", alpha=1)

        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # vmax=1.5,
            # vmin=-1.5,
            cmap=cmap,
        )
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cbar = fig.colorbar(
            a,
            ax=[ax1],
            location="right",
            shrink=0.9,
            extend="both",
            label="PC 1",
        )
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        cbar.ax.tick_params(labelsize=24)

        ax2 = plt.subplot(
            212,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        ax2.coastlines(resolution="50m", lw=0.3)
        ax2.set_global()
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # vmax=30,
            # vmin=0,
            cmap=cmap,
        )
        gl = ax2.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cbar = fig.colorbar(
            b,
            ax=[ax2],
            location="right",
            shrink=0.9,
            extend="both",
            label="HCF (%)",
        )
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        cbar.ax.tick_params(labelsize=24)
        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_PC1_Cld_Difference(
        self,
        start,
        end,
        PC_match_PC_gap,
        Cld_match_PC_gap,
        pc_max,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, -1, 90)

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        print("****** Start plot PC1 ******")
        fig, (ax1, ax2) = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(15, 15),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            211,
            projection=ccrs.PlateCarree(),
        )

        ax1.set_global()

        # norm1 = colors.CenteredNorm(halfrange=pc_max)
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # norm=norm1,
            vmax=pc_max,
            vmin=-pc_max,
            cmap=cmap,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb1 = fig.colorbar(
            a,
            ax=ax1,
            extend="both",
            location="right",
            shrink=0.8,
        )
        # adjust the colorbar label size
        cb1.set_label(label="PC 1", size=24)
        cb1.ax.tick_params(labelsize=24)

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        # set serial number for this subplot
        # ax1.text(
        #     0.05,
        #     0.95,
        #     "PC",
        #     transform=ax1.transAxes,
        #     fontsize=24,
        #     verticalalignment="top",
        # )

        ax2 = plt.subplot(
            212,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        ax2.set_global()
        norm2 = colors.CenteredNorm(halfrange=cld_max)
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            norm=norm2,
            cmap=cmap,
        )
        ax2.coastlines(resolution="50m", lw=0.9)
        gl = ax2.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = fig.colorbar(
            b,
            ax=ax2,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_Difference(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, -1, 90)

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        print("****** Start plot PC1 ******")
        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(11, 4),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        # ax1.set_global()
        norm2 = colors.CenteredNorm(halfrange=cld_max)
        b = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            norm=norm2,
            cmap=cmap,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

        plt.savefig(
            "/RAID01/data/muqy/PYTHONFIG/Volc_shit/"
            + str(np.round((np.array(start + var1)) * coef, 2))
            + "_PC1_"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2))
            + ".png",
            dpi=250,
            facecolor=fig.get_facecolor(),
            transparent=True,
        )

    def plot_Cld_simple_shit(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, -1, 90)

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        print("****** Start plot PC1 ******")
        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(11, 4),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        # ax1.set_global()
        norm2 = colors.CenteredNorm(halfrange=cld_max)
        b = ax1.pcolormesh(
            lon,
            lat,
            Cld_match_PC_gap,
            transform=ccrs.PlateCarree(),
            norm=norm2,
            cmap=cmap,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

        plt.savefig(
            "/RAID01/data/muqy/PYTHONFIG/Volc_shit/"
            + str(np.round((np.array(start + var1)) * coef, 2))
            + "_PC1_"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2))
            + "after.png",
            dpi=250,
            facecolor=fig.get_facecolor(),
            transparent=True,
        )

    def plot_Cld_simple_test_half_hemisphere(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, -1, 90)

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        print("****** Start plot PC1 ******")
        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(11, 4),
            constrained_layout=True,
        )
        mpl.style.use("seaborn-v0_8-ticks")
        mpl.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        # ax1.set_global()
        b = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=cld_min,
            vmax=cld_max,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_simple_test_full_hemisphere(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
        cld_name,
        cmap_file,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        cmap = dcmap(cmap_file)
        cmap.set_over("#800000")
        cmap.set_under("#191970")
        cmap.set_bad("silver", alpha=0)

        print("****** Start plot PC1 ******")
        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(11, 4),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        # ax1.set_global()
        ax1.set_facecolor("silver")
        b = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(central_longitude=0),
            cmap=cmap,
            vmin=cld_min,
            vmax=cld_max,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb.set_label(label=cld_name, size=24)
        cb.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.5,
            y=1.11,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_simple_test_tropical(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
        cld_name,
        cmap_file,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        cmap = dcmap(cmap_file)
        cmap.set_over("#800000")
        cmap.set_under("#191970")
        cmap.set_bad("silver", alpha=0)

        print("****** Start plot PC1 ******")
        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(12, 2.5),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        # ax1.set_global()
        ax1.set_extent([-180, 180, -30, 30], ccrs.PlateCarree())
        ax1.set_facecolor("silver")
        b = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(central_longitude=0),
            cmap=cmap,
            vmin=cld_min,
            vmax=cld_max,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb.set_label(label=cld_name, size=24)
        cb.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.5,
            y=1.11,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_no_mean_full_hemisphere(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
        cld_name,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        print("****** Start plot PC1 ******")
        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(11, 4),
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
            Cld_match_PC_gap,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=cld_min,
            vmax=cld_max,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb2.set_label(label=cld_name, size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.5,
            y=1.11,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_no_mean_simple_full_hemisphere(
        self,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
        cld_name,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        print("****** Start plot PC1 ******")
        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(11, 4),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        # ax1.set_global()
        b = ax1.pcolormesh(
            lon,
            lat,
            Cld_match_PC_gap,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=cld_min,
            vmax=cld_max,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb2.set_label(label=cld_name, size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        # plt.suptitle(
        #     str(np.round((np.array(start + var1)) * coef, 2))
        #     + "<=PC1<"
        #     + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
        #     x=0.5,
        #     y=1.11,
        #     size=42,
        #     fontweight="bold",
        # )

    def plot_PC1_Cld_test(
        self,
        start,
        end,
        Cld_data,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        print("****** Start plot PC1 ******")
        fig, ax1 = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(15, 8),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        # norm1 = colors.CenteredNorm(halfrange=pc_max)
        a = ax1.pcolormesh(
            lon,
            lat,
            Cld_data,
            transform=ccrs.PlateCarree(),
            # norm=norm1,
            vmax=cld_max,
            vmin=-cld_max,
            cmap=cmap,
        )
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb1 = fig.colorbar(
            a,
            ax=ax1,
            extend="both",
            location="right",
            shrink=0.7,
        )
        # adjust the colorbar label size
        cb1.set_label(label="PC 1", size=24)
        cb1.ax.tick_params(labelsize=24)

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_All_year_mean_PC1_Cld(self, PC_data, Cld_data):
        # plot all year mean PC1 and Cld
        # ! Input PC_data and Cld_data must be the same shape
        # ! [time, lat, lon]
        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)
        # lon,lat1 = np.meshgrid(lon,lat1)

        print("****** Start plot all year mean PC1 and Cld ******")
        fig = plt.figure(figsize=(18, 15))
        plt.rc("font", size=10, weight="bold")

        cmap = dcmap("/RAID01/data/muqy/color/test_cld.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("white")

        cmap1 = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap1.set_bad("gray")
        cmap1.set_over("#800000")
        cmap1.set_under("#191970")

        ax1 = plt.subplot(
            2,
            1,
            1,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_data[:, :, :], axis=0),
            linewidth=0,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmax=40,
            vmin=0,
        )
        ax1.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        ax1.set_title(" High Cloud Fraction (HCF) ", size=12)
        fig.colorbar(
            a,
            ax=[ax1],
            location="right",
            shrink=0.9,
            extend="both",
            label="HCF (%)",
        )

        ax2 = plt.subplot(
            2,
            1,
            2,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax2.coastlines(resolution="50m", lw=0.7)
        ax2.set_global()
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_data[:, :, :], axis=0),
            linewidth=0,
            transform=ccrs.PlateCarree(),
            # norm=MidpointNormalize(midpoint=0),
            cmap=cmap1,
            vmax=2,
            vmin=-1,
        )
        ax2.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        ax2.set_title(" Principle Component 1 (PC1) ", size=12)
        fig.colorbar(
            b,
            ax=[ax2],
            location="right",
            shrink=0.9,
            extend="both",
            label="PC1",
        )
        # plt.savefig('PC1_CLDAREA1.pdf')
        # plt.tight_layout()
        plt.show()

    def plot_correlation_PC1_Cld(self, Corr_data):
        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)
        lat1 = np.linspace(0, 69, 70)
        # lon,lat1 = np.meshgrid(lon,lat1)

        fig = plt.figure(figsize=(10, 6))
        plt.rc("font", size=10, weight="bold")

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        ax1 = plt.subplot(
            1,
            1,
            1,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax1.coastlines(resolution="50m", lw=0.3)
        ax1.set_global()
        a = ax1.pcolor(
            lon,
            lat,
            Corr_data,
            # Corr_all,
            # Corr_d,
            linewidth=0,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmax=1,
            vmin=-1,
        )
        ax1.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        ax1.set_title(" PC1-HCF Correlation (Corr) ", size=12)
        fig.colorbar(
            a,
            ax=[ax1],
            location="right",
            shrink=0.9,
            extend="both",
            label="Corr",
        )

        plt.show()

    def Convert_pandas(self, Cld_match_PC_gap):
        gap_num = Cld_match_PC_gap.shape[0]
        Box = np.zeros(
            (
                Cld_match_PC_gap.shape[1]
                * Cld_match_PC_gap.shape[2],
                gap_num,
            )
        )
        # Box = np.zeros((gap_num, 64800, ))

        for i in range(gap_num):
            Box[:, i] = Cld_match_PC_gap[i, :, :].reshape(-1)

        Box = pd.DataFrame(Box)

        # Number of PC1 interval is set in np.arange(start, end, interval gap)
        Box.columns = np.round(
            np.arange(self.start, self.end, self.gap), 3
        )

        return Box

    def plot_box_plot(self, Cld_match_PC_gap):
        """
        Plot boxplot of Cld data match each PC1 interval
        Main plot function
        """
        Box = self.Convert_pandas(Cld_match_PC_gap)

        plt.style.use("seaborn-v0_8-ticks")  # type: ignore
        plt.rc("font", family="Times New Roman")

        fig, ax = plt.subplots(figsize=(18, 10))
        flierprops = dict(
            marker="o",
            markersize=7,
            markeredgecolor="grey",
        )
        Box.boxplot(
            # sym="o",
            flierprops=flierprops,
            whis=[10, 90],
            meanline=None,
            showmeans=True,
            notch=True,
        )
        plt.xlabel("PC1", size=26, weight="bold")
        plt.ylabel("HCF (%)", size=26, weight="bold")
        # plt.xticks((np.round(np.arange(-1.5, 3.5, 0.5), 2)),fontsize=26, weight="bold", )
        plt.xticks(rotation=45)
        plt.yticks(
            fontsize=26,
            weight="bold",
        )
        # plt.savefig(
        #     "Box_plot_PC1_Cld_" + self.time_str + ".png",
        #     dpi=500,
        #     facecolor=fig.get_facecolor(),
        #     edgecolor="none",
        # )
        plt.show()


# Filter data with gap, start and end with giving gap
# This is for atmos para filter


class FilterAtmosDataFitPCgap(Filter_data_fit_PC1_gap_plot):
    def Filter_data_fit_PC1_gap_atmos(self, Atmos_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Atmos_data : numpy.array
            Atmos data
        PC_data : numpy.array
            PC data
        start : int
            Start PC value, like -1
        end : int
            End PC value, like 2
        gap : int
            Giving gap, like 0.2

        Returns
        -------
        Atmos_data_fit : numpy.array
            Filtered data, Atmos data for each PC gap
            array(PC_gap, lat, lon)
        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num, " ******")
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
            " ******",
        )
        PC_gap = [i for i in range(0, int(gap_num), 1)]

        Atmos_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )

        for lat in self.latitude:
            for lon in self.longitude:
                for gap_num in PC_gap:
                    # Filter Cld data with gap, start and end with giving gap
                    Atmos_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        Atmos_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )

        return Atmos_match_PC_gap


#########################################################
########### Box Plot #######################################
########################################################


class Box_plot(object):
    """
    Plot boxplot of Cld data match each PC1 interval

    """

    def __init__(self, Cld_match_PC_gap, time_str):
        """
        Initialize the class

        Parameters
        ----------
        Cld_match_PC_gap : Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
        time_str : string
            time string like '2010to2019' or "2010to2019_4_6_month" or "2018only"
        """
        # Input array must be in shape of (PC1_gap, lat, lon)
        self.Cld_match_PC_gap = Cld_match_PC_gap
        self.time_str = time_str

    def Convert_pandas(self):
        gap_num = self.Cld_match_PC_gap.shape[0]
        Box = np.zeros(
            (
                self.Cld_match_PC_gap.shape[1]
                * self.Cld_match_PC_gap.shape[2],
                gap_num,
            )
        )
        # Box = np.zeros((gap_num, 64800, ))

        for i in range(gap_num):
            Box[:, i] = self.Cld_match_PC_gap[i, :, :].reshape(-1)

        Box = pd.DataFrame(Box)
        # Number of PC1 interval is set in np.arange(start, end, interval gap)
        Box.columns = np.round(np.arange(-1.5, 4.5, 0.05), 3)

        return Box

    def plot_box_plot(self):
        """
        Plot boxplot of Cld data match each PC1 interval
        Main plot function
        """
        Box = self.Convert_pandas()

        fig, ax = plt.subplots(figsize=(18, 10))
        flierprops = dict(
            marker="o",
            markersize=7,
            markeredgecolor="grey",
        )
        Box.boxplot(
            # sym="o",
            flierprops=flierprops,
            whis=[10, 90],
            meanline=None,
            showmeans=True,
            notch=True,
        )
        plt.xlabel("PC1", size=26, weight="bold")
        plt.ylabel("HCF (%)", size=26, weight="bold")
        # plt.xticks((np.round(np.arange(-1.5, 3.5, 0.5), 2)),fontsize=26, weight="bold", )
        plt.yticks(
            fontsize=26,
            weight="bold",
        )
        plt.savefig(
            "Box_plot_PC1_Cld_" + self.time_str + ".png",
            dpi=500,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.show()


# ---------- Comparing the difference between other years and 2020 -------------------
# ---------- In the giving PC1 gap, and the giving atmospheric gap -------------------


def compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020,
    Cld_all_match_PC_gap_others,
    start,
    end,
):
    """
    Loop over the given array to See if the data at each location is nan, if so,
    assign it to nan, if not, subtract the data at that location within two years
    """
    Cld_all_match_PC_gap_2020_sub_others = np.zeros(
        (
            Cld_all_match_PC_gap_2020.shape[1],
            Cld_all_match_PC_gap_2020.shape[2],
        )
    )

    Cld_all_match_PC_gap_2020_sub_others[:, :] = np.nan

    for lat in range((Cld_all_match_PC_gap_2020.shape[1])):
        for lon in range((Cld_all_match_PC_gap_2020.shape[2])):
            for gap in range(start, end):
                if (
                    np.isnan(
                        Cld_all_match_PC_gap_2020[gap, lat, lon]
                    )
                    == False
                ) and (
                    np.isnan(
                        Cld_all_match_PC_gap_others[gap, lat, lon]
                    )
                    == False
                ):
                    Cld_all_match_PC_gap_2020_sub_others[
                        lat, lon
                    ] = (
                        Cld_all_match_PC_gap_2020[gap, lat, lon]
                        - Cld_all_match_PC_gap_others[gap, lat, lon]
                    )
                elif (
                    np.isnan(
                        Cld_all_match_PC_gap_2020[gap, lat, lon]
                    )
                    == True
                ) or (
                    np.isnan(
                        Cld_all_match_PC_gap_others[gap, lat, lon]
                    )
                    == True
                ):
                    if (
                        np.isnan(
                            Cld_all_match_PC_gap_2020_sub_others[
                                lat, lon
                            ]
                        )
                        == True
                    ):
                        Cld_all_match_PC_gap_2020_sub_others[
                            lat, lon
                        ] = np.nan
                    else:
                        pass

    return Cld_all_match_PC_gap_2020_sub_others


# -------------- Calculate the coefficient of variation (CV) ---------------
# -------------- of the given array ---------------
# -------------- and the Index of dispersion ---------------


def Calculate_coefficient_of_variation(input_array):
    """
    Calculated input array's coefficient of variation

    Parameters
    ----------
    input_array : array
        an array of numnber waiting to be calculated

    Returns
    -------
    float
        the CV of input array in %
    """
    # reshape the array to 1D
    array = np.array(input_array.reshape(-1))

    return (np.nanstd(array) / np.nanmean(array)) * 100


def Calculate_index_of_dispersion(input_array):
    """
    Calculated input array's index of dispersion

    Parameters
    ----------
    input_array : array
        an array of numnber waiting to be calculated

    Returns
    -------
    float
        the Index of dispersion of input array in %
    """
    # reshape the array to 1D
    array = np.array(input_array.reshape(-1))

    return ((np.nanstd(array) ** 2) / np.nanmean(array)) * 100


# ------------- Save calculated data as netcdf file ----------------------------------------------------


def save_PCA_data_as_netcdf(PC_filtered, Cld_filtered):
    """
    Save the PCA and Cld data as netcdf file

    Parameters
    ----------
    PC_filtered : array
        the filtered PCA data
    Cld_filtered : array
        the filtered Cld data
    """
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    PC_all = PC_all.reshape(132, 28, 180, 360)  # PCA

    ds = xr.Dataset(
        {
            "PC1": (
                ("Gap", "Latitude", "Longitude"),
                PC_filtered[:, :, :],
            ),
            "CLD": (
                ("Gap", "Latitude", "Longitude"),
                Cld_filtered[:, :, :],
            ),
        },
        coords={
            "gap_num": ("Gap", np.linspace(-90, 89, 180)),
            "lat": ("Latitude", np.linspace(-90, 89, 180)),
            "lon": ("Longitude", np.linspace(0, 359, 360)),
        },
    )

    os.makedirs("/RAID01/data/PCA_data/", exist_ok=True)
    ds.to_netcdf(
        "/RAID01/data/2010_2020_5_parameters_300hPa_PC1.nc"
    )
