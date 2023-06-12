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

    Code to test if filter data for 10% lowermost and 90% highermost
    can reveal the anormal cirrus signal in 2020.
        
    Owner: Mu Qingyu
    version 1.0
        version 1.1: 2022-11-28
        - we only analyze the filtered data for 10% lowermost and 90% highermost
        - In order to extract maximum signal of contrail, only april to july 
          cld and pc1 data are used
          
    Created: 2022-10-27
    
    Including the following parts:

        1) Read in basic PCA & Cirrus data (include cirrus morphology and microphysics)
                
        2) Filter anormal hcf data within lowermost 10% or highermost 90%
        
        3) Plot the filtered data to verify the anormal cirrus signal
        
        4) Calculate the mean and std of filtered data, cv of filtered data
        
"""

# import modules
import gc

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)

# --------- import done ------------
# --------- Plot style -------------
mpl.rc("font", family="Times New Roman")
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")
mpl.rc("font", family="Times New Roman")

# ---------- Read PCA&CLD data from netcdf file --------

# ---------- Read in Cloud area data ----------
# now we read IWP and other cld data (not IWP) from netcdf file
(
    # pc
    PC_all,
    PC_years,
    # cld
    Cld_all_HCF,
    Cld_years,
    # iwp
    IWP_data,
    IWP_years,
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name="Cldarea")

# Delete the unused variables and free the memory
del PC_years, Cld_years, IWP_years
gc.collect()

(
    # pc
    _,
    PC_years,
    # cld
    Cld_all_IPR,
    Cld_years,
    # iwp
    _,
    IWP_years,
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name="Cldicerad")

# Delete the unused variables and free the memory
del PC_years, Cld_years, IWP_years
gc.collect()

(
    # pc
    _,
    PC_years,
    # cld
    Cld_all_CEH,
    Cld_years,
    # iwp
    _,
    IWP_years,
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name="Cldeff_hgth")

# Delete the unused variables and free the memory
del PC_years, Cld_years, IWP_years
gc.collect()    

# CERES_Cld_dataset = [
#     "Cldarea",
#     "Cldicerad",
#     "Cldeff_hgth",
#     "Cldpress_base",
#     "Cldhgth_top",
#     "Cldtau",
#     "Cldtau_lin",
#     "IWP",
#     "Cldemissir",
# ]

# Implementation for MERRA2 dust AOD
# extract the data from 2010 to 2014 like above
data_merra2_2010_2020_new_lon = xr.open_dataset(
    "/RAID01/data/merra2/merra_2_daily_2010_2020_new_lon.nc"
)

# # Extract Dust aerosol data from all data
Dust_AOD = data_merra2_2010_2020_new_lon["DUEXTTAU"].values.reshape(
    3696, 180, 360
)

# use the 2010-2020 PC1 only
PC_all = PC_all[-11:].astype(np.float32).reshape(3696, 180, 360)
Cld_all_HCF = Cld_all_HCF.astype(np.float32).reshape(3696, 180, 360)
Cld_all_IPR = Cld_all_IPR.astype(np.float32).reshape(3696, 180, 360)
Cld_all_CEH = Cld_all_CEH.astype(np.float32).reshape(3696, 180, 360)
IWP_data = IWP_data.astype(np.float32).reshape(3696, 180, 360)

# convert the data type to float32
Dust_AOD = Dust_AOD.astype(np.float32)


# Data read finished
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Set the polar regions to nan
def set_polar_regions_to_nan(data_array):
    # Assuming the shape is (time, lat, lon)
    _, num_latitudes, _ = data_array.shape

    # Calculate the indices corresponding to -90 + 30 and 90 - 30 degrees of latitude
    lower_index = int((num_latitudes * 30) / 180)
    upper_index = int((num_latitudes * 150) / 180)

    # Set the polar regions to nan
    data_array[:, :lower_index, :] = np.nan
    data_array[:, upper_index:, :] = np.nan


# Assuming Cld_all is your input array with shape (3696, 180, 360)
set_polar_regions_to_nan(Cld_all_HCF)
set_polar_regions_to_nan(Cld_all_IPR)
set_polar_regions_to_nan(Cld_all_CEH)
set_polar_regions_to_nan(IWP_data)
set_polar_regions_to_nan(Dust_AOD)
set_polar_regions_to_nan(PC_all)


# filter the extreme 2.5% data for IWP and AOD only
def filter_extreme_5_percent(IWP_data, AOD_data, PC_data, Cld_data):
    # Calculate the threshold values for the largest and smallest 2.5% of AOD and IWP data
    lower_threshold_IWP = np.nanpercentile(IWP_data, 5)
    upper_threshold_IWP = np.nanpercentile(IWP_data, 95)
    lower_threshold_AOD = np.nanpercentile(AOD_data, 5)
    upper_threshold_AOD = np.nanpercentile(AOD_data, 95)

    # Create a mask for extreme values in AOD and IWP data
    extreme_mask = (
        (IWP_data < lower_threshold_IWP)
        | (IWP_data > upper_threshold_IWP)
        | (AOD_data < lower_threshold_AOD)
        | (AOD_data > upper_threshold_AOD)
    )

    # Apply the mask to IWP, AOD, PC, and CLD data
    IWP_filtered = np.where(extreme_mask, np.nan, IWP_data)
    AOD_filtered = np.where(extreme_mask, np.nan, AOD_data)
    PC_filtered = np.where(extreme_mask, np.nan, PC_data)
    Cld_filtered = np.where(extreme_mask, np.nan, Cld_data)

    return IWP_filtered, AOD_filtered, PC_filtered, Cld_filtered


# Assuming IWP_data, Dust_AOD, PC_all, and Cld_all are your input arrays with shape (3686, 180, 360)
(
    IWP_all_filtered,
    Dust_AOD_filtered,
    PC_all_filtered,
    Cld_all_HCF_filtered,
) = filter_extreme_5_percent(IWP_data, Dust_AOD, PC_all, Cld_all_HCF)

(
    _,
    _,
    _,
    Cld_all_CEH_filtered,
) = filter_extreme_5_percent(IWP_data, Dust_AOD, PC_all, Cld_all_CEH)

(
    _,
    _,
    _,
    Cld_all_IPR_filtered,
) = filter_extreme_5_percent(IWP_data, Dust_AOD, PC_all, Cld_all_IPR)

# -------------------------------------------------------------------
# Mean by time for verification
Dust_AOD_filtered_mean = np.nanmean(Dust_AOD_filtered, axis=0)
IWP_all_filtered_mean = np.nanmean(IWP_all_filtered, axis=0)
Cld_all_HCF_filtered_mean = np.nanmean(Cld_all_HCF_filtered, axis=0)
Cld_all_IPR_filtered_mean = np.nanmean(Cld_all_IPR_filtered, axis=0)
Cld_all_CEH_filtered_mean = np.nanmean(Cld_all_CEH_filtered, axis=0)
PC_all_filtered_mean = np.nanmean(PC_all_filtered, axis=0)

##############################################################################################################
### Data verification ##############################################################################
##############################################################################################################


def plot_spatial_distribution(
    data, var_name, title, AOD_intervals, cmap="jet"
):
    from matplotlib.colors import BoundaryNorm

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    from matplotlib.colors import LinearSegmentedColormap

    # Define colors for each interval (use as many colors as the number of intervals)
    colors = ["green", "blue", "yellow", "orange", "red", "purple"]

    # Create custom colormap
    cmap = plt.cm.get_cmap(cmap, len(AOD_intervals) - 1)

    # Set up the norm and boundaries for the colormap
    norm = BoundaryNorm(AOD_intervals, cmap.N)

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
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


def plot_spatial_distribution_mask(
    data,
    var_name,
    title,
    data_intervals,
    mask_threshold=None,
    cmap="jet",
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    # Create custom colormap
    cmap = plt.cm.get_cmap(cmap, len(data_intervals) - 1)

    # Set up the norm and boundaries for the colormap
    norm = BoundaryNorm(data_intervals, cmap.N)

    # Mask the data based on the mask_threshold
    if mask_threshold is not None:
        data = np.where(
            (data >= data_intervals[mask_threshold[0]])
            & (data <= data_intervals[mask_threshold[1]]),
            data,
            np.nan,
        )

    # Calculate the mean of the data array along the first axis (time)
    data = np.nanmean(data, axis=0)

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
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


# version 2: no colorbar limit
def plot_spatial_distribution_mask(
    data,
    var_name,
    title,
    data_intervals,
    mask_threshold=None,
    cmap="jet",
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    # Create custom colormap
    cmap = plt.cm.get_cmap(cmap)

    # Mask the data based on the mask_threshold
    if mask_threshold is not None:
        data = np.where(
            (data >= data_intervals[mask_threshold[0]])
            & (data <= data_intervals[mask_threshold[1]]),
            data,
            np.nan,
        )

    # Calculate the mean of the data array along the first axis (time)
    data = np.nanmean(data, axis=0)

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
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



# version 3: grid point frequency sum
def plot_spatial_distribution_mask(
    data,
    var_name,
    title,
    data_intervals,
    mask_threshold=None,
    cmap="jet",
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    # Create custom colormap
    cmap = plt.cm.get_cmap(cmap)

    # Count the frequency of values within the specified range at each grid point
    if mask_threshold is not None:
        data = np.count_nonzero(
            (data >= data_intervals[mask_threshold[0]])
            & (data <= data_intervals[mask_threshold[1]]),
            axis=0,
        )
        
    # Count the frequency of values within the specified range at each grid point
    # if mask_threshold is not None:
    #     data = np.where(
    #         (data >= data_intervals[mask_threshold[0]])
    #         & (data <= data_intervals[mask_threshold[1]]),
    #         1,
    #         0,
    #     )

    # Calculate the sum of the data array along the first axis (time)
    # data = np.sum(data, axis=0)
    # data[np.isnan(data)] = 0
    data = np.where(data == 0, np.nan, data)

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
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


# Example usage:
# plot_spatial_distribution(data, 'IWP', 'IWP Distribution', [0, 5, 10, 15], mask_threshold=15)

# Plot the distribution gap for Dust AOD
AOD_temp = Dust_AOD_filtered

divide_AOD = DividePCByDataVolume(
    dataarray_main=AOD_temp,
    n=6,
)
AOD_gap = divide_AOD.main_gap()


plot_spatial_distribution(
    data=Dust_AOD_filtered,
    var_name="Dust AOD",
    title="Spatial Distribution of Dust AOD Section",
    AOD_intervals=AOD_gap,
    cmap="RdYlBu_r",
)

plot_spatial_distribution_mask(
    data=Dust_AOD_filtered,
    var_name="Dust AOD",
    title="Spatial Distribution of Dust AOD Section",
    data_intervals=AOD_gap,
    mask_threshold=[0, 6],
    cmap="RdYlBu_r",
)

# -----------------------------
# Plot the distribution gap for IWP
IWP_temp = IWP_all_filtered

divide_IWP = DividePCByDataVolume(
    dataarray_main=IWP_temp,
    n=30,
)
IWP_gap = divide_IWP.main_gap()


plot_spatial_distribution(
    data=IWP_all_filtered,
    var_name="IWP",
    title="Spatial Distribution of IWP Section",
    AOD_intervals=IWP_gap,
    cmap="RdYlBu_r",
)

plot_spatial_distribution_mask(
    data=IWP_all_filtered,
    var_name="IWP Frequency",
    title="Spatial Distribution of IWP Frequency Section",
    data_intervals=IWP_gap,
    mask_threshold=[22, 29],
    cmap="RdYlBu_r",
)

# --------------------------------------
# --------------------------------------
# Plot the distribution gap for Cld data
Cld_temp = Cld_all_IPR_filtered

divide_Cld = DividePCByDataVolume(
    dataarray_main=Cld_temp,
    n=50,
)
Cld_gap = divide_Cld.main_gap()


plot_spatial_distribution(
    data=Cld_all_IPR_filtered_mean,
    var_name="IPR",
    title="Spatial Distribution of IPR Section",
    AOD_intervals=Cld_gap,
    cmap="RdYlBu_r",
)

plot_spatial_distribution_mask(
    data=Cld_all_IPR_filtered,
    var_name="IPR Frequency",
    title="Spatial Distribution of IPR Frequency Section",
    data_intervals=Cld_gap,
    mask_threshold=[0, 6],
    cmap="RdYlBu_r",
)



def plot_global_spatial_distribution(
    data,
    var_name,
    title,
):
    # Create lon and lat
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    # Create custom colormap
    cmap = plt.cm.get_cmap("RdYlBu_r")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
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


# verify the input data
plot_global_spatial_distribution(
    Dust_AOD_filtered_mean,
    "Dust AOD",
    "Dust AOD Spatial Distribution",
)

# verify the input data
plot_global_spatial_distribution(
    IWP_all_filtered_mean,
    "IWP",
    "IWP Spatial Distribution",
)

# verify the input data
plot_global_spatial_distribution(
    Cld_all_HCF_filtered_mean,
    "HCF",
    "HCF Spatial Distribution",
)

# verify the input data
plot_global_spatial_distribution(
    Cld_all_IPR_filtered_mean,
    "IPR",
    "IPR Spatial Distribution",
)

# verify the input data
plot_global_spatial_distribution(
    Cld_all_CEH_filtered_mean,
    "CEH",
    "CEH Spatial Distribution",
)

# verify the input data
plot_global_spatial_distribution(
    PC_all_filtered_mean,
    "PC1",
    "PC1 Spatial Distribution",
)
