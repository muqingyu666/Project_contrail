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
from statistics import mean

import matplotlib as mpl
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)
from PIL import Image
from scipy import stats
from scipy.stats import norm

# --------- import done ------------
# --------- Plot style -------------
mpl.rc("font", family="Times New Roman")
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")

# ----------------------------------
# Read in filtered data
def read_filtered_data_out(
    file_name: str = "Cld_match_PC_gap_IWP_AOD_constrain_mean_2010_2020.nc",
):
    Cld_match_PC_gap_IWP_AOD_constrain_mean = xr.open_dataarray(
        "/RAID01/data/Filtered_data/" + file_name
    )

    return Cld_match_PC_gap_IWP_AOD_constrain_mean


# Read the filtered data
# Read the Dust constrain data
# ----------------------------------
# Pristine filtered data
# Read the Dust constrain HCF data
Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_HCF = read_filtered_data_out(
    file_name="Cldarea_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_pristine_6_aod_gaps.nc"
)
# Read the Dust constrain cld icerad data
Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_ICERAD = read_filtered_data_out(
    file_name="Cldicerad_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_pristine_6_aod_gaps.nc"
)
# Read the Dust constrain cld effective hgt data
Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_EFFHGT = read_filtered_data_out(
    file_name="Cldeff_hgth_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_pristine_6_aod_gaps.nc"
)

# ----------------------------------
# Masked filtered data
# Read the masked Dust constrain HCF data
Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_HCF_masked = read_filtered_data_out(
    file_name="Cldarea_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_mask_4_aod_gaps.nc"
)
# Read the masked Dust constrain cld icerad data
Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_ICERAD_masked = read_filtered_data_out(
    file_name="Cldicerad_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_mask_4_aod_gaps.nc"
)
# Read the masked Dust constrain cld effective hgt data
Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_EFFHGT_masked = read_filtered_data_out(
    file_name="Cldeff_hgth_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_mask_4_aod_gaps.nc"
)

######################################################################################
###### Plot 3d filled figure to represent the cloud amount in each PC1 interval ######
######################################################################################


# color nan values with self-defined color
def create_colormap_with_nan(cmap_name, nan_color="silver"):
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))
    cmap_with_nan = ListedColormap(colors)
    cmap_with_nan.set_bad(nan_color)
    return cmap_with_nan


def plot_3d_colored_IWP_PC1_AOD(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    high_cloud_amount_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    savefig_str,
    aod_range,  # Add this parameter to define the AOD range
    cmap="Spectral_r",  # Add this parameter to define the custom colormap
):
    """
    Create a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval.

    Args:
        Cld_match_PC_gap_IWP_AOD_constrain_mean (numpy.array): Cloud data under each restriction, shape (AOD_bin, IWP_bin, PC_bin, lat, lon)
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        savefig_str (str): String for saving the figure
        aod_range (tuple): Tuple defining the start and end indices of the AOD cases to plot
        cmap (str or matplotlib.colors.Colormap): Custom colormap to use for the plot
    """

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    # Compute the 90th percentile value for normalization
    norm_value = np.nanpercentile(high_cloud_amount_mean, 99.999)

    # Add this line after the function's docstring to create a colormap that handles NaN values
    cmap_with_nan = create_colormap_with_nan(cmap)

    # Iterate over the specified AOD range
    for aod_num in range(aod_range[0], aod_range[1]):
        Z = aod_num * np.ones_like(X)

        # Plot the 2D pcolormesh color fill map for the current AOD interval
        ax.plot_surface(
            Z,
            X,
            Y,
            rstride=1,
            cstride=1,
            facecolors=cmap_with_nan(
                high_cloud_amount_mean[aod_num] / norm_value
            ),
            shade=False,
            edgecolor="none",
            alpha=0.95,
            antialiased=False,
            linewidth=0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # Turn off the grid lines
    ax.grid(False)

    # Add color bar
    m = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap))
    m.set_cmap(cmap_with_nan)
    m.set_array(high_cloud_amount_mean)
    m.set_clim(np.nanmin(high_cloud_amount_mean), norm_value)
    fig.colorbar(
        m, shrink=0.3, aspect=9, pad=0.01, label=colobar_label
    )

    ax.view_init(elev=20, azim=-66)
    ax.dist = 12

    # Save the figure
    plt.savefig(savefig_str)
    plt.show()


def plot_both_3d_fill_plot(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
):
    """
    This function is basiclly used to call the above main plot function
    This function plots a 3D fill plot with the following parameters:
    Cld_match_PC_gap_IWP_AOD_constrain_mean: 3D array of IWP, AOD, and PC1 values
    high_cloud_amount_mean: 2D array of cloud amount values
    xlabel: label for x-axis
    ylabel: label for y-axis
    zlabel: label for z-axis
    savefig_str: string to name the saved figure with
    aod_range: range of AOD values to plot
    """
    # Calculate the mean high cloud amount for each AOD interval, IWP, and PC1 bin
    high_cloud_amount_mean = np.nanmean(
        Cld_match_PC_gap_IWP_AOD_constrain_mean, axis=(3, 4)
    )
    plot_3d_colored_IWP_PC1_AOD(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        savefig_str="subplot_1.png",
        aod_range=(0, 2),
    )
    plot_3d_colored_IWP_PC1_AOD(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        savefig_str="subplot_2.png",
        aod_range=(2, 4),
    )


# version 4
def plot_3d_colored_IWP_PC1_AOD_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    high_cloud_amount_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    savefig_str,
    aod_range,  # Add this parameter to define the AOD range
    vmin,  # Add this parameter to define the minimum value for the color scale
    vmax,  # Add this parameter to define the maximum value for the color scale
    cmap="Spectral_r",  # Add this parameter to define the custom colormap
):
    """
    Create a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval.

    Args:
        Cld_match_PC_gap_IWP_AOD_constrain_mean (numpy.array): Cloud data under each restriction, shape (AOD_bin, IWP_bin, PC_bin, lat, lon)
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        savefig_str (str): String for saving the figure
        aod_range (tuple): Tuple defining the start and end indices of the AOD cases to plot
        vmin (float): Minimum value for the color scale
        vmax (float): Maximum value for the color scale
    """

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    # Add this line after the function's docstring to create a colormap that handles NaN values
    cmap_with_nan = create_colormap_with_nan(cmap)

    # Iterate over the specified AOD range
    for aod_num in range(aod_range[0], aod_range[1]):
        Z = aod_num * np.ones_like(X)

        # Plot the 2D pcolormesh color fill map for the current AOD interval
        ax.plot_surface(
            Z,
            X,
            Y,
            rstride=1,
            cstride=1,
            facecolors=cmap_with_nan(
                (high_cloud_amount_mean[aod_num] - vmin)
                / (vmax - vmin)
            ),
            shade=False,
            edgecolor="none",
            alpha=0.95,
            antialiased=False,
            linewidth=0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # Turn off the grid lines
    ax.grid(False)

    # Add color bar
    m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r)
    m.set_cmap(cmap_with_nan)
    m.set_array(high_cloud_amount_mean)
    m.set_clim(vmin, vmax)
    fig.colorbar(
        m, shrink=0.3, aspect=9, pad=0.01, label=colobar_label
    )

    ax.view_init(elev=10, azim=-60)
    ax.dist = 12

    # Save the figure
    plt.savefig(savefig_str)
    plt.show()


def plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    vmin,  # Add this parameter to define the minimum value for the color scale
    vmax,  # Add this parameter to define the maximum value for the color scale
):
    """
    This function is basiclly used to call the above main plot function
    This function plots a 3D fill plot with the following parameters:
    Cld_match_PC_gap_IWP_AOD_constrain_mean: 3D array of IWP, AOD, and PC1 values
    high_cloud_amount_mean: 2D array of cloud amount values
    xlabel: label for x-axis
    ylabel: label for y-axis
    zlabel: label for z-axis
    savefig_str: string to name the saved figure with
    aod_range: range of AOD values to plot
    """
    # Calculate the mean high cloud amount for each AOD interval, IWP, and PC1 bin
    high_cloud_amount_mean = np.nanmean(
        Cld_match_PC_gap_IWP_AOD_constrain_mean, axis=(3, 4)
    )
    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        "subplot_1.png",
        (0, 3),
        vmin,  # Add this parameter to define the minimum value for the color scale
        vmax,  # Add this parameter to define the maximum value for the color scale
    )
    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        "subplot_2.png",
        (3, 6),
        vmin,  # Add this parameter to define the minimum value for the color scale
        vmax,  # Add this parameter to define the maximum value for the color scale
    )


# Call the function with different AOD ranges and save each figure separately
# -----------------------------------------------
# Plot the dust-AOD constrained data
# high cloud area
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_HCF,
    "Dust-AOD",
    "PC1",
    "IWP",
    "HCF (%)",
    vmin=0,
    vmax=43,
)

# ice effective radius
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_ICERAD,
    "Dust-AOD",
    "PC1",
    "IWP",
    "IPR (micron)",
    vmin=14.2,
    vmax=34.3,
)

# cloud effective height
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_EFFHGT,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CEH (km)",
    vmin=9.8,
    vmax=14.5,
)

# ----------------------------
# Plot the masked data
# high cloud area
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_HCF_masked,
    "Dust-AOD",
    "PC1",
    "IWP",
    "HCF (%)",
    vmin=0,
    vmax=53,
)

# ice effective radius
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_ICERAD_masked,
    "Dust-AOD",
    "PC1",
    "IWP",
    "IPR (micron)",
    vmin=14,
    vmax=34,
)

# cloud effective height
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_Dust_EFFHGT_masked,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CEH (km)",
    vmin=10.4,
    vmax=14.7,
)


# -----------------------------------------------------------------------
# Plot spatial distribution of different AOD gap values
# -----------------------------------------------------------------------
# Divide 3, Dust AOD data
# Divide AOD data as well
def plot_spatial_distribution(data, var_name, title, AOD_intervals, cmap="jet"):
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

# Read the data
(
    # pc
    PC_all,
    PC_years,
    # cld
    Cld_all,
    Cld_years,
    # iwp
    IWP_data,
    IWP_years,
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name="Cldicerad")

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

# AOD file
data_merra2_2010_2020_new_lon = xr.open_dataset("/RAID01/data/merra2/merra_2_daily_2010_2020_new_lon.nc")

# Plot the distribution gap for Dust AOD
AOD_temp = data_merra2_2010_2020_new_lon["DUEXTTAU"].values

divide_AOD = DividePCByDataVolume(
    dataarray_main=AOD_temp,
    n=6,
)
AOD_gap = divide_AOD.main_gap()

Dust_2010_2020_mean = np.nanmean(AOD_temp, axis=0)

plot_spatial_distribution(
    data=Dust_2010_2020_mean,
    var_name="Dust AOD",
    title="Spatial Distribution of Dust AOD Section",
    AOD_intervals=AOD_gap,
    cmap="RdYlBu_r",
)

# Plot the distribution gap for IWP
PC_temp = PC_all

divide_PC = DividePCByDataVolume(
    dataarray_main=PC_temp,
    n=30,
)
PC_gap = divide_PC.main_gap()

PC_mean = np.nanmean(PC_temp, axis=0)

plot_spatial_distribution(
    data=PC_mean,
    var_name="PC1",
    title="Spatial Distribution of PC1 Section",
    AOD_intervals=PC_gap,
    cmap="RdYlBu_r",
)

# Plot the distribution gap for PC1
IWP_temp = IWP_data

divide_IWP = DividePCByDataVolume(
    dataarray_main=IWP_temp,
    n=30,
)
IWP_gap = divide_IWP.main_gap()

IWP_mean = np.nanmean(IWP_temp, axis=0)

plot_spatial_distribution(
    data=IWP_mean,
    var_name="IWP",
    title="Spatial Distribution of IWP Section",
    AOD_intervals=IWP_gap,
    cmap="RdYlBu_r",
)


# -----------------------------------------------------------------------
#  Plot spatial distribution of global vars distribution
# -----------------------------------------------------------------------


def plot_global_spatial_distribution(
    data,
    var_name,
    title,
):
    from matplotlib.colors import BoundaryNorm

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    from matplotlib.colors import LinearSegmentedColormap

    # Create custom colormap
    cmap = plt.cm.get_cmap("RdBu_r")

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
    IWP_mean,
    "IWP",
    "IWP Spatial Distribution",
)

data = xr.open_dataset(
    "/RAID01/data/merra2/2010/MERRA2_300.tavg1_2d_aer_Nx.20100330.nc4"
)

Dust_AOD_2020_mean = np.nanmean(Dust_AOD_2020, axis=0)
PC_2020_mean = np.nanmean(PC_2020, axis=0)
Cld_2020_mean = np.nanmean(Cld_2020, axis=0)

Dust_AOD_2017_2019_mean = np.nanmean(Dust_AOD_2017_2019, axis=0)
PC_2017_2019_mean = np.nanmean(PC_2017_2019, axis=0)
Cld_2017_2019_mean = np.nanmean(Cld_2017_2019, axis=0)

PC_all = PC_all.reshape(-1, 180, 360)
Cld_all = Cld_all.reshape(-1, 180, 360)
PC_2010_2020_mean = np.nanmean(PC_all, axis=0)
Cld_2010_2020_mean = np.nanmean(Cld_all, axis=0)
Dust_AOD_2010_2020_mean = np.nanmean(Dust_AOD_2010_2020, axis=0)

# verify the input data
plot_global_spatial_distribution(
    Dust_AOD_2010_2020_mean,
    "AOD",
    "AOD Spatial Distribution",
)

plot_global_spatial_distribution(
    PC_2010_2020_mean,
    "PC1",
    "PC1 Spatial Distribution",
)

plot_global_spatial_distribution(
    Cld_2010_2020_mean,
    "CEH",
    "CEH Spatial Distribution",
)

plot_global_spatial_distribution(
    Dust_AOD_2017_2019_mean,
    "AOD",
    "AOD Spatial Distribution",
)
plot_global_spatial_distribution(
    PC_2017_2019_mean,
    "PC1",
    "PC1 Spatial Distribution",
)
plot_global_spatial_distribution(
    Cld_2017_2019_mean,
    "HCF",
    "HCF Spatial Distribution",
)

PC_all_mean = np.nanmean(PC_all, axis=(0, 1, 2))
Cld_all_mean = np.nanmean(Cld_all, axis=(0, 1, 2))


plot_global_spatial_distribution(
    PC_all_mean,
    "PC1",
    "PC1 Spatial Distribution",
)

plot_global_spatial_distribution(
    Cld_all_mean,
    "Cld",
    "Cld Spatial Distribution",
)
