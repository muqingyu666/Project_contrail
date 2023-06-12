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

    Code to test filter extrame value from data
        
"""

# import modules
import gc

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

# Delete the unused variables and free the memory
del PC_years, Cld_years, IWP_years
gc.collect()

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
PC_all = PC_all[-11:].reshape(3696, 180, 360)
Cld_all = Cld_all.reshape(3696, 180, 360)
IWP_data = IWP_data.reshape(3696, 180, 360)

# convert the data type to float32
PC_all = PC_all.astype(np.float32)
Cld_all = Cld_all.astype(np.float32)
IWP_data = IWP_data.astype(np.float32)
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
set_polar_regions_to_nan(Cld_all)
set_polar_regions_to_nan(IWP_data)
set_polar_regions_to_nan(Dust_AOD)
set_polar_regions_to_nan(PC_all)

# Set the largest and smallest 5% of the data to nan
def set_extreme_5_percent_to_nan(data_array):
    
    # Make a copy of the data array
    data_array_filtered = np.copy(data_array)
    
    # Calculate the threshold values for the largest and smallest 5%
    try:
        lower_threshold = np.nanpercentile(data_array, 5)
    except IndexError:
        print("ERROR: Data array is empty.")
        return None
    else:
        upper_threshold = np.nanpercentile(data_array, 95)
    
    # Set the largest and smallest 5% of the data array to nan
    data_array_filtered[data_array_filtered < lower_threshold] = np.nan
    data_array_filtered[data_array_filtered > upper_threshold] = np.nan
    
    return data_array_filtered

# Assuming Dust_AOD, IWP_data, and Cld_all are your input arrays with shape (3686, 180, 360)
Dust_AOD_filtered = set_extreme_5_percent_to_nan(Dust_AOD)
IWP_all_filtered = set_extreme_5_percent_to_nan(IWP_data)
Cld_all_filtered = set_extreme_5_percent_to_nan(Cld_all)
PC_all_filtered = set_extreme_5_percent_to_nan(PC_all)

#########################################
##### start seperate time test ##########
#########################################

# Plot global spatial distributio
def plot_global_spatial_distribution(
    data,
    var_name,
    title,
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    # Create custom colormap
    cmap = mpl.cm.get_cmap("RdYlBu_r")

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

# This function plots the statistical distribution of the data. 
# The function takes in the name of the variable and title of the plot.
# The function then flattens the data and plots the histogram and KDE.
def plot_statistical_distribution(data, var_name):
    
    # Flatten the data
    flat_data = data.flatten()
    
    # Plot histogram and KDE
    plt.figure(figsize=(12, 6))

    # Plot the histogram
    sns.histplot(flat_data, kde=False, bins=100, color='blue', alpha=0.6, stat='density', label='Histogram')

    # Plot the KDE
    sns.kdeplot(flat_data, color='red', lw=2, label='KDE')

    plt.xlabel(var_name)
    plt.ylabel('Probability Density of ' + var_name)
    plt.title('Probability Density Function (PDF)')
    plt.legend()
    plt.show()


# Plot the global spatial distribution of the data
plot_global_spatial_distribution(
    data=np.nanmean(Dust_AOD, axis=0),
    var_name="Dust AOD",
    title="Dust AOD",
)

plot_global_spatial_distribution(
    data=np.nanmean(IWP_data, axis=0),
    var_name="IWP",
    title="IWP",
)

plot_global_spatial_distribution(
    data=np.nanmean(PC_all, axis=0),
    var_name="PC1",
    title="PC1",
)

# Plot the cld data
plot_global_spatial_distribution(
    data=np.nanmean(Cld_all, axis=0),
    var_name="HCF",
    title="HCF",
)

plot_global_spatial_distribution(
    data=np.nanmean(Cld_all, axis=0),
    var_name="IPR",
    title="IPR",
)

plot_global_spatial_distribution(
    data=np.nanmean(Cld_all, axis=0),
    var_name="CEH",
    title="CEH",
)

# Plot the filtered global spatial distribution of the data
plot_global_spatial_distribution(
    data=np.nanmean(Dust_AOD_filtered, axis=0),
    var_name="Dust AOD",
    title="Dust AOD filtered",
)

plot_global_spatial_distribution(
    data=np.nanmean(IWP_all_filtered, axis=0),
    var_name="IWP",
    title="IWP filtered",
)

plot_global_spatial_distribution(
    data=np.nanmean(PC_all_filtered, axis=0),
    var_name="PC1",
    title="PC1 filtered",
)

# Plot the cld data
plot_global_spatial_distribution(
    data=np.nanmean(Cld_all_filtered, axis=0),
    var_name="HCF",
    title="HCF filtered",
)

plot_global_spatial_distribution(
    data=np.nanmean(Cld_all_filtered, axis=0),
    var_name="IPR",
    title="IPR filtered",
)

plot_global_spatial_distribution(
    data=np.nanmean(Cld_all_filtered, axis=0),
    var_name="CEH",
    title="CEH filtered",
)

# Plot the statistical distribution of the data
plot_statistical_distribution(data=Dust_AOD, var_name='Dust AOD')

plot_statistical_distribution(data=IWP_data, var_name='IWP')

plot_statistical_distribution(data=PC_all, var_name='PC1')

# Plot the cld data
plot_statistical_distribution(data=Cld_all, var_name='HCF') 

plot_statistical_distribution(data=Cld_all, var_name='IPR') 

plot_statistical_distribution(data=Cld_all, var_name='CEH') 

# Plot filtered data
plot_statistical_distribution(data=Dust_AOD_filtered, var_name='Dust AOD filtered')

plot_statistical_distribution(data=IWP_all_filtered, var_name='IWP filtered')

plot_statistical_distribution(data=PC_all_filtered, var_name='PC1 filtered')

# Plot the cld data
plot_statistical_distribution(data=Cld_all_filtered, var_name='HCF filtered')

plot_statistical_distribution(data=Cld_all_filtered, var_name='IPR filtered')

plot_statistical_distribution(data=Cld_all_filtered, var_name='CEH filtered')

########################################################
##### Plot the mean AOD data to verify the data #########
########################################################

# ------ Segmentation of cloud data within each PC interval ---------------------------------
#### triout for IWP constrain the same time with PC1 gap constrain ####
# first we need to divide IWP data and PC1 data into n intervals
# this step is aimed to create pcolormesh plot for PC1 and IWP data
# Divide 1, IWP data
def generate_filtered_data_for_all_years(
    AOD_data: np.ndarray,
    IWP_data: np.ndarray,
    PC_all: np.ndarray,
    Cld_all: np.ndarray,
    AOD_n: int = 5,
    IWP_n: int = 50,
    PC_n: int = 50,
):
    """
    Generate filtered data for all years based on
    input AOD, IWP, PC, and cloud data.

    Parameters
    ----------
    AOD_data : np.ndarray
    Array of Dust Aerosol Optical Depth (AOD) data.
    IWP_data : np.ndarray
    Array of Ice Water Path (IWP) data.
    PC_all : np.ndarray
    Array of principal component (PC) data.
    Cld_data_all : np.ndarray
    Array of cloud data.
    AOD_n : int, optional
    Number of AOD data bins to divide the data into. Default is 5.
    IWP_n : int, optional
    Number of IWP data bins to divide the data into. Default is 50.
    PC_n : int, optional
    Number of PC data bins to divide the data into. Default is 50.

    Returns:
    tuple
    A tuple of the following elements:
    - Cld_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
    Array of filtered cloud data matched with PC, IWP, AOD data.
    - PC_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
    Array of filtered PC data matched with PC, IWP, AOD data.
    - AOD_gap : np.ndarray
    Array of AOD data bins.
    - IWP_gap : np.ndarray
    Array of IWP data bins.
    - PC_gap : np.ndarray
    Array of PC data bins.
    """
    divide_IWP = DividePCByDataVolume(
        dataarray_main=IWP_data,
        n=IWP_n,
    )
    IWP_gap = divide_IWP.main_gap()

    # Divide 2, PC1 data
    divide_PC = DividePCByDataVolume(
        dataarray_main=PC_all,
        n=PC_n,
    )
    PC_gap = divide_PC.main_gap()

    # Divide 3, Dust AOD data
    # Divide AOD data as well
    divide_AOD = DividePCByDataVolume(
        dataarray_main=AOD_data,
        n=AOD_n,
    )
    AOD_gap = divide_AOD.main_gap()

    filter_cld_under_AOD_IWP_PC_constrain = (
        Filter_data_fit_PC1_gap_IWP_AOD_constrain(
            lat=[i for i in range(180)],
            lon=[i for i in range(360)],
        )
    )

    # Now we can filter the CLd and PC1 data into pieces
    # Based on AOD, IWP, PC1 gap we just created
    # Shape is (AOD_bin, IWP_bin, PC_bin, lat, lon)
    (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        PC_match_PC_gap_IWP_AOD_constrain_mean,
    ) = filter_cld_under_AOD_IWP_PC_constrain.Filter_data_fit_gap(
        Cld_data=Cld_all.reshape(-1, 180, 360),
        PC_data=PC_all.reshape(-1, 180, 360),
        IWP_data=IWP_data.reshape(-1, 180, 360),
        AOD_data=AOD_data.reshape(-1, 180, 360),
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
        AOD_gap=AOD_gap,
    )

    return (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        PC_match_PC_gap_IWP_AOD_constrain_mean,
        AOD_gap,
        IWP_gap,
        PC_gap,
    )

def save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean: np.ndarray,
    AOD_gap: np.ndarray,
    IWP_gap: np.ndarray,
    PC_gap: np.ndarray,
    AOD_name: str = "Dust_AOD",
    save_path: str = "/RAID01/data/Filtered_data/",
):
    """
    Save the fitted data as netcdf file.

    Parameters
    ----------
    Cld_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
        Mean cloud data with matching PC1 gap, IWP, and AOD constraint.
    PC_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
        Mean PC1 data with matching PC1 gap, IWP, and AOD constraint.
    AOD_gap : np.ndarray
        AOD gap.
    IWP_gap : np.ndarray
        IWP gap.
    PC_gap : np.ndarray
        PC gap.
    AOD_name : str, optional
        Name of AOD data, by default "Dust_AOD".
    save_path : str, optional
        Path to save netcdf file, by default "/RAID01/data/Filtered_data/".
    """
    # Save the fitted data as netcdf file
    Cld_match_PC_gap_IWP_AOD_constrain_mean = xr.DataArray(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        dims=["AOD_bin", "IWP_bin", "PC_bin", "lat", "lon"],
        coords={
            "AOD_bin": AOD_gap[1:],
            "IWP_bin": IWP_gap[1:],
            "PC_bin": PC_gap[1:],
            "lat": np.arange(180),
            "lon": np.arange(360),
        },
    )
    Cld_match_PC_gap_IWP_AOD_constrain_mean.to_netcdf(
        save_path
        + "Cldeff_hgth_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_"
        + AOD_name
        + "_pristine_6_aod_gaps.nc"
    )

# Dust AOD
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD,
    IWP_data=IWP_data,
    PC_all=PC_all,
    Cld_all=Cld_all,
    AOD_n=6,
    IWP_n=30,
    PC_n=30,
)

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
)
