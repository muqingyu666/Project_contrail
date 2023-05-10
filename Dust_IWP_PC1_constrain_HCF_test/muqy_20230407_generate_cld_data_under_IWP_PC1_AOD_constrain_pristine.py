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

import dask.array as da
import matplotlib as mpl
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
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
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name="Cldarea")

PC_years = []
Cld_years = []
IWp_years = []
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
data_merra2_2010_2020 = xr.open_dataset(
    "/RAID01/data/merra2/merra_2_daily_2010_2020.nc"
)

# Set values less than 0 to 0
data_merra2_2010_2020 = data_merra2_2010_2020.where(
    data_merra2_2010_2020 >= 0, 0
)

# Change the longitude values from -180 to 180 to 0 to 360
# Find the index where the longitude values change from negative to positive
lon_0_index = (data_merra2_2010_2020.lon >= 0).argmax().values

# Slice the dataset into two parts
left_side = data_merra2_2010_2020.isel(lon=slice(0, lon_0_index))
right_side = data_merra2_2010_2020.isel(lon=slice(lon_0_index, None))

# Change the longitude values to the 0 to 360 range
left_side = left_side.assign_coords(lon=(left_side.lon + 360) % 360)
right_side = right_side.assign_coords(lon=(right_side.lon + 360) % 360)

# Concatenate the left and right sides
data_merra2_2010_2020_new_lon = xr.concat(
    [right_side, left_side], dim="lon"
)

# Sort the dataset by the new longitude values
data_merra2_2010_2020_new_lon = data_merra2_2010_2020_new_lon.sortby(
    "lon"
)

# # Extract Dust aerosol data from all data
# Dust_AOD = data_merra2_2010_2020_new_lon["DUEXTTAU"].values.reshape(
#     3696, 180, 360
# )
# # Dust_mass = data_merra2_2010_2020["DUCMASS"].values.reshape(
# #     3696, 180, 360
# # )

SO4_AOD = data_merra2_2010_2020_new_lon["SUEXTTAU"].values.reshape(
    3696, 180, 360
)
# # SO4_mass = data_merra2_2010_2020["SO4SMASS"].values.reshape(
# #     3696, 180, 360
# # )

# # # Extract Dust aerosol data from all data
# OC_AOD = data_merra2_2010_2020_new_lon["OCEXTTAU"].values.reshape(
#     3696, 180, 360
# )
# # OC_mass = data_merra2_2010_2020["OCCMASS"].values.reshape(
# #     3696, 180, 360
# # )

# use the 2010-2020 PC1 only
PC_all = PC_all[-11:].reshape(3696, 180, 360)
Cld_all = Cld_all.reshape(3696, 180, 360)
IWP_data = IWP_data.reshape(3696, 180, 360)

# convert the data type to float32
PC_all = PC_all.astype(np.float32)
Cld_all = Cld_all.astype(np.float32)
IWP_data = IWP_data.astype(np.float32)
SO4_AOD = SO4_AOD.astype(np.float32)

#########################################
##### start seperate time test ##########
#########################################


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
    Cld_data_all: np.ndarray,
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
        Cld_data=Cld_data_all.reshape(-1, 180, 360),
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
        + "Cldarea_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_"
        + AOD_name
        + "_pristine_4_aod_gaps.nc"
    )


# Dust AOD
# Load the data
# (
#     Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
#     PC_match_PC_gap_IWP_AOD_constrain_mean_Dust,
#     AOD_gap,
#     IWP_gap,
#     PC_gap,
# ) = generate_filtered_data_for_all_years(
#     AOD_data=Dust_AOD,
#     IWP_data=IWP_data,
#     PC_all=PC_all,
#     Cld_data_all=Cld_all,
#     AOD_n=4,
#     IWP_n=30,
#     PC_n=30,
# )

# # Save the filtered data
# save_filtered_data_as_nc(
#     Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
#     AOD_gap,
#     IWP_gap,
#     PC_gap,
#     AOD_name="Dust_AOD",
# )

# # SO4 AOD
# # Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_SO4,
    PC_match_PC_gap_IWP_AOD_constrain_mean_SO4,
    AOD_gap,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=SO4_AOD,
    IWP_data=IWP_data,
    PC_all=PC_all,
    Cld_data_all=Cld_all,
    AOD_n=6,
    IWP_n=40,
    PC_n=40,
)

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_SO4,
    AOD_gap,
    IWP_gap,
    PC_gap,
    AOD_name="SO4_AOD",
)

# # OC AOD
# # Load the data
# (
#     Cld_match_PC_gap_IWP_AOD_constrain_mean_OC,
#     PC_match_PC_gap_IWP_AOD_constrain_mean_OC,
#     AOD_gap,
#     IWP_gap,
#     PC_gap,
# ) = generate_filtered_data_for_all_years(
#     AOD_data=OC_AOD,
#     IWP_data=IWP_data,
#     PC_all=PC_all,
#     Cld_data_all=Cld_all,
#     AOD_n=6,
#     IWP_n=40,
#     PC_n=40,
# )

# # Save the filtered data
# save_filtered_data_as_nc(
#     Cld_match_PC_gap_IWP_AOD_constrain_mean_OC,
#     AOD_gap,
#     IWP_gap,
#     PC_gap,
#     AOD_name="OC_AOD",
# )

