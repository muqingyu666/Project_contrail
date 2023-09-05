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
# --------------------------------------------------------------- #
# import modules

from typing import Union

import matplotlib as mpl
from matplotlib.colors import ListedColormap
import numpy as np
import xarray as xr

# ----------  importing dcmap from my util ----------#
from muqy_20220413_util_useful_functions import dcmap as dcmap
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)

# --------- import done ------------
# --------- Plot style -------------
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")
mpl.rc("font", family="Times New Roman")

# ---------- Read PCA&CLD data from netcdf file --------

(
    PC_best_corr,
    _,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/1990_2020_250_hPa_vars_250_300_Instab_PC1_no_antarc.nc",
    CERES_Cld_dataset_name="Cldicerad",
)

# --------------------------------------------------------------- #
# ------ Read in CERES SSF data and filter anormal data --------- #
# --------------------------------------------------------------- #
# read in CERES SSF data

# Specify the output path
CERES_SSF_28day = xr.open_dataset(
    "/RAID01/data/Cld_data/CERES_SSF_data_2005_2020_28_days.nc"
)

CERES_SSF_HCF = CERES_SSF_28day[
    "cldarea_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_ice_HCF = CERES_SSF_28day[
    "cldarea_ice_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_CEP = CERES_SSF_28day[
    "cldpress_eff_high_daynight_daily"
].values.reshape(-1, 180, 360)

CERES_SSF_IPR_12 = CERES_SSF_28day[
    "cldicerad12_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_IPR_21 = CERES_SSF_28day[
    "cldicerad21_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_IPR_37 = CERES_SSF_28day[
    "cldicerad37_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_COD = CERES_SSF_28day[
    "cldtau_high_daynight_daily"
].values.reshape(-1, 180, 360)

CERES_SSF_IWP = CERES_SSF_28day[
    "iwp37_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_CTP = CERES_SSF_28day[
    "cldpress_top_high_daynight_daily"
].values.reshape(-1, 180, 360)


# use the 2010-2020 PC1 only
PC_best_corr = PC_best_corr.reshape(31, 12, 28, 150, 360)[
    -16:, :, :, :, :
]
PC_best_corr = PC_best_corr.reshape(-1, 150, 360)

# --------------------------------------------------------------- #
# ------------------ Read the MERRA2 dust data ------------------ #
# --------------------------------------------------------------- #
# Implementation for MERRA2 dust AOD
# extract the data from 2010 to 2014 like above
data_merra2_2010_2020_new_lon = xr.open_dataset(
    "/RAID01/data/merra2/merra2_2005_2020_new_lon.nc"
)

# # Extract Dust aerosol data from all data
Dust_AOD = data_merra2_2010_2020_new_lon["DUEXTTAU"].values.reshape(
    5376, 180, 360
)


# Set the largest and smallest 5% of the data to nan
def set_extreme_percent_to_nan(
    threshold: float, data_array: np.ndarray
) -> Union[np.ndarray, None]:
    """
    Set the largest and smallest percentage of the data to NaN.

    Args:
        threshold (float):
            The percentage of the data to set to NaN.
        data_array (numpy.ndarray):
            The input data array.

    Returns:
        numpy.ndarray or None: A copy of the input data array with the largest
        and smallest percentage of the data set to NaN, or None if the
        input data array is empty.

    Raises:
        None.

    Notes:
        The function sets the largest and smallest percentage of the data to NaN.
        It takes as input the percentage of the data to set to NaN and the input data array.
        The function returns a copy of the input data array with the largest
        and smallest percentage of the data set to NaN. If the input data array is empty,
        the function returns None.

    Examples:
        >>> import numpy as np
        >>> data_array = np.random.rand(10, 10)
        >>> threshold = 5.0
        >>> data_array_filtered = set_extreme_percent_to_nan(threshold, data_array)
    """
    # Make a copy of the data array
    data_array_filtered = np.copy(data_array)

    # Calculate the threshold values for the largest and smallest 5%
    try:
        lower_threshold = np.nanpercentile(data_array, threshold)
    except IndexError:
        print("ERROR: Data array is empty.")
        return None
    else:
        upper_threshold = np.nanpercentile(
            data_array, 100 - threshold
        )

    # Set the largest and smallest 5% of the data array to nan
    data_array_filtered[
        data_array_filtered < lower_threshold
    ] = np.nan
    data_array_filtered[
        data_array_filtered > upper_threshold
    ] = np.nan

    return data_array_filtered


# Filter out extreme 2.5% AOD data to avoid the influence of dust origin
Dust_AOD_filtered = set_extreme_percent_to_nan(
    threshold=1, data_array=Dust_AOD
)

# Filter out extreme 1% IWP data to avoid extreme large IWP values
IWP_data_filtered = set_extreme_percent_to_nan(
    threshold=1, data_array=CERES_SSF_IWP
)

# delete antarctica region
Dust_AOD_filtered = Dust_AOD_filtered[:, 30:, :]
IWP_data_filtered = IWP_data_filtered[:, 30:, :]
CERES_SSF_HCF = CERES_SSF_HCF[:, 30:, :]
CERES_SSF_ice_HCF = CERES_SSF_ice_HCF[:, 30:, :]
CERES_SSF_IPR_37 = CERES_SSF_IPR_37[:, 30:, :]
CERES_SSF_IWP = CERES_SSF_IWP[:, 30:, :]
CERES_SSF_CTP = CERES_SSF_CTP[:, 30:, :]
CERES_SSF_CEP = CERES_SSF_CEP[:, 30:, :]
CERES_SSF_COD = CERES_SSF_COD[:, 30:, :]

#########################################
##### start seperate time test ##########
#########################################

PC_all = PC_best_corr.reshape(-1, 150, 360)

# PC_all = PC_best_corr_spatial.reshape(-1, 150, 360)
# region

# ------ Segmentation of cloud data within each PC interval ---------------------------------
### triout for IWP constrain the same time with PC1 gap constrain ####
# first we need to divide IWP data into n intervals
# what i think is 6 parts of IWP, 0-1, 1-2, 2-3, 3-4, 4-5, 5-6
# Filtered version of IWP data and AOD data
# ------------------ Divide IWP data into 6 intervals ------------------
# divideIWP = DividePCByDataVolume(
#     dataarray_main=IWP_data_filtered,
#     n=10,
# )
# IWP_gap = divideIWP.main_gap()

IWP_gap = np.array([0.5, 5, 50, 100, 500])

# Filter the data by AOD gap
divideAOD = DividePCByDataVolume(
    dataarray_main=Dust_AOD_filtered,
    n=4,
)
AOD_gap = divideAOD.main_gap()

PC1_gap = np.array([-3, 0, 3, 6])

filter_data_fit_PC1_gap_IWP_constrain = (
    Filter_data_fit_PC1_gap_plot_IWP_AOD_constrain(
        start=-3,
        end=6,
        gap=0.06,
        lat=np.arange(150),
        lon=np.arange(360),
    )
)

# empty array for storing the data
HCF_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))
HCF_ice_match_PC_gap_IWP_constrain_mean = np.empty(
    (4, 150, 150, 360)
)
IPR_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))
CTP_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))
COD_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))
CEP_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))

HCF_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))
HCF_ice_match_PC_gap_AOD_constrain_mean = np.empty(
    (4, 150, 150, 360)
)
IPR_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))
CTP_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))
COD_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))
CEP_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))

# Main loop for the data
for i in range(0, len(IWP_gap) - 1):
    # HCF data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    HCF_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_HCF.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    HCF_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_HCF.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )

    # HCF data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    HCF_ice_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_ice_HCF.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    HCF_ice_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_ice_HCF.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )

    # IPR data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    IPR_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_IPR_37.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    IPR_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_IPR_37.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )

    # CTP data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    CTP_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_CTP.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    CTP_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_CTP.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )

    # CEP data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    CEP_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_CEP.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    CEP_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_CEP.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )

    # COD data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    COD_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_COD.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    COD_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_COD.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )


# save all the data to netcdf4 format

# Create a Dataset with DataArrays
ds = xr.Dataset(
    {
        "HCF_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            HCF_match_PC_gap_IWP_constrain_mean,
        ),
        "HCF_ice_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            HCF_ice_match_PC_gap_IWP_constrain_mean,
        ),
        "IPR_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            IPR_match_PC_gap_IWP_constrain_mean,
        ),
        "CTP_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            CTP_match_PC_gap_IWP_constrain_mean,
        ),
        "HCF_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            HCF_match_PC_gap_AOD_constrain_mean,
        ),
        "HCF_ice_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            HCF_ice_match_PC_gap_AOD_constrain_mean,
        ),
        "IPR_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            IPR_match_PC_gap_AOD_constrain_mean,
        ),
        "CTP_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            CTP_match_PC_gap_AOD_constrain_mean,
        ),
        "CEP_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            COD_match_PC_gap_IWP_constrain_mean,
        ),
        "CEP_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            CEP_match_PC_gap_AOD_constrain_mean,
        ),
        "COD_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            COD_match_PC_gap_IWP_constrain_mean,
        ),
        "COD_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            COD_match_PC_gap_AOD_constrain_mean,
        ),
    },
)

# Save to a netCDF file
ds.to_netcdf(
    "/RAID01/data/Filtered_data/CERES_SSF_under_diverse_IWP_AOD_conditions_filtered_1_percent_no_antarctica_250hPa_PC1_4_gaps.nc"
    # "/RAID01/data/Filtered_data/CERES_SSF_under_diverse_IWP_AOD_conditions_filtered_1_percent_no_antarctica_best_corr_spatial.nc"
)

# endregion

CERES_SSF_cld_under_diverser_IWP_AOD_conditions = xr.open_dataset(
    "/RAID01/data/Filtered_data/CERES_SSF_under_diverse_IWP_AOD_conditions_filtered_1_percent_no_antarctica_250hPa_PC1_4_gaps.nc"
)

# Extract the data
HCF_match_PC_gap_IWP_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "HCF_IWP_bin"
    ].values
)
IPR_match_PC_gap_IWP_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "IPR_IWP_bin"
    ].values
)
CTP_match_PC_gap_IWP_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "CTP_IWP_bin"
    ].values
)
CEP_match_PC_gap_IWP_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "CEP_IWP_bin"
    ].values
)
HCF_match_PC_gap_AOD_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "HCF_AOD_bin"
    ].values
)
IPR_match_PC_gap_AOD_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "IPR_AOD_bin"
    ].values
)
CTP_match_PC_gap_AOD_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "CTP_AOD_bin"
    ].values
)
CEP_match_PC_gap_AOD_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "CEP_AOD_bin"
    ].values
)
COD_match_PC_gap_IWP_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "COD_IWP_bin"
    ].values
)

#######################################################################
## Use box plot to quantify the cld distribution within each PC interval
## Under the IWP constrain ####
#######################################################################

# ------------------------------------------------------------
# Plot 3D error fill plot for different IWP conditions
# ------------------------------------------------------------


def create_colormap_with_nan(
    cmap_name: str, nan_color: str = "silver"
) -> ListedColormap:
    """
    Create a colormap with NaN values colored with a self-defined color.

    Args:
        cmap_name (str): The name of the colormap to use.
        nan_color (str, optional): The color to use for NaN values. Defaults to "silver".

    Returns:
        ListedColormap: A colormap with NaN values colored with the specified color.

    Notes:
        The function creates a colormap with NaN values colored with a self-defined color. It uses the specified colormap name to create a colormap, and then sets the color for NaN values to the specified color. The function returns the resulting colormap.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> cmap = create_colormap_with_nan("viridis", "gray")
        >>> plt.imshow([[1, 2], [3, np.nan]], cmap=cmap)
    """
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))
    cmap_with_nan = ListedColormap(colors)
    cmap_with_nan.set_bad(nan_color)
    return cmap_with_nan


def plot_3d_colored_IWP_PC1_AOD_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    high_cloud_amount_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    savefig_str,
    aod_range,
    vmin,
    vmax,
    AOD_bin_values,
    PC1_bin_values,
    IWP_bin_values,
    cmap="Spectral_r",
) -> None:
    """
    Create a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval.

    Args:
        Cld_match_PC_gap_IWP_AOD_constrain_mean (numpy.array): Cloud data under each restriction, shape (AOD_bin, IWP_bin, PC_bin, lat, lon)
        high_cloud_amount_mean (numpy.array): Mean high cloud amount for each AOD interval, shape (AOD_bin,)
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        colobar_label (str): Label for the color bar
        savefig_str (str): String for saving the figure
        aod_range (tuple): Tuple defining the start and end indices of the AOD cases to plot
        vmin (float): Minimum value for the color scale
        vmax (float): Maximum value for the color scale
        cmap (str, optional): The name of the colormap to use. Defaults to "Spectral_r".

    Returns:
        None: The function displays the plot using matplotlib.pyplot.show().

    Raises:
        ValueError: If the input arrays are not of the expected shape.

    Notes:
        The function creates a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval. It takes as input the cloud data under each restriction, the mean high cloud amount for each AOD interval, the labels for the x, y, and z axes, the label for the color bar, the string for saving the figure, the range of AOD cases to plot, the minimum and maximum values for the color scale, and the name of the colormap to use. The function displays the plot using matplotlib.pyplot.show().

    Examples:
        >>> import numpy as np
        >>> Cld_match_PC_gap_IWP_AOD_constrain_mean = np.random.rand(3, 4, 5, 6, 7)
        >>> high_cloud_amount_mean = np.random.rand(3)
        >>> xlabel = "PC1"
        >>> ylabel = "IWP"
        >>> zlabel = "AOD"
        >>> colobar_label = "High cloud amount"
        >>> savefig_str = "3d_colored_IWP_PC1_AOD_min_max_version.png"
        >>> aod_range = (1, 3)
        >>> vmin = 0.0
        >>> vmax = 1.0
        >>> cmap = "Spectral_r"
        >>> plot_3d_colored_IWP_PC1_AOD_min_max_version(
        ...     Cld_match_PC_gap_IWP_AOD_constrain_mean,
        ...     high_cloud_amount_mean,
        ...     xlabel,
        ...     ylabel,
        ...     zlabel,
        ...     colobar_label,
        ...     savefig_str,
        ...     aod_range,
        ...     vmin,
        ...     vmax,
        ...     cmap
        ... )
    """
    fig = plt.figure(figsize=(13, 13), dpi=400)
    ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    cmap_with_nan = create_colormap_with_nan(cmap)

    for aod_num in range(aod_range[0], aod_range[1]):
        Z = aod_num * np.ones_like(X)
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

    # Define the number of ticks you want for y and z axes (for example 5)
    n_ticks = 5

    # Define tick positions for each axis

    xticks = np.arange(
        aod_range[0], aod_range[1]
    )  # Only three AOD bins per 3D image
    yticks = np.linspace(0, PC_bin - 1, n_ticks, dtype=int)
    zticks = np.linspace(0, IWP_bin - 1, n_ticks, dtype=int)

    # Set tick positions
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    # Set tick labels using the provided bin values and format them with three decimal places
    ax.set_xticklabels(
        ["{:.3f}".format(val) for val in AOD_bin_values[xticks]]
    )
    ax.set_yticklabels(
        ["{:.3f}".format(val) for val in PC1_bin_values[yticks]]
    )
    ax.set_zticklabels(
        ["{:.3f}".format(val) for val in IWP_bin_values[zticks]]
    )

    # set labels and distance from axis
    ax.set_xlabel(xlabel, labelpad=27, fontsize=16.5)
    ax.set_ylabel(ylabel, labelpad=27, fontsize=16.5)
    ax.set_zlabel(zlabel, labelpad=27, fontsize=16.5)

    # set tick label distance from axis
    ax.tick_params(axis="x", which="major", pad=13, labelsize=14)
    ax.tick_params(axis="y", which="major", pad=13, labelsize=14)
    ax.tick_params(axis="z", which="major", pad=13, labelsize=14)

    ax.grid(False)

    m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r)
    m.set_cmap(cmap_with_nan)
    m.set_array(high_cloud_amount_mean)
    m.set_clim(vmin, vmax)
    cb = fig.colorbar(
        m, shrink=0.3, aspect=9, pad=0.01, label=colobar_label
    )

    # set colorbar tick label size
    cb.set_label(colobar_label, fontsize=16.5)
    cb.ax.tick_params(labelsize=14)

    ax.view_init(elev=10, azim=-65)
    ax.dist = 12

    plt.savefig(savefig_str)
    plt.show()

    AOD_bin_values,
    PC1_bin_values,
    IWP_bin_values,


def error_fill_3d_plot(
    data_list,
    xlabel,
    ylabel,
    zlabel,
    legend_labels,
    savefig_str,
    zmin: float = None,
    zmax: float = None,
):
    """
    Create a 3D error fill plot with different colors for each IWP condition

    Args:
        data_list (list): List of data arrays for different IWP conditions
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        legend_labels (list): List of legend labels for each IWP condition
        savefig_str (str): String for saving the figure
    """
    # Create a figure instance
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Loop through data_list to create error fill plots for each IWP condition
    for i, data in enumerate(data_list):
        # Input array must be in shape of (PC1_gap, lat, lon)
        # reshape data to (PC1_gap, lat*lon)
        data = data.reshape(data.shape[0], -1)

        # Calculate mean and std of each PC1 interval
        data_y = np.round(np.nanmean(data, axis=1), 3)
        data_x = np.round(np.arange(-2.5, 5.5, 0.05), 3)
        data_std = np.nanstd(data, axis=1)

        # Create up and down limit of error bar
        data_up = data_y + data_std
        data_down = data_y - data_std

        # Create IWP condition coordinate on y-axis
        iwp_condition = np.ones_like(data_x) * i

        # Plot the mean line and fill between up and down limits for each IWP condition
        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=colors[i % len(colors)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=colors[i % len(colors)],
                alpha=0.2,
            ),
            zs=i,
            zdir="y",
        )

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_zlim(zmin=zmin, zmax=zmax)

    # Add legend
    custom_lines = [
        plt.Line2D([0], [0], color=colors[i % len(colors)], lw=3)
        for i in range(len(data_list))
    ]
    ax.legend(custom_lines, legend_labels)

    # Set the viewing angle
    ax.view_init(elev=15, azim=-18)
    ax.dist = 12

    # Turn off the grid lines
    ax.grid(False)

    # Save figure
    os.makedirs(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/",
        exist_ok=True,
    )
    plt.savefig(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/"
        + savefig_str,
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()


def error_fill_3d_plot_no_legend_3f(
    data_list,
    xlabel,
    ylabel,
    zlabel,
    IWP_or_AOD_bin_values,
    savefig_str,
    zmin: float = None,
    zmax: float = None,
):
    """
    Create a 3D error fill plot with different colors for each IWP condition

    Args:
        data_list (list): List of data arrays for different IWP conditions
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        legend_labels (list): List of legend labels for each IWP condition
        savefig_str (str): String for saving the figure
    """
    IWP_or_AOD_bin = IWP_or_AOD_bin_values.shape[0]

    # Create a figure instance
    fig = plt.figure(
        figsize=(6, 6), constrained_layout=True, dpi=300
    )
    ax = fig.add_subplot(111, projection="3d")

    colors = [
        (255 / 255, 157 / 255, 0 / 255),
        (255 / 255, 126 / 255, 0 / 255),
        (255 / 255, 76 / 255, 0 / 255),
        (255 / 255, 11 / 255, 0 / 255),
        (181 / 255, 0 / 255, 0 / 255),
        (160 / 255, 0 / 255, 0 / 255),
        (95 / 255, 0 / 255, 0 / 255),
    ]

    # Loop through data_list to create error fill plots for each IWP condition
    for i, data in enumerate(data_list):
        # Input array must be in shape of (PC1_gap, lat, lon)
        # reshape data to (PC1_gap, lat*lon)
        data = data.reshape(data.shape[0], -1)

        # Calculate mean and std of each PC1 interval
        data_y = np.round(np.nanmean(data, axis=1), 3)
        data_x = np.round(np.arange(-3, 6, 0.03), 3)
        data_std = np.nanstd(data, axis=1)

        # Create up and down limit of error bar
        data_up = data_y + data_std
        data_down = data_y - data_std

        # Create IWP condition coordinate on y-axis
        iwp_condition = np.ones_like(data_x) * i

        # Plot the mean line and fill between up and down limits for each IWP condition
        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=colors[i % len(colors)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=colors[i % len(colors)],
                alpha=0.25,
            ),
            zs=i,
            zdir="y",
        )

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_zlim(zmin=zmin, zmax=zmax)

    # Define the number of ticks you want for y and z axes (for example 5)
    n_ticks = 7

    # Define tick positions for each axis

    yticks = np.linspace(0, IWP_or_AOD_bin - 1, n_ticks, dtype=int)

    # Set tick positions
    ax.set_yticks(yticks)

    # Set tick labels using the provided bin values and format them with three decimal places
    ax.set_yticklabels(
        [
            "{:.3f}".format(val)
            for val in IWP_or_AOD_bin_values[yticks]
        ]
    )

    # Set the viewing angle
    ax.view_init(elev=15, azim=-13)
    ax.dist = 11

    # Turn off the grid lines
    ax.grid(False)

    # Save figure
    os.makedirs(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/",
        exist_ok=True,
    )
    plt.savefig(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/"
        + savefig_str,
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()


COLORS = [
    (90 / 255, 199 / 255, 183 / 255),
    (39 / 255, 171 / 255, 193 / 255),
    (0 / 255, 147 / 255, 196 / 255),
    (0 / 255, 126 / 255, 174 / 255),
    (0 / 255, 105 / 255, 151 / 255),
    (0 / 255, 48 / 255, 91 / 255),
    (0 / 255, 41 / 255, 84 / 255),
]

XTICK_LABELS = {-4: -3, -1: 0, 2: 3, 5: 6}


def error_fill_3d_plot_no_legend_1f(
    data_list,
    xlabel,
    ylabel,
    zlabel,
    IWP_or_AOD_bin_values,
    PC1_bin_values,
    savefig_str,
    savefig_path,
    zmin: float = None,
    zmax: float = None,
):
    fig = plt.figure(
        figsize=(6, 6), constrained_layout=True, dpi=300
    )
    ax = fig.add_subplot(111, projection="3d")

    for i, data in enumerate(data_list):
        data = data.reshape(data.shape[0], -1)
        data_y = np.round(np.nanmean(data, axis=1), 3)
        data_x = np.round(np.arange(-3, 6, 0.06), 3)
        data_std = np.nanstd(data, axis=1)
        data_up = data_y + data_std
        data_down = data_y - data_std
        iwp_condition = np.ones_like(data_x) * i

        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=COLORS[i % len(COLORS)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=COLORS[i % len(COLORS)],
                alpha=0.35,
            ),
            zs=i,
            zdir="y",
        )

    yticks = np.linspace(
        0, len(IWP_or_AOD_bin_values) - 1, 7, dtype=int
    )
    xticks = list(XTICK_LABELS.keys())

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        yticks=yticks,
        xticks=xticks,
        zlim=(zmin, zmax),
    )
    ax.set_yticklabels(
        [
            "{:.1f}".format(val)
            for val in IWP_or_AOD_bin_values[yticks]
        ]
    )
    ax.set_xticklabels([XTICK_LABELS[val] for val in xticks])
    ax.set_xlim(-4.3, 5.3)

    ax.view_init(elev=26, azim=-20)
    ax.dist = 12
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.grid(
        True,
        color="silver",
        linestyle="-.",
        linewidth=0.3,
        alpha=0.4,
    )

    plt.savefig(
        os.path.join(savefig_path, savefig_str),
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()


COLORS = [
    (90 / 255, 199 / 255, 183 / 255),
    (0 / 255, 147 / 255, 196 / 255),
    (0 / 255, 105 / 255, 151 / 255),
    (0 / 255, 41 / 255, 84 / 255),
]


def error_fill_3d_plot_no_legend_1f_4_gaps(
    data_list,
    xlabel,
    ylabel,
    zlabel,
    IWP_or_AOD_bin_values,
    elev_angle,
    azim_angle,
    savefig_str,
    savefig_path,
    annotations: dict,
    zmin: float = None,
    zmax: float = None,
):
    fig = plt.figure(
        figsize=(6, 6), constrained_layout=True, dpi=300
    )
    ax = fig.add_subplot(111, projection="3d")

    for i, data in enumerate(data_list):
        data = data.reshape(data.shape[0], -1)
        data_y = np.round(np.nanmean(data, axis=1), 3)
        data_x = np.round(np.arange(-3, 6, 0.06), 3)
        data_std = np.nanstd(data, axis=1)
        data_up = data_y + data_std
        data_down = data_y - data_std
        iwp_condition = np.ones_like(data_x) * i

        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=COLORS[i % len(COLORS)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=COLORS[i % len(COLORS)],
                alpha=0.35,
            ),
            zs=i,
            zdir="y",
        )

    # Annotate figure
    for key, value in annotations.items():
        fig.text(*key, value, va='top',fontsize=21)
        
    yticks = np.linspace(
        0, len(IWP_or_AOD_bin_values) - 1, 7, dtype=int
    )
    xticks = list(XTICK_LABELS.keys())

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        yticks=yticks,
        xticks=xticks,
        zlim=(zmin, zmax),
    )
    ax.set_yticklabels(
        [
            "{:.1f}".format(val)
            for val in IWP_or_AOD_bin_values[yticks]
        ]
    )
    ax.set_xticklabels([XTICK_LABELS[val] for val in xticks])
    ax.set_xlim(-4.3, 5.3)

    ax.view_init(elev=elev_angle, azim=azim_angle)
    ax.dist = 12
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.grid(
        True,
        color="silver",
        linestyle="-.",
        linewidth=0.3,
        alpha=0.4,
    )

    plt.savefig(
        os.path.join(savefig_path, savefig_str),
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()



COLORS = [
    (255 / 255, 157 / 255, 0 / 255),
    (255 / 255, 76 / 255, 0 / 255),
    (181 / 255, 0 / 255, 0 / 255),
    (95 / 255, 0 / 255, 0 / 255),
]


def error_fill_3d_plot_no_legend_3f_4_gaps(
    data_list,
    xlabel,
    ylabel,
    zlabel,
    IWP_or_AOD_bin_values,
    elev_angle,
    azim_angle,
    savefig_str,
    savefig_path,
    annotations: dict,
    zmin: float = None,
    zmax: float = None,
):
    fig = plt.figure(
        figsize=(6, 6), constrained_layout=True, dpi=300
    )
    ax = fig.add_subplot(111, projection="3d")

    for i, data in enumerate(data_list):
        data = data.reshape(data.shape[0], -1)
        data_y = np.round(np.nanmean(data, axis=1), 3)
        data_x = np.round(np.arange(-3, 6, 0.06), 3)
        data_std = np.nanstd(data, axis=1)
        data_up = data_y + data_std
        data_down = data_y - data_std
        iwp_condition = np.ones_like(data_x) * i

        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=COLORS[i % len(COLORS)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=COLORS[i % len(COLORS)],
                alpha=0.35,
            ),
            zs=i,
            zdir="y",
        )

    # Annotate figure
    for key, value in annotations.items():
        fig.text(*key, value, va='top',fontsize=21)

    yticks = np.linspace(
        0, len(IWP_or_AOD_bin_values) - 1, 7, dtype=int
    )
    xticks = list(XTICK_LABELS.keys())

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        yticks=yticks,
        xticks=xticks,
        zlim=(zmin, zmax),
    )
    ax.set_yticklabels(
        [
            "{:.3f}".format(val)
            for val in IWP_or_AOD_bin_values[yticks]
        ]
    )
    ax.set_xticklabels([XTICK_LABELS[val] for val in xticks])
    ax.set_xlim(-4.3, 5.3)

    ax.view_init(elev=elev_angle, azim=azim_angle)
    ax.dist = 12
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.grid(
        True,
        color="silver",
        linestyle="-.",
        linewidth=0.3,
        alpha=0.4,
    )

    plt.savefig(
        os.path.join(savefig_path, savefig_str),
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()


# Call the error_fill_3d_plot function

# Create a list of data arrays for different IWP conditions
# region
# HCF_list_IWP = [
#     HCF_match_PC_gap_IWP_constrain_mean[0],
#     HCF_match_PC_gap_IWP_constrain_mean[1],
#     HCF_match_PC_gap_IWP_constrain_mean[2],
#     HCF_match_PC_gap_IWP_constrain_mean[3],
#     HCF_match_PC_gap_IWP_constrain_mean[4],
#     HCF_match_PC_gap_IWP_constrain_mean[5],
#     # Add more data arrays as needed
# ]
# IPR_list_IWP = [
#     IPR_match_PC_gap_IWP_constrain_mean[0],
#     IPR_match_PC_gap_IWP_constrain_mean[1],
#     IPR_match_PC_gap_IWP_constrain_mean[2],
#     IPR_match_PC_gap_IWP_constrain_mean[3],
#     IPR_match_PC_gap_IWP_constrain_mean[4],
#     IPR_match_PC_gap_IWP_constrain_mean[5],
# ]
# CTP_list_IWP = [
#     CTP_match_PC_gap_IWP_constrain_mean[0],
#     CTP_match_PC_gap_IWP_constrain_mean[1],
#     CTP_match_PC_gap_IWP_constrain_mean[2],
#     CTP_match_PC_gap_IWP_constrain_mean[3],
#     CTP_match_PC_gap_IWP_constrain_mean[4],
#     CTP_match_PC_gap_IWP_constrain_mean[5],
# ]
# CEP_list_IWP = [
#     CEP_match_PC_gap_AOD_constrain_mean[0],
#     CEP_match_PC_gap_AOD_constrain_mean[1],
#     CEP_match_PC_gap_AOD_constrain_mean[2],
#     CEP_match_PC_gap_AOD_constrain_mean[3],
#     CEP_match_PC_gap_AOD_constrain_mean[4],
#     CEP_match_PC_gap_AOD_constrain_mean[5],
# ]

# # AOD clipped data
# HCF_list_AOD = [
#     HCF_match_PC_gap_AOD_constrain_mean[0],
#     HCF_match_PC_gap_AOD_constrain_mean[1],
#     HCF_match_PC_gap_AOD_constrain_mean[2],
#     HCF_match_PC_gap_AOD_constrain_mean[3],
#     HCF_match_PC_gap_AOD_constrain_mean[4],
#     HCF_match_PC_gap_AOD_constrain_mean[5],
#     # Add more data arrays as needed
# ]
# IPR_list_AOD = [
#     IPR_match_PC_gap_AOD_constrain_mean[0],
#     IPR_match_PC_gap_AOD_constrain_mean[1],
#     IPR_match_PC_gap_AOD_constrain_mean[2],
#     IPR_match_PC_gap_AOD_constrain_mean[3],
#     IPR_match_PC_gap_AOD_constrain_mean[4],
#     IPR_match_PC_gap_AOD_constrain_mean[5],
# ]
# CTP_list_AOD = [
#     CTP_match_PC_gap_AOD_constrain_mean[0],
#     CTP_match_PC_gap_AOD_constrain_mean[1],
#     CTP_match_PC_gap_AOD_constrain_mean[2],
#     CTP_match_PC_gap_AOD_constrain_mean[3],
#     CTP_match_PC_gap_AOD_constrain_mean[4],
#     CTP_match_PC_gap_AOD_constrain_mean[5],
# ]
# CEP_list_AOD = [
#     CEP_match_PC_gap_AOD_constrain_mean[0],
#     CEP_match_PC_gap_AOD_constrain_mean[1],
#     CEP_match_PC_gap_AOD_constrain_mean[2],
#     CEP_match_PC_gap_AOD_constrain_mean[3],
#     CEP_match_PC_gap_AOD_constrain_mean[4],
#     CEP_match_PC_gap_AOD_constrain_mean[5],
# ]

# 4 gaps clipped data
HCF_list_IWP = [
    HCF_match_PC_gap_IWP_constrain_mean[0],
    HCF_match_PC_gap_IWP_constrain_mean[1],
    HCF_match_PC_gap_IWP_constrain_mean[2],
    HCF_match_PC_gap_IWP_constrain_mean[3],
    # Add more data arrays as needed
]
IPR_list_IWP = [
    IPR_match_PC_gap_IWP_constrain_mean[0],
    IPR_match_PC_gap_IWP_constrain_mean[1],
    IPR_match_PC_gap_IWP_constrain_mean[2],
    IPR_match_PC_gap_IWP_constrain_mean[3],
]
CTP_list_IWP = [
    CTP_match_PC_gap_IWP_constrain_mean[0],
    CTP_match_PC_gap_IWP_constrain_mean[1],
    CTP_match_PC_gap_IWP_constrain_mean[2],
    CTP_match_PC_gap_IWP_constrain_mean[3],
]
CEP_list_IWP = [
    CEP_match_PC_gap_AOD_constrain_mean[0],
    CEP_match_PC_gap_AOD_constrain_mean[1],
    CEP_match_PC_gap_AOD_constrain_mean[2],
    CEP_match_PC_gap_AOD_constrain_mean[3],
]

# AOD clipped data
HCF_list_AOD = [
    HCF_match_PC_gap_AOD_constrain_mean[0],
    HCF_match_PC_gap_AOD_constrain_mean[1],
    HCF_match_PC_gap_AOD_constrain_mean[2],
    HCF_match_PC_gap_AOD_constrain_mean[3],
    # Add more data arrays as needed
]
IPR_list_AOD = [
    IPR_match_PC_gap_AOD_constrain_mean[0],
    IPR_match_PC_gap_AOD_constrain_mean[1],
    IPR_match_PC_gap_AOD_constrain_mean[2],
    IPR_match_PC_gap_AOD_constrain_mean[3],
]
CTP_list_AOD = [
    CTP_match_PC_gap_AOD_constrain_mean[0],
    CTP_match_PC_gap_AOD_constrain_mean[1],
    CTP_match_PC_gap_AOD_constrain_mean[2],
    CTP_match_PC_gap_AOD_constrain_mean[3],
]
CEP_list_AOD = [
    CEP_match_PC_gap_AOD_constrain_mean[0],
    CEP_match_PC_gap_AOD_constrain_mean[1],
    CEP_match_PC_gap_AOD_constrain_mean[2],
    CEP_match_PC_gap_AOD_constrain_mean[3],
]

# endregion

# ------------------- Plotting --------------------------------
# Call the error_fill_3d_plot function with your data

annotations = {
    (0.1, 0.8): '(A)',
}

error_fill_3d_plot_no_legend_1f_4_gaps(
    data_list=HCF_list_IWP,
    xlabel="PC1",
    ylabel="IWP (g/m" + r"$^2$)",
    zlabel="Cld Area (%)",
    IWP_or_AOD_bin_values=IWP_gap,
    elev_angle=8,
    azim_angle=-21,
    savefig_str="HCF_IWP_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=-10,
    zmax=85,
)

annotations = {
    (0.1, 0.8): '(B)',
}

# plot the constrained ice particle radiusd
# IWP constrained
error_fill_3d_plot_no_legend_1f_4_gaps(
    data_list=IPR_list_IWP,
    xlabel="PC1",
    ylabel="IWP (g/m" + r"$^2$)",
    zlabel="Ice Particle Radius " + r"$(\mu m)$",
    IWP_or_AOD_bin_values=IWP_gap,
    elev_angle=8,
    azim_angle=-21,
    savefig_str="HCF_IPR_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=10,
    zmax=36,
)

# annotations = {
#     (0.1, 0.8): '(C)',
# }

# Plot the constrained cloud top height
# IWP constrained
# error_fill_3d_plot_no_legend_1f_4_gaps(
#     data_list=CTP_list_IWP,
#     xlabel="PC1",
#     ylabel="IWP (g/m" + r"$^2$)",
#     zlabel="Cloud Top Pressure (hPa)",
#     IWP_or_AOD_bin_values=IWP_gap,
#     elev_angle=8,
#     azim_angle=-21,
#     savefig_str="HCF_CTP_constrain_3D",
#     savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
#     annotations=annotations,
#     zmin=150,
#     zmax=300,
# )

annotations = {
    (0.1, 0.8): '(C)',
}

# Plot the constrained cloud effect height
# IWP constrained
error_fill_3d_plot_no_legend_1f_4_gaps(
    data_list=CEP_list_IWP,
    xlabel="PC1",
    ylabel="IWP (g/m" + r"$^2$)",
    zlabel="Cloud Effective Pressure (hPa)",
    IWP_or_AOD_bin_values=IWP_gap,
    elev_angle=8,
    azim_angle=-21,
    savefig_str="HCF_CEP_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=175,
    zmax=300,
)

# ------------------- Plotting --------------------------------
# Call the error_fill_3d_plot function with your data
annotations = {
    (0.1, 0.8): '(D)',
}

error_fill_3d_plot_no_legend_3f_4_gaps(
    data_list=HCF_list_AOD,
    xlabel="PC1",
    ylabel="AOD",
    zlabel="Cld Area (%)",
    IWP_or_AOD_bin_values=AOD_gap,
    elev_angle=8,
    azim_angle=-21,
    savefig_str="HCF_AOD_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=-10,
    zmax=100,
)

# plot the constrained ice particle radiusd
# AOD constrained
annotations = {
    (0.1, 0.8): '(E)',
}

error_fill_3d_plot_no_legend_3f_4_gaps(
    data_list=IPR_list_AOD,
    xlabel="PC1",
    ylabel="AOD",
    zlabel="Ice Particle Radius " + r"$(\mu m)$",
    IWP_or_AOD_bin_values=AOD_gap,
    elev_angle=8,
    azim_angle=-21,
    savefig_str="IPR_AOD_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=14,
    zmax=36,
)

# Plot the constrained cloud top height
# AOD constrained
# error_fill_3d_plot_no_legend_3f_4_gaps(
#     data_list=CTP_list_AOD,
#     xlabel="PC1",
#     ylabel="AOD",
#     zlabel="Cloud Top Pressure (hPa)",
#     IWP_or_AOD_bin_values=AOD_gap,
#     elev_angle=8,
#     azim_angle=-21,
#     savefig_str="CTP_AOD_constrain_3D",
#     savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
#     zmin=150,
#     zmax=300,
# )

# Plot the constrained cloud effect height
# AOD constrained
annotations = {
    (0.1, 0.8): '(F)',
}

error_fill_3d_plot_no_legend_3f_4_gaps(
    data_list=CEP_list_AOD,
    xlabel="PC1",
    ylabel="AOD",
    zlabel="Cloud Effective Pressure (hPa)",
    IWP_or_AOD_bin_values=IWP_gap,
    elev_angle=8,
    azim_angle=-21,
    savefig_str="CEP_AOD_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=150,
    zmax=300,
)
