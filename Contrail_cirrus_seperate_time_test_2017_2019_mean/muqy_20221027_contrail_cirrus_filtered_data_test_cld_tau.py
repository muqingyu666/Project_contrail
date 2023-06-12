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

from statistics import mean

import matplotlib as mpl
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# ----------  importing dcmap from my util ----------#
from muqy_20220413_util_useful_functions import dcmap as dcmap
from muqy_20220519_sactter_plot import scatter_plot_simulated_observed as scatter_plot
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)
from scipy import stats
from scipy.stats import norm

# --------- import done ------------
# --------- Plot style -------------

mpl.rc("font", family="Times New Roman")
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")


# ---------- Read PCA&CLD data from netcdf file --------
(
    # pc
    PC_all,
    PC_years,
    # cld
    Cld_all,
    Cld_years,
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_num=2)
# 0 for Cldarea dataset, 1 for Cldicerad dataset
# 2 for Cldtau dataset, 3 for Cldtau_lin dataset, 4 for IWP dataset
# 5 for Cldemissirad dataset

# use the 2010-2020 PC1 only
PC_all = PC_all[-11:]

# read in the correlation coefficient and pvalue
data = xr.open_dataset(
    "corr_data/correlation_pvalue_PC1_Cldarea.nc"
)
correlation = data["correlation"].values
pvalue = data["p_values"].values

# Mask the PC1 and Cldarea data where the correlation coefficients are less than 0.45
# Assuming PC_years is a dictionary with 11 arrays and correlation has shape (180, 360)
mask = correlation < 0.45

for year in range(2017, 2021):
    PC_years[year] = np.ma.masked_array(
        PC_years[year],
        mask=np.repeat(
            mask[np.newaxis, ...], 12 * 28, axis=0
        ).reshape(12, 28, 180, 360),
    )
    Cld_years[year] = np.ma.masked_array(
        Cld_years[year],
        mask=np.repeat(
            mask[np.newaxis, ...], 12 * 28, axis=0
        ).reshape(12, 28, 180, 360),
    )


# Extract the PC1 and Cldarea data for each year
def extract_PC1_CERES_each_year(
    PC_years: dict[int, np.ndarray],
    Cld_years: dict[int, np.ndarray],
) -> None:
    """
    Extracts data for principal component 1 (PC1) and cloud data (CERES) for each year from 2010 to 2020,
    and assigns these data to global variables with names that include the year.

    Parameters:
    -----------
    PC_years: dict[int, np.ndarray]
        A dictionary containing the principal component 1 data for each year from 2010 to 2020. The keys
        are integers representing years, and the values are numpy arrays containing the PC1 data.
    Cld_years: dict[int, np.ndarray]
        A dictionary containing the CERES cloud data for each year from 2010 to 2020. The keys are integers
        representing years, and the values are numpy arrays containing the CERES data.

    Returns:
    --------
    None
    """
    for year in range(2017, 2021):
        globals()[f"PC_{year}"] = PC_years[year]

    for year in range(2017, 2021):
        globals()[f"Cld_{year}"] = Cld_years[year]


extract_PC1_CERES_each_year(PC_years, Cld_years)

#########################################
##### start seperate time test ##########
#########################################

# region
# extract 3-4 month data & 5-6 month data & 7-8 month data only
years = range(2017, 2021)
# months to extract
months_1_2_month = slice(0, 2)
months_3_4_month = slice(2, 4)
months_5_6_month = slice(4, 6)
months_7_8_month = slice(6, 8)

for year in years:
    # get the data from the global namespace
    pc_data = globals()[f"PC_{year}"]
    cld_data = globals()[f"Cld_{year}"]

    # apply the months slice to the data
    # extract 1-2 month data & 3-4 month data & 5-6 month data & 7-8 month data only
    # 1-2 month data
    pc_1_2_month = pc_data.reshape(12, 28, 180, 360)[
        months_1_2_month, :, :, :
    ].reshape(2 * 28, 180, 360)
    cld_1_2_month = cld_data.reshape(12, 28, 180, 360)[
        months_1_2_month, :, :, :
    ].reshape(2 * 28, 180, 360)

    # 3-4 month data
    pc_3_4_month = pc_data.reshape(12, 28, 180, 360)[
        months_3_4_month, :, :, :
    ].reshape(2 * 28, 180, 360)
    cld_3_4_month = cld_data.reshape(12, 28, 180, 360)[
        months_3_4_month, :, :, :
    ].reshape(2 * 28, 180, 360)

    # 5-6 month data
    pc_5_6_month = pc_data.reshape(12, 28, 180, 360)[
        months_5_6_month, :, :, :
    ].reshape(2 * 28, 180, 360)
    cld_5_6_month = cld_data.reshape(12, 28, 180, 360)[
        months_5_6_month, :, :, :
    ].reshape(2 * 28, 180, 360)

    # 7-8 month data
    pc_7_8_month = pc_data.reshape(12, 28, 180, 360)[
        months_7_8_month, :, :, :
    ].reshape(2 * 28, 180, 360)
    cld_7_8_month = cld_data.reshape(12, 28, 180, 360)[
        months_7_8_month, :, :, :
    ].reshape(2 * 28, 180, 360)

    # assign the data to the global namespace
    globals()[f"PC_{year}_1_2_month"] = pc_1_2_month
    globals()[f"Cld_{year}_1_2_month"] = cld_1_2_month

    globals()[f"PC_{year}_3_4_month"] = pc_3_4_month
    globals()[f"Cld_{year}_3_4_month"] = cld_3_4_month

    globals()[f"PC_{year}_5_6_month"] = pc_5_6_month
    globals()[f"Cld_{year}_5_6_month"] = cld_5_6_month

    globals()[f"PC_{year}_7_8_month"] = pc_7_8_month
    globals()[f"Cld_{year}_7_8_month"] = cld_7_8_month


plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    Cld_match_PC_gap=PC_2018_3_4_month[1, :, :],
    # p_value,
    cld_min=-2,
    cld_max=5.5,
    cld_name="Cloud fraction",
    cmap_file="/RAID01/data/muqy/color/test.txt",
)

plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    Cld_match_PC_gap=Cld_2018_3_4_month[1, :, :],
    # p_value,
    cld_min=0,
    cld_max=80,
    cld_name="Cloud fraction",
    cmap_file="/RAID01/data/muqy/color/test.txt",
)


# concatenate 1-2 month & 3-4 month & 5-6 month & 7-8 month data
# to form a 2010->2019 dataset
PC_2017_2019_1_2_month = np.concatenate(
    [
        globals()[f"PC_{year}_1_2_month"]
        for year in range(2017, 2020)
    ],
    axis=0,
)
Cld_2017_2019_1_2_month = np.concatenate(
    [
        globals()[f"Cld_{year}_1_2_month"]
        for year in range(2017, 2020)
    ],
    axis=0,
)

PC_2017_2019_3_4_month = np.concatenate(
    [
        globals()[f"PC_{year}_3_4_month"]
        for year in range(2017, 2020)
    ],
    axis=0,
)
Cld_2017_2019_3_4_month = np.concatenate(
    [
        globals()[f"Cld_{year}_3_4_month"]
        for year in range(2017, 2020)
    ],
    axis=0,
)

PC_2017_2019_5_6_month = np.concatenate(
    [
        globals()[f"PC_{year}_5_6_month"]
        for year in range(2017, 2020)
    ],
    axis=0,
)
Cld_2017_2019_5_6_month = np.concatenate(
    [
        globals()[f"Cld_{year}_5_6_month"]
        for year in range(2017, 2020)
    ],
    axis=0,
)

PC_2017_2019_7_8_month = np.concatenate(
    [
        globals()[f"PC_{year}_7_8_month"]
        for year in range(2017, 2020)
    ],
    axis=0,
)
Cld_2017_2019_7_8_month = np.concatenate(
    [
        globals()[f"Cld_{year}_7_8_month"]
        for year in range(2017, 2020)
    ],
    axis=0,
)
# endregion

# ------ Segmentation of cloud data within each PC interval ---------------------------------
#
filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=Cld_2018.reshape(-1, 180, 360),
    start=-2.5,
    end=5.5,
    gap=0.05,
)

# region
# 1 - 2 month
(
    Cld_2020_match_PC_gap_1_2_month_median,
    PC_2020_match_PC_gap_1_2_month_median,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new_median(
    Cld_data=Cld_2020_1_2_month,
    PC_data=PC_2020_1_2_month,
)

(
    Cld_2017_2019_match_PC_gap_1_2_month_median,
    PC_2017_2019_match_PC_gap_1_2_month_median,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new_median(
    Cld_data=Cld_2017_2019_1_2_month,
    PC_data=PC_2017_2019_1_2_month,
)

(
    Cld_2020_match_PC_gap_1_2_month_mean,
    PC_2020_match_PC_gap_1_2_month_mean,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_2020_1_2_month,
    PC_data=PC_2020_1_2_month,
)

(
    Cld_2017_2019_match_PC_gap_1_2_month_mean,
    PC_2017_2019_match_PC_gap_1_2_month_mean,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_2017_2019_1_2_month,
    PC_data=PC_2017_2019_1_2_month,
)

# 3 - 4 month
(
    Cld_2020_match_PC_gap_3_4_month_median,
    PC_2020_match_PC_gap_3_4_month_median,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new_median(
    Cld_data=Cld_2020_3_4_month,
    PC_data=PC_2020_3_4_month,
)

(
    Cld_2017_2019_match_PC_gap_3_4_month_median,
    PC_2017_2019_match_PC_gap_3_4_month_median,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new_median(
    Cld_data=Cld_2017_2019_3_4_month,
    PC_data=PC_2017_2019_3_4_month,
)

(
    Cld_2020_match_PC_gap_3_4_month_mean,
    PC_2020_match_PC_gap_3_4_month_mean,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_2020_3_4_month,
    PC_data=PC_2020_3_4_month,
)

(
    Cld_2017_2019_match_PC_gap_3_4_month_mean,
    PC_2017_2019_match_PC_gap_3_4_month_mean,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_2017_2019_3_4_month,
    PC_data=PC_2017_2019_3_4_month,
)

# 5 - 6 month
(
    Cld_2020_match_PC_gap_5_6_month_median,
    PC_2020_match_PC_gap_5_6_month_median,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new_median(
    Cld_data=Cld_2020_5_6_month,
    PC_data=PC_2020_5_6_month,
)

(
    Cld_2017_2019_match_PC_gap_5_6_month_median,
    PC_2017_2019_match_PC_gap_5_6_month_median,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new_median(
    Cld_data=Cld_2017_2019_5_6_month,
    PC_data=PC_2017_2019_5_6_month,
)

(
    Cld_2020_match_PC_gap_5_6_month_mean,
    PC_2020_match_PC_gap_5_6_month_mean,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_2020_5_6_month,
    PC_data=PC_2020_5_6_month,
)

(
    Cld_2017_2019_match_PC_gap_5_6_month_mean,
    PC_2017_2019_match_PC_gap_5_6_month_mean,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_2017_2019_5_6_month,
    PC_data=PC_2017_2019_5_6_month,
)

# 7 - 8 month
(
    Cld_2020_match_PC_gap_7_8_month_median,
    PC_2020_match_PC_gap_7_8_month_median,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new_median(
    Cld_data=Cld_2020_7_8_month,
    PC_data=PC_2020_7_8_month,
)

(
    Cld_2017_2019_match_PC_gap_7_8_month_median,
    PC_2017_2019_match_PC_gap_7_8_month_median,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new_median(
    Cld_data=Cld_2017_2019_7_8_month,
    PC_data=PC_2017_2019_7_8_month,
)

(
    Cld_2020_match_PC_gap_7_8_month_mean,
    PC_2020_match_PC_gap_7_8_month_mean,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_2020_7_8_month,
    PC_data=PC_2020_7_8_month,
)

(
    Cld_2017_2019_match_PC_gap_7_8_month_mean,
    PC_2017_2019_match_PC_gap_7_8_month_mean,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_2017_2019_7_8_month,
    PC_data=PC_2017_2019_7_8_month,
)
# endregion

#######################################################################
################# Filter data within PC interval ######################
####### Using 10% lowest and highest PC values in each interval #######
#######################################################################

#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################

# region
(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_1_2_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_1_2_month_median
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_1_2_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_1_2_month_median
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_1_2_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_1_2_month_mean
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_1_2_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_1_2_month_mean
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_3_4_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_3_4_month_median
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_3_4_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_3_4_month_median
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_3_4_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_3_4_month_mean
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_3_4_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_3_4_month_mean
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_5_6_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_5_6_month_median
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_5_6_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_5_6_month_median
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_5_6_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_5_6_month_mean
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_5_6_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_5_6_month_mean
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_7_8_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_7_8_month_median
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_7_8_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_7_8_month_median
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_7_8_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_7_8_month_mean
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_7_8_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_7_8_month_mean
)
# endregion

#######################################################################
#######################################################################
#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################

# region
# -----------------------------------------------------------------------------
# divide the PC1 data into 6 parts

concatenated_data = np.concatenate(
    (
        PC_2020_1_2_month[:, 110:140, :],
        PC_2020_3_4_month[:, 110:140, :],
        PC_2020_5_6_month[:, 110:140, :],
        PC_2020_7_8_month[:, 110:140, :],
    ),
    axis=0,
)

dividePC = DividePCByDataVolume(
    dataarray_main=concatenated_data,
    n=5,
    start=-2.5,
    end=5.5,
    gap=0.05,
)
gap_num_min, gap_num_max = dividePC.gap_number_for_giving_main_gap()

# retrieve the sparse PC_constrain data
# dimension 1 means atmospheric condiations
# 0 means bad, 1 means moderate, 2 means good
Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean = np.empty(
    (3, 360)
)
Cld_delta_2020_match_PC_gap_filtered_3_4_month_mean = np.empty(
    (3, 360)
)
Cld_delta_2020_match_PC_gap_filtered_5_6_month_mean = np.empty(
    (3, 360)
)
Cld_delta_2020_match_PC_gap_filtered_7_8_month_mean = np.empty(
    (3, 360)
)
# 1-2 month
for i in range(0, 3):
    Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean[
        i
    ] = np.nanmean(
        Cld_2020_match_PC_gap_filtered_1_2_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    ) - np.nanmean(
        Cld_2017_2019_match_PC_gap_filtered_1_2_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    )
    # 3-4 month
    Cld_delta_2020_match_PC_gap_filtered_3_4_month_mean[
        i
    ] = np.nanmean(
        Cld_2020_match_PC_gap_filtered_3_4_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    ) - np.nanmean(
        Cld_2017_2019_match_PC_gap_filtered_3_4_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    )
    # 5-6 month
    Cld_delta_2020_match_PC_gap_filtered_5_6_month_mean[
        i
    ] = np.nanmean(
        Cld_2020_match_PC_gap_filtered_5_6_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    ) - np.nanmean(
        Cld_2017_2019_match_PC_gap_filtered_5_6_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    )
    # 7-8 month
    Cld_delta_2020_match_PC_gap_filtered_7_8_month_mean[
        i
    ] = np.nanmean(
        Cld_2020_match_PC_gap_filtered_7_8_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    ) - np.nanmean(
        Cld_2017_2019_match_PC_gap_filtered_7_8_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    )


# Improved version
# define lists for atmospheric conditions and calculation type
atm_conditions = ["bad", "moderate", "good"]
calc_types = ["median", "mean"]

# create empty dictionaries to store the results
results_median = {}
results_mean = {}

# -----------------------------------------------------------------------------
# 1 - 2 month

# iterate over atmospheric conditions and calculation types
for i, atm in enumerate(atm_conditions):
    for calc_type in calc_types:
        # create the variable names dynamically
        var_name = f"{atm}_filtered_{calc_type}"
        cld_2020_var = (
            f"Cld_2020_match_PC_gap_filtered_1_2_month_{calc_type}"
        )
        cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_1_2_month_{calc_type}"
        gap_min_var = gap_num_min[i]
        gap_max_var = gap_num_max[i]

        # call the compare_cld_between_2020_others function
        results = compare_cld_between_2020_others(
            Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
            Cld_all_match_PC_gap_others=locals()[cld_others_var],
            start=int(gap_min_var),
            end=int(gap_max_var),
        )

        # store the results in the appropriate dictionary
        if calc_type == "median":
            results_median[var_name] = results
        elif calc_type == "mean":
            results_mean[var_name] = results

# assign the results to variables
Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median = (
    results_median["bad_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median = (
    results_median["moderate_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median = (
    results_median["good_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean = (
    results_mean["bad_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean = (
    results_mean["moderate_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean = (
    results_mean["good_filtered_mean"]
)

# ---------------------------------------------------------------------------------
# 3 - 4 month
# iterate over atmospheric conditions and calculation types
for i, atm in enumerate(atm_conditions):
    for calc_type in calc_types:
        # create the variable names dynamically
        var_name = f"{atm}_filtered_{calc_type}"
        cld_2020_var = (
            f"Cld_2020_match_PC_gap_filtered_3_4_month_{calc_type}"
        )
        cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_3_4_month_{calc_type}"
        gap_min_var = gap_num_min[i]
        gap_max_var = gap_num_max[i]

        # call the compare_cld_between_2020_others function
        results = compare_cld_between_2020_others(
            Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
            Cld_all_match_PC_gap_others=locals()[cld_others_var],
            start=int(gap_min_var),
            end=int(gap_max_var),
        )

        # store the results in the appropriate dictionary
        if calc_type == "median":
            results_median[var_name] = results
        elif calc_type == "mean":
            results_mean[var_name] = results

# assign the results to variables
Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median = (
    results_median["bad_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median = (
    results_median["moderate_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median = (
    results_median["good_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean = (
    results_mean["bad_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean = (
    results_mean["moderate_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean = (
    results_mean["good_filtered_mean"]
)

# --------------------------------------------------------------------------------------------
# 5 - 6 month
# iterate over atmospheric conditions and calculation types
for i, atm in enumerate(atm_conditions):
    for calc_type in calc_types:
        # create the variable names dynamically
        var_name = f"{atm}_filtered_{calc_type}"
        cld_2020_var = (
            f"Cld_2020_match_PC_gap_filtered_5_6_month_{calc_type}"
        )
        cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_5_6_month_{calc_type}"
        gap_min_var = gap_num_min[i]
        gap_max_var = gap_num_max[i]

        # call the compare_cld_between_2020_others function
        results = compare_cld_between_2020_others(
            Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
            Cld_all_match_PC_gap_others=locals()[cld_others_var],
            start=int(gap_min_var),
            end=int(gap_max_var),
        )

        # store the results in the appropriate dictionary
        if calc_type == "median":
            results_median[var_name] = results
        elif calc_type == "mean":
            results_mean[var_name] = results

Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median = (
    results_median["bad_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median = (
    results_median["moderate_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median = (
    results_median["good_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean = (
    results_mean["bad_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean = (
    results_mean["moderate_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean = (
    results_mean["good_filtered_mean"]
)

# --------------------------------------------------------------------------------------------
# 7 - 8 month
# iterate over atmospheric conditions and calculation types
for i, atm in enumerate(atm_conditions):
    for calc_type in calc_types:
        # create the variable names dynamically
        var_name = f"{atm}_filtered_{calc_type}"
        cld_2020_var = (
            f"Cld_2020_match_PC_gap_filtered_7_8_month_{calc_type}"
        )
        cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_7_8_month_{calc_type}"
        gap_min_var = gap_num_min[i]
        gap_max_var = gap_num_max[i]

        # call the compare_cld_between_2020_others function
        results = compare_cld_between_2020_others(
            Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
            Cld_all_match_PC_gap_others=locals()[cld_others_var],
            start=int(gap_min_var),
            end=int(gap_max_var),
        )

        # store the results in the appropriate dictionary
        if calc_type == "median":
            results_median[var_name] = results
        elif calc_type == "mean":
            results_mean[var_name] = results

Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median = (
    results_median["bad_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median = (
    results_median["moderate_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median = (
    results_median["good_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean = (
    results_mean["bad_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean = (
    results_mean["moderate_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean = (
    results_mean["good_filtered_mean"]
)
# --------------------------------------------------------------------------------------------
# endregion

########################################################################
###### Draw the Cld difference between 2020 and 2010-2019 ###############
###### But by latitude this time, 20N-60N, Contrail region ##############
########################################################################

#########################################################
############ smoothed data ##############################
#########################################################

# ------------------------------------------------------------
# plot the aviation fill between version
# ------------------------------------------------------------
y_lim_lst = [[-2, 2.5], [-2.5, 2.5], [-4.5, 2.5]]

# 1-2 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
    - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
    Cld_data_name=r"$\Delta$" + "COD",
    y_lim_lst=y_lim_lst,
    title="January-February",
    step=5,
)

# 3-4 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
    - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
    Cld_data_name=r"$\Delta$" + "COD",
    y_lim_lst=y_lim_lst,
    title="March-April",
    step=5,
)

# 5-6 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
    - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
    Cld_data_name=r"$\Delta$" + "COD",
    y_lim_lst=y_lim_lst,
    title="May-June",
    step=5,
)

# 7-8 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
    - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
    Cld_data_name=r"$\Delta$" + "COD",
    y_lim_lst=y_lim_lst,
    title="July-August",
    step=5,
)

# no compare between PC1 and no PC1 constrain
# ------------------------------------------------------------
# plot the aviation fill between version
# ------------------------------------------------------------
y_lim_lst = [[-1.15, 1.15], [-1.15, 1.15], [-1.15, 1.15]]

# 1-2 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
    - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
    Cld_data_name=r"$\Delta$" + "COD",
    y_lim_lst=y_lim_lst,
    title="January-February",
    step=10,
)

# 3-4 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
    - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
    Cld_data_name=r"$\Delta$" + "COD",
    y_lim_lst=y_lim_lst,
    title="March-April",
    step=10,
)

# 5-6 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
    - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
    Cld_data_name=r"$\Delta$" + "COD",
    y_lim_lst=y_lim_lst,
    title="May-June",
    step=5,
)

# 7-8 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
    - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
    Cld_data_name=r"$\Delta$" + "COD",
    y_lim_lst=y_lim_lst,
    title="July-August",
    step=10,
)


# zscores
# ------------------------------------------------------------
# plot the aviation fill between version
# ------------------------------------------------------------
y_lim_lst = [[-1.15, 1.15], [-1.15, 1.15], [-1.15, 1.15]]

# 1-2 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
    - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="January-February",
    step=10,
)

# 3-4 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
    - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="March-April",
    step=10,
)

# 5-6 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
    - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="May-June",
    step=5,
)

# 7-8 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
    - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="July-August",
    step=10,
)

# --------------------------------------------
# plot the improve version
# --------------------------------------------
# 1-2 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
    Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
    Cld_data_name=r"$\Delta$" + "COD",
    title="January-February",
    step=5,
)

# 3-4 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
    Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
    Cld_data_name=r"$\Delta$" + "COD",
    title="March-April",
    step=5,
)

# 5-6 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
    Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
    Cld_data_name=r"$\Delta$" + "COD",
    title="May-June",
    step=5,
)

# 7-8 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
    Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
    Cld_data_name=r"$\Delta$" + "COD",
    title="July-August",
    step=5,
)
