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
from muqy_20220519_sactter_plot import (
    scatter_plot_simulated_observed as scatter_plot,
)
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

# ---------- Read in Cloud area data ----------
(
    PC_all,
    Cldarea,
    Cldtau,
    CldtauL,
    Cldemissir,
    IWP,
    Cldpress_top,
    Cldtemp_top,
    Cldhgth_top,
    Cldpress_base,
    Cldtemp_base,
    Cldicerad,
    Cldphase,
    Cldeff_press,
    Cldeff_temp,
    Cldeff_hgth,
) = read_PC1_multi_CERES_from_netcdf()

# ---------- Extract 2020 data ----------
(
    PC_2020,
    Cldarea_2020,
    Cldtau_2020,
    CldtauL_2020,
    Cldemissir_2020,
    Cldicerad_2020,
    Cldphase_2020,
    Cldpress_top_2020,
    Cldtemp_top_2020,
    Cldhgth_top_2020,
    Cldpress_base_2020,
    Cldtemp_base_2020,
    Cldeff_press_2020,
    Cldeff_temp_2020,
    Cldeff_hgth_2020,
    IWP_2020,
) = [
    x[-1, :, :, :, :]
    for x in (
        PC_all,
        Cldarea,
        Cldtau,
        CldtauL,
        Cldemissir,
        Cldicerad,
        Cldphase,
        Cldpress_top,
        Cldtemp_top,
        Cldhgth_top,
        Cldpress_base,
        Cldtemp_base,
        Cldeff_press,
        Cldeff_temp,
        Cldeff_hgth,
        IWP,
    )
]

# ---------- Extract 2010-2019 data ----------
(
    PC_2010_2019,
    Cldarea_2010_2019,
    Cldtau_2010_2019,
    CldtauL_2010_2019,
    Cldemissir_2010_2019,
    Cldicerad_2010_2019,
    Cldphase_2010_2019,
    Cldpress_top_2010_2019,
    Cldtemp_top_2010_2019,
    Cldhgth_top_2010_2019,
    Cldpress_base_2010_2019,
    Cldtemp_base_2010_2019,
    Cldeff_press_2010_2019,
    Cldeff_temp_2010_2019,
    Cldeff_hgth_2010_2019,
    IWP_2010_2019,
) = [
    x[:-1, :, :, :, :]
    for x in (
        PC_all,
        Cldarea,
        Cldtau,
        CldtauL,
        Cldemissir,
        Cldicerad,
        Cldphase,
        Cldpress_top,
        Cldtemp_top,
        Cldhgth_top,
        Cldpress_base,
        Cldtemp_base,
        Cldeff_press,
        Cldeff_temp,
        Cldeff_hgth,
        IWP,
    )
]

###############################################################################
####### China test: Extract Jan to Mar data from 2010 to 2019 and 2020 ######################
####### Euro and USA test: Extract Apr to Jun data from 2010 to 2019 and 2020 ######################
###############################################################################

# region
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
# Extract 2020 data
PC_2020_1_3_month = PC_2020.reshape(12, 28, 180, 360)[
    :3, :, :, :
].reshape(3 * 28, 180, 360)

Cldarea_2020_1_3_month = Cldarea_2020.reshape(12, 28, 180, 360)[
    :3, :, :, :
].reshape(3 * 28, 180, 360)

Cldtau_2020_1_3_month = Cldtau_2020.reshape(12, 28, 180, 360)[
    :3, :, :, :
].reshape(3 * 28, 180, 360)
Cldicerad_2020_1_3_month = Cldicerad_2020.reshape(12, 28, 180, 360)[
    :3, :, :, :
].reshape(3 * 28, 180, 360)
# Extract 2010-2019 data
PC_2010_2019_1_3_month = PC_2010_2019.reshape(120, 28, 180, 360)[
    : 3 * 10, :, :, :
].reshape(3 * 10 * 28, 180, 360)
Cldarea_2010_2019_1_3_month = Cldarea_2010_2019.reshape(
    120, 28, 180, 360
)[: 3 * 10, :, :, :].reshape(3 * 10 * 28, 180, 360)
Cldtau_2010_2019_1_3_month = Cldtau_2010_2019.reshape(
    120, 28, 180, 360
)[: 3 * 10, :, :, :].reshape(3 * 10 * 28, 180, 360)
Cldicerad_2010_2019_1_3_month = Cldicerad_2010_2019.reshape(
    120, 28, 180, 360
)[: 3 * 10, :, :, :].reshape(3 * 10 * 28, 180, 360)

# Euro and USA test: Extract Apr to Jun data from 2010 to 2019 and 2020
# Extract 2020 data
PC_2020_4_6_month = PC_2020.reshape(12, 28, 180, 360)[
    3:6, :, :, :
].reshape(3 * 28, 180, 360)
Cldarea_2020_4_6_month = Cldarea_2020.reshape(12, 28, 180, 360)[
    3:6, :, :, :
].reshape(3 * 28, 180, 360)
Cldtau_2020_4_6_month = Cldtau_2020.reshape(12, 28, 180, 360)[
    3:6, :, :, :
].reshape(3 * 28, 180, 360)
Cldicerad_2020_4_6_month = Cldicerad_2020.reshape(12, 28, 180, 360)[
    3:6, :, :, :
].reshape(3 * 28, 180, 360)

# Extract 2010-2019 data
PC_2010_2019_4_6_month = PC_2010_2019.reshape(120, 28, 180, 360)[
    3 * 10 : 3 * 20, :, :, :
].reshape(3 * 10 * 28, 180, 360)
Cldarea_2010_2019_4_6_month = Cldarea_2010_2019.reshape(
    120, 28, 180, 360
)[3 * 10 : 3 * 20, :, :, :].reshape(3 * 10 * 28, 180, 360)
Cldtau_2010_2019_4_6_month = Cldtau_2010_2019.reshape(
    120, 28, 180, 360
)[3 * 10 : 3 * 20, :, :, :].reshape(3 * 10 * 28, 180, 360)
Cldicerad_2010_2019_4_6_month = Cldicerad_2010_2019.reshape(
    120, 28, 180, 360
)[3 * 10 : 3 * 20, :, :, :].reshape(3 * 10 * 28, 180, 360)
# endregion


# ------ Segmentation of cloud data within each PC interval ---------------------------------
# Filter data and fit PC1
filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=PC_2020_1_3_month, start=-2.5, end=5.5, gap=0.05
)

# region
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
(
    Cldarea_2020_match_PC_gap_1_3_month,
    PC_2020_match_PC_gap_1_3_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldarea_2020_1_3_month, PC_data=PC_2020_1_3_month
)
(
    Cldtau_2020_match_PC_gap_1_3_month,
    PC_2020_match_PC_gap_1_3_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtau_2020_1_3_month, PC_data=PC_2020_1_3_month
)
(
    Cldicerad_2020_match_PC_gap_1_3_month,
    PC_2020_match_PC_gap_1_3_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldicerad_2020_1_3_month, PC_data=PC_2020_1_3_month
)

(
    Cldarea_2010_2019_match_PC_gap_1_3_month,
    PC_2010_2019_match_PC_gap_1_3_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldarea_2010_2019_1_3_month,
    PC_data=PC_2010_2019_1_3_month,
)
(
    Cldtau_2010_2019_match_PC_gap_1_3_month,
    PC_2010_2019_match_PC_gap_1_3_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtau_2010_2019_1_3_month,
    PC_data=PC_2010_2019_1_3_month,
)
(
    Cldicerad_2010_2019_match_PC_gap_1_3_month,
    PC_2010_2019_match_PC_gap_1_3_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldicerad_2010_2019_1_3_month,
    PC_data=PC_2010_2019_1_3_month,
)

# Euro and US test: Extract  Apr to Jun data from 2010 to 2019 and 2020
(
    Cldarea_2020_match_PC_gap_4_6_month,
    PC_2020_match_PC_gap_4_6_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldarea_2020_4_6_month, PC_data=PC_2020_4_6_month
)
(
    Cldtau_2020_match_PC_gap_4_6_month,
    PC_2020_match_PC_gap_4_6_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtau_2020_4_6_month, PC_data=PC_2020_4_6_month
)
(
    Cldicerad_2020_match_PC_gap_4_6_month,
    PC_2020_match_PC_gap_4_6_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldicerad_2020_4_6_month, PC_data=PC_2020_4_6_month
)
(
    Cldarea_2010_2019_match_PC_gap_4_6_month,
    PC_2010_2019_match_PC_gap_4_6_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldarea_2010_2019_4_6_month,
    PC_data=PC_2010_2019_4_6_month,
)
(
    Cldtau_2010_2019_match_PC_gap_4_6_month,
    PC_2010_2019_match_PC_gap_4_6_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtau_2010_2019_4_6_month,
    PC_data=PC_2010_2019_4_6_month,
)
(
    Cldicerad_2010_2019_match_PC_gap_4_6_month,
    PC_2010_2019_match_PC_gap_4_6_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldicerad_2010_2019_4_6_month,
    PC_data=PC_2010_2019_4_6_month,
)
# endregion

#######################################################################
################# Filter data within PC interval ######################
####### Using 10% lowest and highest PC values in each interval #######
#######################################################################

# region
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
(
    Cldarea_2020_lowermost_error_1_3_month,
    Cldarea_2020_highermost_error_1_3_month,
    Cldarea_2020_match_PC_gap_filtered_1_3_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldarea_2020_match_PC_gap_1_3_month
)
(
    Cldtau_2020_lowermost_error_1_3_month,
    Cldtau_2020_highermost_error_1_3_month,
    Cldtau_2020_match_PC_gap_filtered_1_3_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldtau_2020_match_PC_gap_1_3_month
)
(
    Cldicerad_2020_lowermost_error_1_3_month,
    Cldicerad_2020_highermost_error_1_3_month,
    Cldicerad_2020_match_PC_gap_filtered_1_3_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldicerad_2020_match_PC_gap_1_3_month
)

(
    Cldarea_2010_2019_lowermost_error_1_3_month,
    Cldarea_2010_2019_highermost_error_1_3_month,
    Cldarea_2010_2019_match_PC_gap_filtered_1_3_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldarea_2010_2019_match_PC_gap_1_3_month
)
(
    Cldtau_2010_2019_lowermost_error_1_3_month,
    Cldtau_2010_2019_highermost_error_1_3_month,
    Cldtau_2010_2019_match_PC_gap_filtered_1_3_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldtau_2010_2019_match_PC_gap_1_3_month
)
(
    Cldicerad_2010_2019_lowermost_error_1_3_month,
    Cldicerad_2010_2019_highermost_error_1_3_month,
    Cldicerad_2010_2019_match_PC_gap_filtered_1_3_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldicerad_2010_2019_match_PC_gap_1_3_month
)

# Euro and US test: Extract  Apr to Jun data from 2010 to 2019 and 2020
(
    Cldarea_2020_lowermost_error_4_6_month,
    Cldarea_2020_highermost_error_4_6_month,
    Cldarea_2020_match_PC_gap_filtered_4_6_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldarea_2020_match_PC_gap_4_6_month
)
(
    Cldtau_2020_lowermost_error_4_6_month,
    Cldtau_2020_highermost_error_4_6_month,
    Cldtau_2020_match_PC_gap_filtered_4_6_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldtau_2020_match_PC_gap_4_6_month
)
(
    Cldicerad_2020_lowermost_error_4_6_month,
    Cldicerad_2020_highermost_error_4_6_month,
    Cldicerad_2020_match_PC_gap_filtered_4_6_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldicerad_2020_match_PC_gap_4_6_month
)

(
    Cldarea_2010_2019_lowermost_error_4_6_month,
    Cldarea_2010_2019_highermost_error_4_6_month,
    Cldarea_2010_2019_match_PC_gap_filtered_4_6_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldarea_2010_2019_match_PC_gap_4_6_month
)
(
    Cldtau_2010_2019_lowermost_error_4_6_month,
    Cldtau_2010_2019_highermost_error_4_6_month,
    Cldtau_2010_2019_match_PC_gap_filtered_4_6_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldtau_2010_2019_match_PC_gap_4_6_month
)
(
    Cldicerad_2010_2019_lowermost_error_4_6_month,
    Cldicerad_2010_2019_highermost_error_4_6_month,
    Cldicerad_2010_2019_match_PC_gap_filtered_4_6_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cldicerad_2010_2019_match_PC_gap_4_6_month
)
# endregion

#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################


#################################################################################
###### Plot the HCF difference when no pc1 constrain are used ###################
#################################################################################
# Plot mean HCF of 2020
# filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
#     start=0,
#     end=18,
#     Cld_match_PC_gap=np.nanmean(Cld_2020_3_7_month, axis=0),
#     cld_min=0,
#     cld_max=60,
#     cld_name="HCF (%)",
# )

# # Plot mean HCF of 2010-2019
# filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
#     start=0,
#     end=18,
#     Cld_match_PC_gap=np.nanmean(Cld_2010_2019_3_7_month, axis=0),
#     cld_min=0,
#     cld_max=60,
#     cld_name="HCF (%)",
# )

# # Plot mean HCF of 2020 subtract 2010-2019
# filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
#     start=0,
#     end=18,
#     Cld_match_PC_gap=np.nanmean(Cld_2020_3_7_month, axis=0)
#     - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
#     cld_min=-0.15,
#     cld_max=0.15,
#     # cld_name="HCF Diff (%)",
#     # cld_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
#     # cld_name="Cld tau Diff",
#     cld_name="Cld IRemiss Diff",
# )

#################################################################################
###### Plot the filtered data with new filter method ############################
#################################################################################

# --------------------------------------------------------------------------------------------
#  Filtered version
# --------------------------------------------------------------------------------------------
# region
# Bad atmospheric condition
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
Cldarea_match_PC_gap_mean_bad_filtered_1_3_month_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldarea_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldarea_2010_2019_match_PC_gap_filtered_1_3_month,
    start=0,
    end=26,
)
Cldtau_match_PC_gap_mean_bad_filtered_1_3_month_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldtau_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldtau_2010_2019_match_PC_gap_filtered_1_3_month,
    start=0,
    end=26,
)
Cldicerad_match_PC_gap_mean_bad_filtered_1_3_month_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldicerad_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldicerad_2010_2019_match_PC_gap_filtered_1_3_month,
    start=0,
    end=26,
)

# Euro and US test: Extract  Apr to Jun data from 2010 to 2019 and 2020
Cldarea_match_PC_gap_mean_bad_filtered_4_6_month_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldarea_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldarea_2010_2019_match_PC_gap_filtered_4_6_month,
    start=0,
    end=26,
)
Cldtau_match_PC_gap_mean_bad_filtered_4_6_month_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldtau_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldtau_2010_2019_match_PC_gap_filtered_4_6_month,
    start=0,
    end=26,
)
Cldicerad_match_PC_gap_mean_bad_filtered_4_6_month_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldicerad_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldicerad_2010_2019_match_PC_gap_filtered_4_6_month,
    start=0,
    end=26,
)

# Moderate atmospheric condition
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
Cldarea_match_PC_gap_mean_bad_filtered_1_3_month_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldarea_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldarea_2010_2019_match_PC_gap_filtered_1_3_month,
    start=34,
    end=64,
)
Cldtau_match_PC_gap_mean_bad_filtered_1_3_month_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldtau_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldtau_2010_2019_match_PC_gap_filtered_1_3_month,
    start=34,
    end=64,
)
Cldicerad_match_PC_gap_mean_bad_filtered_1_3_month_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldicerad_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldicerad_2010_2019_match_PC_gap_filtered_1_3_month,
    start=34,
    end=64,
)

# Euro and US test: Extract  Apr to Jun data from 2010 to 2019 and 2020
Cldarea_match_PC_gap_mean_bad_filtered_4_6_month_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldarea_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldarea_2010_2019_match_PC_gap_filtered_4_6_month,
    start=34,
    end=64,
)
Cldtau_match_PC_gap_mean_bad_filtered_4_6_month_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldtau_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldtau_2010_2019_match_PC_gap_filtered_4_6_month,
    start=34,
    end=64,
)
Cldicerad_match_PC_gap_mean_bad_filtered_4_6_month_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldicerad_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldicerad_2010_2019_match_PC_gap_filtered_4_6_month,
    start=34,
    end=64,
)

# Good atmospheric condition
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
Cldarea_match_PC_gap_mean_bad_filtered_1_3_month_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldarea_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldarea_2010_2019_match_PC_gap_filtered_1_3_month,
    start=64,
    end=116,
)
Cldtau_match_PC_gap_mean_bad_filtered_1_3_month_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldtau_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldtau_2010_2019_match_PC_gap_filtered_1_3_month,
    start=64,
    end=116,
)
Cldicerad_match_PC_gap_mean_bad_filtered_1_3_month_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldicerad_2020_match_PC_gap_filtered_1_3_month,
    Cld_all_match_PC_gap_others=Cldicerad_2010_2019_match_PC_gap_filtered_1_3_month,
    start=64,
    end=116,
)

# Euro and US test: Extract  Apr to Jun data from 2010 to 2019 and 2020
Cldarea_match_PC_gap_mean_bad_filtered_4_6_month_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldarea_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldarea_2010_2019_match_PC_gap_filtered_4_6_month,
    start=64,
    end=116,
)
Cldtau_match_PC_gap_mean_bad_filtered_4_6_month_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldtau_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldtau_2010_2019_match_PC_gap_filtered_4_6_month,
    start=64,
    end=116,
)
Cldicerad_match_PC_gap_mean_bad_filtered_4_6_month_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldicerad_2020_match_PC_gap_filtered_4_6_month,
    Cld_all_match_PC_gap_others=Cldicerad_2010_2019_match_PC_gap_filtered_4_6_month,
    start=64,
    end=116,
)
# endregion

filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=0,
    end=26,
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_bad_filtered,
    cld_min=-0.2,
    cld_max=0.2,
    # cld_name="HCF Diff (%)",
    # cld_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    # cld_name="Cld tau Diff",
    cld_name="Cld IRemiss Diff",
)


filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=34,
    end=64,
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered,
    cld_min=-0.3,
    cld_max=0.3,
    # cld_name="HCF Diff (%)",
    # cld_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    # cld_name="Cld tau Diff",
    cld_name="Cld IRemiss Diff",
)


filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=64,
    end=116,
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_good_filtered,
    cld_min=-0.2,
    cld_max=0.2,
    # cld_name="HCF Diff (%)",
    # cld_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    # cld_name="Cld tau Diff",
    cld_name="Cld IRemiss Diff",
)

########################################################################
###### Draw the Cld difference between 2020 and 2010-2019 ###############
###### But by latitude this time, 20N-60N, Contrail region ##############
########################################################################

#########################################################
############ smoothed data ##############################
#########################################################

# region
# Cldarea
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0=Cldarea_match_PC_gap_mean_bad_filtered_1_3_month_bad,
    Cld_data_PC_condition_1=Cldarea_match_PC_gap_mean_bad_filtered_1_3_month_moderate,
    Cld_data_PC_condition_2=Cldarea_match_PC_gap_mean_bad_filtered_1_3_month_good,
    Cld_data_aux=np.nanmean(
        Cldarea_2020_match_PC_gap_filtered_1_3_month, axis=0
    )
    - np.nanmean(
        Cldarea_2010_2019_match_PC_gap_filtered_1_3_month, axis=0
    ),
    Cld_data_name=r"$\Delta$" + "Cld Area Jan to Mar",
    step=5,
)

# Euro and US test: Extract  Apr to Jun data from 2010 to 2019 and 2020
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0=Cldarea_match_PC_gap_mean_bad_filtered_4_6_month_bad,
    Cld_data_PC_condition_1=Cldarea_match_PC_gap_mean_bad_filtered_4_6_month_moderate,
    Cld_data_PC_condition_2=Cldarea_match_PC_gap_mean_bad_filtered_4_6_month_good,
    Cld_data_aux=np.nanmean(
        Cldarea_2020_match_PC_gap_filtered_4_6_month, axis=0
    )
    - np.nanmean(
        Cldarea_2010_2019_match_PC_gap_filtered_4_6_month, axis=0
    ),
    Cld_data_name=r"$\Delta$" + "Cld Area Apr to Jun",
    step=5,
)

# Cldtau
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0=Cldtau_match_PC_gap_mean_bad_filtered_1_3_month_bad,
    Cld_data_PC_condition_1=Cldtau_match_PC_gap_mean_bad_filtered_1_3_month_moderate,
    Cld_data_PC_condition_2=Cldtau_match_PC_gap_mean_bad_filtered_1_3_month_good,
    Cld_data_aux=np.nanmean(
        Cldtau_2020_match_PC_gap_filtered_1_3_month, axis=0
    )
    - np.nanmean(
        Cldtau_2010_2019_match_PC_gap_filtered_1_3_month, axis=0
    ),
    Cld_data_name=r"$\Delta$" + "Cld Tau Jan to Mar",
    step=5,
)
# Euro and US test: Extract  Apr to Jun data from 2010 to 2019 and 2020
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0=Cldtau_match_PC_gap_mean_bad_filtered_4_6_month_bad,
    Cld_data_PC_condition_1=Cldtau_match_PC_gap_mean_bad_filtered_4_6_month_moderate,
    Cld_data_PC_condition_2=Cldtau_match_PC_gap_mean_bad_filtered_4_6_month_good,
    Cld_data_aux=np.nanmean(
        Cldtau_2020_match_PC_gap_filtered_4_6_month, axis=0
    )
    - np.nanmean(
        Cldtau_2010_2019_match_PC_gap_filtered_4_6_month, axis=0
    ),
    Cld_data_name=r"$\Delta$" + "Cld Tau Apr to Jun",
    step=5,
)

# Cldicerad
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0=Cldicerad_match_PC_gap_mean_bad_filtered_1_3_month_bad,
    Cld_data_PC_condition_1=Cldicerad_match_PC_gap_mean_bad_filtered_1_3_month_moderate,
    Cld_data_PC_condition_2=Cldicerad_match_PC_gap_mean_bad_filtered_1_3_month_good,
    Cld_data_aux=np.nanmean(
        Cldicerad_2020_match_PC_gap_filtered_1_3_month, axis=0
    )
    - np.nanmean(
        Cldicerad_2010_2019_match_PC_gap_filtered_1_3_month, axis=0
    ),
    Cld_data_name=r"$\Delta$" + "Cld Icerad Jan to Mar",
    step=5,
)
# Euro and US test: Extract  Apr to Jun data from 2010 to 2019 and 2020
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0=Cldicerad_match_PC_gap_mean_bad_filtered_4_6_month_bad,
    Cld_data_PC_condition_1=Cldicerad_match_PC_gap_mean_bad_filtered_4_6_month_moderate,
    Cld_data_PC_condition_2=Cldicerad_match_PC_gap_mean_bad_filtered_4_6_month_good,
    Cld_data_aux=np.nanmean(
        Cldicerad_2020_match_PC_gap_filtered_4_6_month, axis=0
    )
    - np.nanmean(
        Cldicerad_2010_2019_match_PC_gap_filtered_4_6_month, axis=0
    ),
    Cld_data_name=r"$\Delta$" + "Cld Icerad Apr to Jun",
    step=5,
)
# endregion


###########################################################
####### plot cld effective height by each PC condition ####
###########################################################

PC_2020_3_7_month = PC_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)

Cldeff_hgth_2020_3_7_month = Cldeff_hgth_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)


PC_2010_2019_3_7_month = PC_2010_2019.reshape(10, 12, 28, 180, 360)[
    :, 2:7, :, :, :
].reshape(10 * 5 * 28, 180, 360)
Cldeff_hgth_2010_2019_3_7_month = Cldeff_hgth_2010_2019.reshape(
    10, 12, 28, 180, 360
)[:, 2:7, :, :, :].reshape(10 * 5 * 28, 180, 360)

# filter data
filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=PC_2020_1_3_month, start=-2.5, end=5.5, gap=0.05
)

(
    Cldeff_hgth_2020_match_PC_gap_3_7_month,
    PC_2020_match_PC_gap_3_7_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_hgth_2020_3_7_month, PC_data=PC_2020_3_7_month
)

(
    Cldeff_hgth_2010_2019_match_PC_gap_3_7_month,
    PC_2010_2019_match_PC_gap_3_7_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_hgth_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)

# bad atmospheric condition
Cldeff_hgth_match_PC_gap_mean_bad_filtered_3_7_month_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldeff_hgth_2020_match_PC_gap_3_7_month,
    Cld_all_match_PC_gap_others=Cldeff_hgth_2010_2019_match_PC_gap_3_7_month,
    start=0,
    end=26,
)

# Moderate atmospheric condition
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
Cldeff_hgth_match_PC_gap_mean_bad_filtered_3_7_month_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldeff_hgth_2020_match_PC_gap_3_7_month,
    Cld_all_match_PC_gap_others=Cldeff_hgth_2010_2019_match_PC_gap_3_7_month,
    start=34,
    end=64,
)

# Good atmospheric condition
# China test: Extract Jan to Mar data from 2010 to 2019 and 2020
Cldeff_hgth_match_PC_gap_mean_bad_filtered_3_7_month_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cldeff_hgth_2020_match_PC_gap_3_7_month,
    Cld_all_match_PC_gap_others=Cldeff_hgth_2010_2019_match_PC_gap_3_7_month,
    start=64,
    end=116,
)

# plot
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0=Cldeff_hgth_match_PC_gap_mean_bad_filtered_3_7_month_bad,
    Cld_data_PC_condition_1=Cldeff_hgth_match_PC_gap_mean_bad_filtered_3_7_month_moderate,
    Cld_data_PC_condition_2=Cldeff_hgth_match_PC_gap_mean_bad_filtered_3_7_month_good,
    Cld_data_aux=np.nanmean(
        Cldeff_hgth_2020_match_PC_gap_3_7_month, axis=0
    )
    - np.nanmean(
        Cldeff_hgth_2010_2019_match_PC_gap_3_7_month, axis=0
    ),
    Cld_data_name=r"$\Delta$" + "Cld Effective Height",
    step=5,
)
