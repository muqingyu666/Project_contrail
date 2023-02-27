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
from muqy_20221026_func_filter_hcf_anormal_data import \
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error
from PIL import Image
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score

# --------- import done ------------
# --------- Plot style -------------

mpl.rc("font", family="Times New Roman")
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")


# ---------- Read PCA&CLD data from netcdf file --------

# ---------- Read in Cloud area data ----------
(
    PC_all_4_para,
    # PC_2010_2019_4_6_month,
    # PC_2017_2019_4_6_month,
    # PC_2020_4_6_month,
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
    # ------ Cloud data ------#
    Cld_all,
    # Cld_2010_2019_4_6_month,
    # Cld_2017_2019_4_6_month,
    # Cld_2020_4_6_month,
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
) = read_PC1_CERES_from_netcdf(
    PC_para_num=0, CERES_Cld_dataset_num=1
)
# 0 for Cldarea dataset, 1 for Cldicerad dataset
# 2 for Cldtau dataset, 3 for Cldtau_lin dataset, 4 for IWP dataset
# 5 for Cldemissirad dataset


# PC 0 means 4-para PCA, PC 1 means 5-para PCA

# --------- transform PC and Cld data to 3-7 month data ---------
# --------- in order to isolate the contrail cirrus ---------

#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################

# region
PC_2010_3_7_month = PC_2010.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2011_3_7_month = PC_2011.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2012_3_7_month = PC_2012.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2013_3_7_month = PC_2013.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2014_3_7_month = PC_2014.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2015_3_7_month = PC_2015.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2016_3_7_month = PC_2016.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2017_3_7_month = PC_2017.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2018_3_7_month = PC_2018.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2019_3_7_month = PC_2019.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
PC_2020_3_7_month = PC_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)

Cld_2010_3_7_month = Cld_2010.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2011_3_7_month = Cld_2011.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2012_3_7_month = Cld_2012.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2013_3_7_month = Cld_2013.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2014_3_7_month = Cld_2014.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2015_3_7_month = Cld_2015.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2016_3_7_month = Cld_2016.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2017_3_7_month = Cld_2017.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2018_3_7_month = Cld_2018.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2019_3_7_month = Cld_2019.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cld_2020_3_7_month = Cld_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
# endregion

# concatenate 3-7 month data
# to form a 2010->2019 dataset
PC_2010_2019_3_7_month = np.concatenate(
    [
        globals()[f"PC_{year}_3_7_month"]
        for year in range(2010, 2020)
    ],
    axis=0,
)

Cld_2010_2019_3_7_month = np.concatenate(
    [
        globals()[f"Cld_{year}_3_7_month"]
        for year in range(2010, 2020)
    ],
    axis=0,
)


# ------ Segmentation of cloud data within each PC interval ---------------------------------
#
filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=Cld_2010, start=-2.5, end=5.5, gap=0.05
)

# region
(
    Cld_2020_match_PC_gap_3_7_month,
    PC_2020_match_PC_gap_3_7_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)

(
    Cld_2010_2019_match_PC_gap_3_7_month,
    PC_2010_2019_match_PC_gap_3_7_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
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

# test only
(
    Cld_lowermost_error_2010_2019,
    Cld_highermost_error_2010_2019,
    Cld_2010_2019_match_PC_gap_filtered_3_7_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2010_2019_match_PC_gap_3_7_month
)

(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_3_7_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_3_7_month
)

# ----- Plot filtered data with boxplot ---------------------------------
# ----- Which is also the verification of the filtered data -------------

#######################################################################
########## verification of the filtered data ##########################
#######################################################################

# Box plot of cld data from 2010 to 2020 all data 4 parameters
# filter_data_fit_PC1_gap_plot.plot_box_plot(
#     Cld_all_match_PC_gap_4_para,
# )
# # Box plot of cld data from 2010 to 2020 all data 5 parameters
# filter_data_fit_PC1_gap_plot.plot_box_plot(
#     Cld_all_match_PC_gap_5_para,
# )


# Box plot of cld data from 2010 to 2020 filtered data 4 parameters
# filter_data_fit_PC1_gap_plot.plot_box_plot(
#     Cld_all_match_PC_gap_filtered_4_para,
# )
# Box plot of cld data from 2010 to 2020 filtered data 5 parameters
# filter_data_fit_PC1_gap_plot.plot_box_plot(
#     Cld_all_match_PC_gap_filtered_5_para,
# )


#######################################################################
#######################################################################
#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################


#################################################################################
###### Plot the HCF difference when no pc1 constrain are used ###################
#################################################################################
# Plot mean HCF of 2020
filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=0,
    end=18,
    Cld_match_PC_gap=np.nanmean(Cld_2020_3_7_month, axis=0),
    cld_min=0,
    cld_max=60,
    cld_name="HCF (%)",
)

# Plot mean HCF of 2010-2019
filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=0,
    end=18,
    Cld_match_PC_gap=np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    cld_min=0,
    cld_max=60,
    cld_name="HCF (%)",
)

# Plot mean HCF of 2020 subtract 2010-2019
filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=0,
    end=18,
    Cld_match_PC_gap=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    cld_min=-0.15,
    cld_max=0.15,
    # cld_name="HCF Diff (%)",
    # cld_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    # cld_name="Cld tau Diff",
    cld_name="Cld IRemiss Diff",
)

#################################################################################
###### Plot the filtered data with new filter method ############################
#################################################################################

# ------------------------------------------------------------
# Unfiltered version
# ------------------------------------------------------------
Cld_all_match_PC_gap_mean_sub_2020_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap_3_7_month,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap_3_7_month,
    start=0,
    end=26,
)

filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=0,
    end=26,
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_bad,
    cld_min=-0.2,
    cld_max=0.2,
    # cld_name="HCF Diff (%)",
    # cld_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    # cld_name="Cld tau Diff",
    cld_name="Cld IRemiss Diff",
)

# Moderate atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap_3_7_month,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap_3_7_month,
    start=34,
    end=64,
)

filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=34,
    end=64,
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_moderate,
    cld_min=-0.3,
    cld_max=0.3,
    # cld_name="HCF Diff (%)",
    # cld_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    # cld_name="Cld tau Diff",
    cld_name="Cld IRemiss Diff",
)


# Good atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap_3_7_month,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap_3_7_month,
    start=64,
    end=116,
)

filter_data_fit_PC1_gap_plot.plot_Cld_no_mean_full_hemisphere(
    start=64,
    end=116,
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_good,
    cld_min=-0.2,
    cld_max=0.2,
    # cld_name="HCF Diff (%)",
    # cld_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    # cld_name="Cld tau Diff",
    cld_name="Cld IRemiss Diff",
)

# --------------------------------------------------------------------------------------------
#  Filtered version
# --------------------------------------------------------------------------------------------
Cld_all_match_PC_gap_mean_sub_2020_bad_filtered = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap_filtered_3_7_month,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap_filtered_3_7_month,
    start=0,
    end=26,
)

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

# Moderate atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap_filtered_3_7_month,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap_filtered_3_7_month,
    start=34,
    end=64,
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


# Good atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_good_filtered = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap_filtered_3_7_month,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap_filtered_3_7_month,
    start=64,
    end=116,
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
###### But by latitude this time, 30N-60N, Contrail region ##############
########################################################################

#########################################################
############ smoothed data ##############################
#########################################################

# --------- 3-7 month -----------
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0=Cld_all_match_PC_gap_mean_sub_2020_bad_filtered,
    Cld_data_PC_condition_1=Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered,
    Cld_data_PC_condition_2=Cld_all_match_PC_gap_mean_sub_2020_good_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    Cld_data_name=r"$\Delta$" + "IPR (" + r"$\mu$" + r"m)",
    step=5,
)
