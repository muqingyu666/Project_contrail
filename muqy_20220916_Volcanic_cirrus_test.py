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

    Code to test if PCA method can be used to analyze 
    the cirrus caused by volcanic ashes.
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2022-09-16
    
    Including the following parts:

        1) Read in basic PCA & Cirrus data (include cirrus morphology and microphysics)  
        
        2) Data segmentation (include cirrus morphology and microphysics) 
          - we only need southern hemisphere data 
                
        3) 
        
"""

from statistics import mean

import matplotlib as mpl
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
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

# --------- import done ------------
# --------- Plot style -------------

plt.rc("font", family="Times New Roman")
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")

# ---------- Read PCA&CLD data from netcdf file --------

# ---------- Read in Cloud area data ----------
(
    PC_all,
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
    PC_para_num=1, CERES_Cld_dataset_num=0
)
# 0 for Cldarea dataset, 1 for Cldicerad dataset
# 2 for Cldtau dataset, 3 for Cldtau_lin dataset, 4 for IWP dataset
# 5 for Cldemissirad dataset

# PC 0 means 4-para PCA, PC 1 means 5-para PCA


# ---------- Read in Cloud icerad data ----------
(
    PC_all,
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
)  # 0 for Cldarea dataset, 1 for Cldicerad dataset
# 2 for Cldtau dataset, 3 for Cldtau_lin dataset, 4 for IWP dataset
# 5 for Cldemissirad dataset

# --------- transform PC and Cld data to 6-12 month data ---------
# --------- in order to isolate the volcanic cirrus ---------
# region
PC_2010_6_12_month = PC_2010.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2011_6_12_month = PC_2011.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2012_6_12_month = PC_2012.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2013_6_12_month = PC_2013.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2014_6_12_month = PC_2014.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2015_6_12_month = PC_2015.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2016_6_12_month = PC_2016.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2017_6_12_month = PC_2017.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2018_6_12_month = PC_2018.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2019_6_12_month = PC_2019.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
PC_2020_6_12_month = PC_2020.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)

Cld_2010_6_12_month = Cld_2010.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2011_6_12_month = Cld_2011.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2012_6_12_month = Cld_2012.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2013_6_12_month = Cld_2013.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2014_6_12_month = Cld_2014.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2015_6_12_month = Cld_2015.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2016_6_12_month = Cld_2016.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2017_6_12_month = Cld_2017.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2018_6_12_month = Cld_2018.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2019_6_12_month = Cld_2019.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
Cld_2020_6_12_month = Cld_2020.reshape(12, 28, 180, 360)[
    5:, :, :, :
].reshape(7 * 28, 180, 360)
# endregion

# --------- Read done ---------------------------------------------------
# --------- put 2010-2015 data together but no 2011 ---------------------
# --------- just in order to avoid volcanic eruption in 2011 ------------
# --------- also in order to calculate the 2010-2015 mean situation -----
PC_2010_2015_no_2011 = np.concatenate(
    (
        PC_2010.reshape(12, 28, 180, 360)[5:, :, :, :],
        PC_2012.reshape(12, 28, 180, 360)[5:, :, :, :],
        PC_2013.reshape(12, 28, 180, 360)[5:, :, :, :],
        PC_2014.reshape(12, 28, 180, 360)[5:, :, :, :],
        PC_2015.reshape(12, 28, 180, 360)[5:, :, :, :],
    ),
    axis=0,
)
Cld_2010_2015_no_2011 = np.concatenate(
    (
        Cld_2010.reshape(12, 28, 180, 360)[5:, :, :, :],
        Cld_2012.reshape(12, 28, 180, 360)[5:, :, :, :],
        Cld_2013.reshape(12, 28, 180, 360)[5:, :, :, :],
        Cld_2014.reshape(12, 28, 180, 360)[5:, :, :, :],
        Cld_2015.reshape(12, 28, 180, 360)[5:, :, :, :],
    ),
    axis=0,
)


# use total dataset trial
# 6-12 month data
PC_2010_2020_no_2011 = np.concatenate(
    (
        PC_2010_6_12_month,
        PC_2012_6_12_month,
        PC_2013_6_12_month,
        PC_2014_6_12_month,
        PC_2015_6_12_month,
        PC_2016_6_12_month,
        PC_2017_6_12_month,
        PC_2018_6_12_month,
        PC_2019_6_12_month,
        PC_2020_6_12_month,
    ),
    axis=0,
)
Cld_2010_2020_no_2011 = np.concatenate(
    (
        Cld_2010_6_12_month,
        Cld_2012_6_12_month,
        Cld_2013_6_12_month,
        Cld_2014_6_12_month,
        Cld_2015_6_12_month,
        Cld_2016_6_12_month,
        Cld_2017_6_12_month,
        Cld_2018_6_12_month,
        Cld_2019_6_12_month,
        Cld_2020_6_12_month,
    ),
    axis=0,
)

# all data
PC_2010_2020_no_2011 = np.concatenate(
    (
        PC_2010,
        PC_2012,
        PC_2013,
        PC_2014,
        PC_2015,
        PC_2016,
        PC_2017,
        PC_2018,
        PC_2019,
        PC_2020,
    ),
    axis=0,
)
Cld_2010_2020_no_2011 = np.concatenate(
    (
        Cld_2010,
        Cld_2012,
        Cld_2013,
        Cld_2014,
        Cld_2015,
        Cld_2016,
        Cld_2017,
        Cld_2018,
        Cld_2019,
        Cld_2020,
    ),
    axis=0,
)

# Done read in data

# ---------------------------------
# ---------- Plot zone ------------
# ---------------------------------
# Scatter plot of PC1 and Cldarea
# Just in order to verify the data
# scatter_plot(PC_all, Cld_all, "PC1", "CLD")


# Plot the PDF of Cldarea 2010-2015 under all PC1 conditions
# plot_cld_pdf(
#     Cld_2010_match_PC_gap.reshape(12, 90 * 360).transpose()
# )


# Data segmentation
# Using southern hemisphere data
# region
# Cld_2010_southern = Cld_2010[:, :90, :]
# PC_2010_southern = PC_2010[:, :90, :]

# Cld_2011_southern = Cld_2011[:, :90, :]
# PC_2011_southern = PC_2011[:, :90, :]

# Cld_2012_southern = Cld_2012[:, :90, :]
# PC_2012_southern = PC_2012[:, :90, :]

# Cld_2013_southern = Cld_2013[:, :90, :]
# PC_2013_southern = PC_2013[:, :90, :]

# Cld_2014_southern = Cld_2014[:, :90, :]
# PC_2014_southern = PC_2014[:, :90, :]

# Cld_2015_southern = Cld_2015[:, :90, :]
# PC_2015_southern = PC_2015[:, :90, :]


Cld_2010_southern = Cld_2010_6_12_month[:, :90, :]
PC_2010_southern = PC_2010_6_12_month[:, :90, :]

Cld_2011_southern = Cld_2011_6_12_month[:, :90, :]
PC_2011_southern = PC_2011_6_12_month[:, :90, :]

Cld_2012_southern = Cld_2012_6_12_month[:, :90, :]
PC_2012_southern = PC_2012_6_12_month[:, :90, :]

Cld_2013_southern = Cld_2013_6_12_month[:, :90, :]
PC_2013_southern = PC_2013_6_12_month[:, :90, :]

Cld_2014_southern = Cld_2014_6_12_month[:, :90, :]
PC_2014_southern = PC_2014_6_12_month[:, :90, :]

Cld_2015_southern = Cld_2015_6_12_month[:, :90, :]
PC_2015_southern = PC_2015_6_12_month[:, :90, :]

Cld_2010_2015_no_2011_southern = Cld_2010_2015_no_2011[:, :90, :]
PC_2010_2015_no_2011_southern = PC_2010_2015_no_2011[:, :90, :]

Cld_2010_2020_no_2011_southern = Cld_2010_2020_no_2011[:, :90, :]
PC_2010_2020_no_2011_southern = PC_2010_2020_no_2011[:, :90, :]
# endregion

# now all data have only the southern hemisphere data
# Cld/PC data now shape in (336, 90, 360)

# ------ Segmentation of cloud data within each PC interval ---------------------------------
#
filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=Cld_2010_southern, start=-1.5, end=4.5, gap=0.05
)

# region
(
    Cld_2010_match_PC_gap,
    PC_2010_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2010_southern,
    PC_data=PC_2010_southern,
)

(
    Cld_2011_match_PC_gap,
    PC_2011_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2011_southern,
    PC_data=PC_2011_southern,
)

(
    Cld_2012_match_PC_gap,
    PC_2012_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2012_southern,
    PC_data=PC_2012_southern,
)

(
    Cld_2013_match_PC_gap,
    PC_2013_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2013_southern,
    PC_data=PC_2013_southern,
)

(
    Cld_2014_match_PC_gap,
    PC_2014_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2014_southern,
    PC_data=PC_2014_southern,
)

(
    Cld_2015_match_PC_gap,
    PC_2015_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2015_southern,
    PC_data=PC_2015_southern,
)

(
    Cld_2010_2020_no_2011_match_PC_gap,
    PC_2010_2020_no_2011_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2010_2020_no_2011_southern,
    PC_data=PC_2010_2020_no_2011_southern,
)


# endregion

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ----- Filter out the data with 10% lowest or 90% highest HCF ----------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

(
    Cld_2010_2020_no_2011_lowermost_error,
    Cld_2010_2020_no_2011_highermost_error,
    Cld_2010_2020_no_2011_match_PC_gap_filtered,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2010_2020_no_2011_match_PC_gap
)

(
    Cld_2011_lowermost_error,
    Cld_2011_highermost_error,
    Cld_2011_match_PC_gap_filtered,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2011_match_PC_gap
)

# ----- Plot filtered data with boxplot ---------------------------------
# ----- Which is also the verification of the filtered data -------------

# Box plot of cld data from 2010 to 2020 all data
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cld_2010_2020_no_2011_match_PC_gap,
)

# Box plot of cld data from 2010 to 2020 filtered data
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cld_2010_2020_no_2011_match_PC_gap_filtered,
)


# ------------------------------------------------------------------
# ----- Alright, if the data pass the verification -----------------
# ----- Now lets start the real fight ------------------------------
# ------------------------------------------------------------------

#############################################################################

# ----- Plot 2011-(2010->2020 (but no 2011) mean) Cld data& PC data ---------------------------------

# Bad atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_Cld_Difference(
    0,
    2,
    Cld_2011_match_PC_gap - Cld_2010_2020_no_2011_match_PC_gap,
    cld_max=3,
)

# Moderate atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_Cld_Difference(
    1,
    3,
    Cld_2011_match_PC_gap - Cld_2010_2020_no_2011_match_PC_gap,
    cld_max=2.5,
)

# Good atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_Cld_Difference(
    3,
    8,
    Cld_2011_match_PC_gap - Cld_2010_2020_no_2011_match_PC_gap,
    cld_max=25,
)

########################################################################################
## this is the finest method to extract difference of cloud data #######################
## extract for each PC interval #######################################################
########################################################################################

# ----- Plot 2011-(2010->2020 (but no 2011) mean) Cld data& PC data ---------------------------------
#############################
# Bad atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2011_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2020_no_2011_match_PC_gap,
    start=0,
    end=12,
)

# normal platecarree plot of cld difference under bad atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_Cld_simple_shit(
    0,
    12,
    Cld_all_match_PC_gap_mean_sub_2020_bad,
    cld_max=2,
)

# southpolar stereographic plot of cld difference under bad atmospheric conditions
plot_Cld_no_mean_simple_sourth_polar_half_hemisphere(
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_bad,
    cld_min=-6,
    cld_max=6,
    cld_name="HCF difference (%)",
)

# southpolar stereographic plot of cld difference under bad atmospheric conditions
# filtered version
Cld_all_match_PC_gap_mean_sub_2020_bad_filtered = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2011_match_PC_gap_filtered,
    Cld_all_match_PC_gap_others=Cld_2010_2020_no_2011_match_PC_gap_filtered,
    start=0,
    end=12,
)

filter_data_fit_PC1_gap_plot.plot_Cld_simple_shit(
    0,
    12,
    Cld_all_match_PC_gap_mean_sub_2020_bad_filtered,
    cld_max=2,
)

plot_Cld_no_mean_simple_sourth_polar_half_hemisphere(
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_bad_filtered,
    cld_min=-3,
    cld_max=3,
    cld_name="HCF difference (%)",
)

##################################
# Moderate atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_moderate = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2011_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2020_no_2011_match_PC_gap,
    start=15,
    end=30,
)

filter_data_fit_PC1_gap_plot.plot_Cld_simple_shit(
    15,
    30,
    Cld_all_match_PC_gap_mean_sub_2020_moderate,
    cld_max=12,
)

# southpolar stereographic plot of cld difference under moderate atmospheric conditions
plot_Cld_no_mean_simple_sourth_polar_half_hemisphere(
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_moderate,
    cld_min=-24,
    cld_max=24,
    cld_name="HCF difference (%)",
)

# southpolar stereographic plot of cld difference under moderate atmospheric conditions
# filtered version
Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2011_match_PC_gap_filtered,
    Cld_all_match_PC_gap_others=Cld_2010_2020_no_2011_match_PC_gap_filtered,
    start=15,
    end=30,
)

filter_data_fit_PC1_gap_plot.plot_Cld_simple_shit(
    15,
    30,
    Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered,
    cld_max=12,
)

plot_Cld_no_mean_simple_sourth_polar_half_hemisphere(
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered,
    cld_min=-16,
    cld_max=16,
    cld_name="HCF difference (%)",
)

####################################
# Good atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2011_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2020_no_2011_match_PC_gap,
    start=38,
    end=60,
)

filter_data_fit_PC1_gap_plot.plot_Cld_simple_shit(
    38,
    60,
    Cld_all_match_PC_gap_mean_sub_2020_good,
    cld_max=33,
)

# southpolar stereographic plot of cld difference under good atmospheric conditions
plot_Cld_no_mean_simple_sourth_polar_half_hemisphere(
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_good,
    cld_min=-50,
    cld_max=50,
    cld_name="HCF difference (%)",
)

# southpolar stereographic plot of cld difference under good atmospheric conditions
# filtered version
Cld_all_match_PC_gap_mean_sub_2020_good_filtered = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2011_match_PC_gap_filtered,
    Cld_all_match_PC_gap_others=Cld_2010_2020_no_2011_match_PC_gap_filtered,
    start=38,
    end=60,
)

filter_data_fit_PC1_gap_plot.plot_Cld_simple_shit(
    38,
    60,
    Cld_all_match_PC_gap_mean_sub_2020_good_filtered,
    cld_max=33,
)

plot_Cld_no_mean_simple_sourth_polar_half_hemisphere(
    Cld_match_PC_gap=Cld_all_match_PC_gap_mean_sub_2020_good_filtered,
    cld_min=-50,
    cld_max=50,
    cld_name="HCF difference (%)",
)


####################################################
########## test spatial distribution ###############
####################################################


def plot_Cld(
    Cld_data,
    cld_max,
):

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
        Cld_data,
        transform=ccrs.PlateCarree(),
        norm=norm2,
        cmap=cmap,
    )
    ax1.coastlines(resolution="50m", lw=0.9)  # type: ignore
    gl = ax1.gridlines(  # type: ignore
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

    plt.show()


def plot_Cld_0(Cld_data, cld_max):

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
    norm2 = colors.CenteredNorm(halfrange=cld_max)
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_data,
        transform=ccrs.PlateCarree(),
        norm=norm2,
        cmap=cmap,
    )
    ax1.coastlines(resolution="50m", lw=0.9)  # type: ignore
    gl = ax1.gridlines(  # type: ignore
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

    plt.show()


plot_Cld(
    np.nanmean(Cld_2011[:, :90, :], axis=0)
    - np.nanmean(Cld_all[:, :90, :], axis=0),
    5,
)

plot_Cld_0(
    np.nanmean(Cld_2011, axis=0) - np.nanmean(Cld_all, axis=0), 7
)


########################################################################
####### abandon part pdf plot ##########################################
########################################################################

# ----- Plot cld pdf for each PC interval test ver ---------------------------------

Cld_2010_2020_no_2011_match_PC_gap_dateframe = pd.DataFrame(
    Cld_2010_2020_no_2011_match_PC_gap.reshape(
        12, 90 * 360
    ).transpose()
)

Cld_2010_2020_no_2011_match_PC_gap_dateframe.columns = [
    "-1.5 < PC1 < -1",
    "-1 < PC1 < -0.5",
    "-0.5 < PC1 < 0",
    "0 < PC1 < 0.5",
    "0.5 < PC1 < 1",
    "1 < PC1 < 1.5",
    "1.5 < PC1 < 2",
    "2 < PC1 < 2.5",
    "2.5 < PC1 < 3",
    "3 < PC1 < 3.5",
    "3.5 < PC1 < 4",
    "4 < PC1 < 4.5",
]  # type: ignore

Cld_2010_2020_no_2011_match_PC_gap_dateframe = pd.DataFrame(
    Cld_2010_2020_no_2011_match_PC_gap.reshape(
        12, 90 * 360
    ).transpose()
)


def plot_cld_pdf(cld_data, cld_min, cld_max, title):
    # Statistics of PC1 and Cldarea
    # Try to find fingerprint of volcanic cirrus using statistics
    # PDF of Cld under certain PC1 conditions
    """
    Plot the PDF of Cldarea under certain PC1 conditions

    Args:
        cld_data (array): Cld data
        title (str): title of the plot
    """
    sns.set_theme()

    sns.set_style(
        "ticks",
        {
            # "axes.spines.top": True,
            # "axes.spines.right": True,
            "font.family": ["Times New Roman"],
        },
    )
    # sns.despine(right=True, top=True)
    pl = sns.displot(
        cld_data,
        kind="kde",
        fill=True,
        # height=5,
        # aspect=2,
    )
    # sns.despine()
    sns.move_legend(pl, "upper right", bbox_to_anchor=(0.72, 0.95))
    pl.fig.set_figwidth(12)
    pl.fig.set_figheight(5)
    pl.fig.set_dpi(200)
    pl.set(xlabel="Cldarea")
    plt.xlim(cld_min, cld_max)
    plt.title(title)
    # plt.savefig('./distribution.svg', format='svg', dpi=300)
    plt.show()


def plot_cld_pdf_2010_2015_for_each_PC_interval(
    start, end, cld_min, cld_max, figure_title
):

    # mean the cld data for specified PC interval
    # and reshape to (1,90*360) shape
    # region
    Cld_2010_match_PC_gap_specified = np.expand_dims(
        np.nanmean(
            Cld_2010_match_PC_gap.reshape(12, 90 * 360)[
                start:end, :
            ],
            axis=0,
        ),
        axis=0,
    )
    Cld_2011_match_PC_gap_specified = np.expand_dims(
        np.nanmean(
            Cld_2011_match_PC_gap.reshape(12, 90 * 360)[
                start:end, :
            ],
            axis=0,
        ),
        axis=0,
    )
    Cld_2012_match_PC_gap_specified = np.expand_dims(
        np.nanmean(
            Cld_2012_match_PC_gap.reshape(12, 90 * 360)[
                start:end, :
            ],
            axis=0,
        ),
        axis=0,
    )
    Cld_2013_match_PC_gap_specified = np.expand_dims(
        np.nanmean(
            Cld_2013_match_PC_gap.reshape(12, 90 * 360)[
                start:end, :
            ],
            axis=0,
        ),
        axis=0,
    )
    Cld_2014_match_PC_gap_specified = np.expand_dims(
        np.nanmean(
            Cld_2014_match_PC_gap.reshape(12, 90 * 360)[
                start:end, :
            ],
            axis=0,
        ),
        axis=0,
    )
    Cld_2015_match_PC_gap_specified = np.expand_dims(
        np.nanmean(
            Cld_2015_match_PC_gap.reshape(12, 90 * 360)[
                start:end, :
            ],
            axis=0,
        ),
        axis=0,
    )
    # endregion

    Cld_2010_2015 = np.concatenate(
        (
            Cld_2010_match_PC_gap_specified,
            Cld_2011_match_PC_gap_specified,
            Cld_2012_match_PC_gap_specified,
            Cld_2013_match_PC_gap_specified,
            Cld_2014_match_PC_gap_specified,
            Cld_2015_match_PC_gap_specified,
        ),
        axis=0,
    ).transpose()
    Cld_2010_2015_dateframe = pd.DataFrame(Cld_2010_2015)
    Cld_2010_2015_dateframe.columns = [
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
    ]  # type: ignore
    plot_cld_pdf(
        cld_data=Cld_2010_2015_dateframe,
        cld_min=cld_min,
        cld_max=cld_max,
        title=figure_title,
    )


def plot_cld_pdf_2010_2020_for_each_PC_interval(
    start, end, cld_min, cld_max, figure_title
):

    # mean the cld data for specified PC interval
    # and reshape to (1,90*360) shape
    # region
    Cld_2011_match_PC_gap_specified = np.expand_dims(
        np.nanmean(
            Cld_2011_match_PC_gap.reshape(12, 90 * 360)[
                start:end, :
            ],
            axis=0,
        ),
        axis=0,
    )
    Cld_2010_2020_match_PC_gap_specified = np.expand_dims(
        np.nanmean(
            Cld_2010_2020_no_2011_match_PC_gap.reshape(
                12, 90 * 360
            )[start:end, :],
            axis=0,
        ),
        axis=0,
    )
    # endregion

    Cld_2010_2015 = np.concatenate(
        (
            Cld_2011_match_PC_gap_specified,
            Cld_2010_2020_match_PC_gap_specified,
        ),
        axis=0,
    ).transpose()
    Cld_2010_2015_dateframe = pd.DataFrame(Cld_2010_2015)
    Cld_2010_2015_dateframe.columns = [
        "2011",
        "2010-2020 but 2011",
    ]  # type: ignore
    plot_cld_pdf(
        cld_data=Cld_2010_2015_dateframe,
        cld_min=cld_min,
        cld_max=cld_max,
        title=figure_title,
    )


# bad condition 0-2
# moderate condition 1-3
# good condition 3-8
# 2010 -> 2015 each year pdf
plot_cld_pdf_2010_2015_for_each_PC_interval(
    0,
    2,
    cld_min=0,
    cld_max=2,
    figure_title="PDF for HCF 2010-2015 bad condition",
)

plot_cld_pdf_2010_2015_for_each_PC_interval(
    1,
    3,
    cld_min=0,
    cld_max=5,
    figure_title="PDF for HCF 2010-2015 moderate condition",
)

plot_cld_pdf_2010_2015_for_each_PC_interval(
    3,
    8,
    cld_min=0,
    cld_max=30,
    figure_title="PDF for HCF 2010-2015 good condition",
)

# bad condition 0-2
# moderate condition 1-3
# good condition 3-8
# 2011 vs 2010-2020 no 2011 pdf
plot_cld_pdf_2010_2020_for_each_PC_interval(
    0,
    2,
    cld_min=0,
    cld_max=2,
    figure_title="PDF for HCF 2010-2020 bad condition",
)

plot_cld_pdf_2010_2020_for_each_PC_interval(
    1,
    3,
    cld_min=0,
    cld_max=5,
    figure_title="PDF for HCF 2010-2020 moderate condition",
)

plot_cld_pdf_2010_2020_for_each_PC_interval(
    3,
    8,
    cld_min=0,
    cld_max=30,
    figure_title="PDF for HCF 2010-2020 good condition",
)

# plot_cld_pdf(
#     Cld_2010_2020_no_2011_match_PC_gap_dateframe,
#     "PDF for HCF 2010-2020 without 2011",
# )

# ----- Data density scatter plot --------------------------------------

plt.style.use("seaborn-deep")  # type: ignore

config = {
    "font.family": "Times New Roman",
    # "font.size": 16,
    "mathtext.fontset": "stix",
}
rcParams.update(config)


def scatter_out_1(x, y):  ## x,y为两个需要做对比分析的两个量。
    # =========== Calculate the point density ==========
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)
    # ===== Sort the points by density, so that ===========
    # ===== the densest points are plotted last ===========
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    (
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
    ) = stats.linregress(x, y)
    x_fit = np.linspace(0, x.max(), 100)
    y_fit = slope * x_fit + intercept

    # def best_fit_slope_and_intercept(xs, ys):
    #     m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (
    #         (mean(xs) * mean(xs)) - mean(xs * xs)
    #     )
    #     b = mean(ys) - m * mean(xs)
    #     return m, b

    # m, b = best_fit_slope_and_intercept(x, y)
    # regression_line = []
    # for a in x:
    #     regression_line.append((m * a) + b)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
    scatter = ax.scatter(
        x,
        y,
        marker="o",  # type: ignore
        c=z * 100,
        s=15,
        label="LST",
        cmap="Spectral_r",
    )
    cbar = plt.colorbar(
        scatter,
        shrink=1,
        orientation="vertical",
        extend="both",
        pad=0.015,
        aspect=30,
        label="frequency",
    )
    plt.plot(
        x_fit,
        y_fit,
        "r-",
        lw=1,
        label="slope = %.3f" % slope,
        lineWidth=4,
    )
    plt.annotate(
        "R = %.3f" % r_value,
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        fontsize=24,
    )
    plt.annotate(
        "slope = %.3f" % slope,
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        fontsize=24,
    )

    # plt.plot(x, regression_line, "red", lw=1.5)  # 预测与实测数据之间的回归线
    # plt.axis([0, 25, 0, 25])  # 设置线的范围
    plt.xlabel("PC1", family="Times New Roman", fontsize=24)
    plt.ylabel("HCF", family="Times New Roman", fontsize=24)
    plt.xticks(fontproperties="Times New Roman")
    plt.yticks(fontproperties="Times New Roman")
    # plt.xlim(0, 25)  # 设置x坐标轴的显示范围
    # plt.ylim(0, 25)  # 设置y坐标轴的显示范围
    # plt.savefig(
    #     "F:/Rpython/lp37/plot71.png",
    #     dpi=800,
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )
    plt.show()


scatter_out_1(
    PC_2010_2020_no_2011.reshape(-1),
    Cld_2010_2020_no_2011.reshape(-1),
)
