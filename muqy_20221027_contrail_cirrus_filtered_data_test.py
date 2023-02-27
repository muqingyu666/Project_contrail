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
from PIL import Image
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
    PC_para_num=0, CERES_Cld_dataset_num=5
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
    (
        PC_2010_3_7_month,
        PC_2012_3_7_month,
        PC_2013_3_7_month,
        PC_2014_3_7_month,
        PC_2015_3_7_month,
        PC_2016_3_7_month,
        PC_2017_3_7_month,
        PC_2018_3_7_month,
        PC_2019_3_7_month,
        PC_2020_3_7_month,
    ),
    axis=0,
)

Cld_2010_2019_3_7_month = np.concatenate(
    (
        Cld_2010_3_7_month,
        Cld_2012_3_7_month,
        Cld_2013_3_7_month,
        Cld_2014_3_7_month,
        Cld_2015_3_7_month,
        Cld_2016_3_7_month,
        Cld_2017_3_7_month,
        Cld_2018_3_7_month,
        Cld_2019_3_7_month,
        Cld_2020_3_7_month,
    ),
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
    Cld_2019_match_PC_gap_3_7_month,
    PC_2019_match_PC_gap_3_7_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2019_3_7_month,
    PC_data=PC_2019_3_7_month,
)

(
    Cld_2018_match_PC_gap_3_7_month,
    PC_2018_match_PC_gap_3_7_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2018_3_7_month,
    PC_data=PC_2018_3_7_month,
)

(
    Cld_2017_match_PC_gap_3_7_month,
    PC_2017_match_PC_gap_3_7_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2017_3_7_month,
    PC_data=PC_2017_3_7_month,
)

(
    Cld_2010_2019_match_PC_gap_3_7_month,
    PC_2010_2019_match_PC_gap_3_7_month,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)

# all years pc1, test only
# (
#     Cld_all_match_PC_gap_5_para,
#     PC_all_match_PC_gap_5_para,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_all,
#     PC_data=PC_all_5_para,
# )

# (
#     Cld_all_match_PC_gap_4_para,
#     PC_all_match_PC_gap_4_para,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_all,
#     PC_data=PC_all_4_para,
# )
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

# (
#     Cld_lowermost_error_all_4_para,
#     Cld_highermost_error_all_4_para,
#     Cld_all_match_PC_gap_filtered_4_para,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_all_match_PC_gap_4_para
# )

# (
#     Cld_lowermost_error_all_5_para,
#     Cld_highermost_error_all_5_para,
#     Cld_all_match_PC_gap_filtered_5_para,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_all_match_PC_gap_5_para
# )

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

(
    Cld_lowermost_error_2019,
    Cld_highermost_error_2019,
    Cld_2019_match_PC_gap_filtered_3_7_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2019_match_PC_gap_3_7_month
)

(
    Cld_lowermost_error_2018,
    Cld_highermost_error_2018,
    Cld_2018_match_PC_gap_filtered_3_7_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2018_match_PC_gap_3_7_month
)

(
    Cld_lowermost_error_2017,
    Cld_highermost_error_2017,
    Cld_2017_match_PC_gap_filtered_3_7_month,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_match_PC_gap_3_7_month
)
# ----- Plot filtered data with boxplot ---------------------------------
# ----- Which is also the verification of the filtered data -------------

#######################################################################
########## verification of the filtered data ##########################
#######################################################################

# Box plot of cld data from 2010 to 2020 all data 4 parameters
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cld_all_match_PC_gap_4_para,
)
# Box plot of cld data from 2010 to 2020 all data 5 parameters
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cld_all_match_PC_gap_5_para,
)


# Box plot of cld data from 2010 to 2020 filtered data 4 parameters
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cld_all_match_PC_gap_filtered_4_para,
)
# Box plot of cld data from 2010 to 2020 filtered data 5 parameters
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cld_all_match_PC_gap_filtered_5_para,
)


##############################################################
######## coefs of variation of filtered data ##################
###############################################################


def calculate_coefs_of_variation(Cld_data):
    """
    Calculate coefs of variation of filtered data
    :param Cld_data:
    :return:
    """
    coefs_of_variation = []
    for i in range(len(Cld_data)):
        coefs_of_variation.append(
            np.nanstd(Cld_data[i]) / np.nanmean(Cld_data[i])
        )
    return np.array(coefs_of_variation)


def calculate_std(Cld_data):
    """
    Calculate std of filtered data
    :param Cld_data:
    :return:
    """
    std = []
    for i in range(len(Cld_data)):
        std.append(np.nanstd(Cld_data[i]))
    return np.array(std)


# coefs of variation of filtered data 4 parameters
coefs_of_variation_all_4_para = calculate_coefs_of_variation(
    Cld_all_match_PC_gap_filtered_4_para
)
# coefs of variation of filtered data 5 parameters
# coefs_of_variation_all_5_para = calculate_coefs_of_variation(
#     Cld_all_match_PC_gap_filtered_5_para
# )

# std of filtered data 4 parameters
std_all_4_para = calculate_std(Cld_all_match_PC_gap_filtered_4_para)
# std of filtered data 5 parameters
# std_all_5_para = calculate_std(Cld_all_match_PC_gap_filtered_5_para)


# ---- plot function for coefs of variation of filtered data -------------
def plot_lines(
    coefs_of_variation0,
    coefs_of_variation1,
    coefs_of_variation0_label,
    coefs_of_variation1_label,
    ylabel,
    fig_title,
):
    """
    Plot coefs of variation of filtered data
    :param coefs_of_variation:
    :return:
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        coefs_of_variation0,
        label=coefs_of_variation0_label,
    )
    plt.plot(
        coefs_of_variation1,
        label=coefs_of_variation1_label,
    )
    plt.xlabel("PC 1")
    plt.ylabel(ylabel)
    plt.title(fig_title)
    plt.grid()
    plt.legend()
    plt.show()


# plot_lines(
#     coefs_of_variation0=std_all_4_para,
#     coefs_of_variation1=std_all_5_para,
#     coefs_of_variation0_label="4 parameters",
#     coefs_of_variation1_label="5 parameters",
#     ylabel="Std",
#     fig_title="Std of filtered data",
# )

##################################################################
### Plot global altitude distribution  ###########################
##################################################################

# region
data_altitude = xr.open_dataset(
    "/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Global_altitude/ASurf_WFDE5_CRU_v2.1.nc"
)
data_altitude = np.array(data_altitude["ASurf"])[::2, ::2]

data_altitude_0, data_altitude_1 = np.split(
    data_altitude, 2, axis=1
)
data_altitude = np.concatenate(
    (data_altitude_1, data_altitude_0), axis=1
)

data_altitude_aux = data_altitude.copy()
data_altitude_aux[data_altitude_aux < 1600] = np.nan

plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    data_altitude_aux,
    0,
    5000,
    "Altitude (m)",
    cmap_file="/RAID01/data/muqy/color/color_b2g2y2r.txt",
)


# calculate the correlation coefficient of the filtered data
def calculate_correlation_coefficient(Cld_data):
    """
    Calculate the correlation coefficient of the filtered data
    :param Cld_data:
    :return:
    """
    correlation_coefficient = []
    Cld_data_aux = Cld_data.reshape(Cld_data.shape[0], -1)
    data_altitude_aux = data_altitude.reshape(-1)

    Cld_data_aux[np.isnan(Cld_data_aux)] = 0
    data_altitude_aux[np.isnan(data_altitude_aux)] = 0

    data_altitude_aux[data_altitude_aux < 4000] = 0

    for i in range(len(Cld_data)):
        correlation_coefficient.append(
            np.corrcoef(
                Cld_data_aux[i][~np.isnan(Cld_data_aux[i])],
                data_altitude_aux[~np.isnan(data_altitude_aux)],
            )[0, 1]
        )
    return np.array(correlation_coefficient)


correlation_coefficient_altitude_cld = (
    calculate_correlation_coefficient(
        Cld_2010_2019_match_PC_gap_filtered_3_7_month
    )
)
# endregion

###############################################################################

######################################################################
############### Calc and Plot correlation between pc1 and cld ########
######################################################################

Correlation_4_para = (
    filter_data_fit_PC1_gap_plot.calc_correlation_PC1_Cld(
        PC_all_4_para, Cld_all
    )
)

# Correlation_5_para = (
#     filter_data_fit_PC1_gap_plot.calc_correlation_PC1_Cld(
#         PC_all_5_para, Cld_all
#     )
# )

######### Calc linear regression and P-value of the correlation between pc1 and cld ########


def calc_correlation_pvalue_PC1_Cld(PC_data, Cld_data):
    Correlation = np.zeros((180, 360))
    P_value = np.zeros((180, 360))

    for i in range(180):
        for j in range(360):
            Correlation[i, j], P_value[i, j] = stats.pearsonr(
                pd.Series(PC_data[:, i, j]),
                pd.Series(Cld_data[:, i, j]),
            )

    return Correlation, P_value


def lm_trend(x):
    import statsmodels.api as sm

    if np.isnan(x).sum() > 25:
        return (np.nan, np.nan)
    else:
        years = np.arange(1, 34)
        years = sm.add_constant(years)
        model = sm.OLS(x, years)
        result = model.fit()
        # print(result.summary())
        slope = result.params[1]

        p = result.pvalues[1]
        return (slope, p)


(
    Correlation_4_para,
    P_value_4_para,
) = calc_correlation_pvalue_PC1_Cld(PC_all_4_para, Cld_all)

(
    Correlation_5_para,
    P_value_5_para,
) = calc_correlation_pvalue_PC1_Cld(PC_all_5_para, Cld_all)

# 使用xarray的apply_ufunc方法计算
# data_lm_trend = xr.apply_ufunc(
#     lm_trend,
#     data,
#     input_core_dims=[["year"]],
#     output_core_dims=[[], []],
#     vectorize=True,
# )

######### Plot correlation between pc1 and cld ########################

plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    Correlation_4_para,
    # P_value_4_para,
    cld_min=0,
    cld_max=1,
    cld_name="Corr",
    cmap_file="/RAID01/data/muqy/color/color_b2g2y2r.txt",
)

plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    Correlation_5_para,
    # P_value_4_para,
    cld_min=0,
    cld_max=1,
    cld_name="Corr",
    cmap_file="/RAID01/data/muqy/color/color_b2g2y2r.txt",
)

plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    Correlation_5_para - Correlation_4_para,
    # P_value_4_para,
    cld_min=-0.05,
    cld_max=0.05,
    cld_name="Corr diff",
)

plot_Cld_no_mean_simple_north_polar_self_cmap(
    Correlation_5_para - Correlation_4_para,
    cld_min=-0.05,
    cld_max=0.05,
    cld_name="Corr",
)

plot_Cld_no_mean_simple_sourth_polar_self_cmap(
    Correlation_5_para - Correlation_4_para,
    cld_min=-0.05,
    cld_max=0.05,
    cld_name="Corr",
)


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

# test only
# %matplotlib inline
# fig = plt.figure(figsize=(10, 10))
# plt.plot(np.arange(0,21),np.arange(0,21),color='black')
# plt.show()

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

# -------- Bad atmospheric conditions ------------
compare_cld_between_PC_condition_by_Lat(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_bad_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
)

# -------- Moderate atmospheric conditions ------------
compare_cld_between_PC_condition_by_Lat(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
)

# -------- Good atmospheric conditions ------------
compare_cld_between_PC_condition_by_Lat(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_good_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
)

#########################################################
############ smoothed data ##############################

compare_cld_between_PC_condition_by_each_Lat_smoothed(
    Cld_data_PC_condition_0=Cld_all_match_PC_gap_mean_sub_2020_bad_filtered,
    Cld_data_PC_condition_1=Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered,
    Cld_data_PC_condition_2=Cld_all_match_PC_gap_mean_sub_2020_good_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    step=5,
)

# expired code
compare_cld_between_PC_condition_by_Lat_smoothed(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_bad_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    step=5,
)

compare_cld_between_PC_condition_by_Lat_smoothed(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    step=5,
)

compare_cld_between_PC_condition_by_Lat_smoothed(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_good_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    step=5,
)
