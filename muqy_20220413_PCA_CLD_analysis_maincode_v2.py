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

    Code for PCA-HCF analyze
    
    Owner: Mu Qingyu
    version 2.0
        changes from v1.0-v2.0: remove functions to util_PCA_CLD_analysis_function
            to save some space
            
    Created: 2022-04-13
    
    Including the following parts:
        
        1) Read the Pre-calculated PC1 and HCF data 
        
        2) Filter the data to fit the PC1 gap like -1.5 ~ 3.5
        
        3) Plot boxplot to show the distribution of HCF data
        
        4) Compare HCF of different years in the same PC1 condition
                
        5) All sort of test code
        
"""

# ------------ PCA analysis ------------
# ------------ Start import ------------
import calendar
import glob
import os
import time
import tkinter
from itertools import product

import cartopy.crs as ccrs
import matplotlib
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from metpy.units import units

# ----------  importing dcmap from my util ----------#
from muqy_20220413_util_useful_functions import dcmap as dcmap
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from numba import jit
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import norm, zscore
from sklearn.decomposition import PCA

# ----------  done importing  ----------#

(
    PC_all,
    PC_2010_2019_4_6_month,
    PC_2017_2019_4_6_month,
    PC_2020_4_6_month,
    PC_2010_2019,
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
    ######### Read in CERES data #############
    Cld_all,
    Cld_2010_2019_4_6_month,
    Cld_2017_2019_4_6_month,
    Cld_2020_4_6_month,
    Cld_2010_2019,
    Cld_2018_2020,
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
) = read_PC1_CERES_from_netcdf()

# --------- Divide 2017-2019 data into 2017,2018,2019 data ---------#

PC_2017_4_6_month = PC_2017_2019_4_6_month[:84, :, :]
PC_2018_4_6_month = PC_2017_2019_4_6_month[84:168, :, :]
PC_2019_4_6_month = PC_2017_2019_4_6_month[168:252, :, :]

Cld_2017_4_6_month = Cld_2017_2019_4_6_month[:84, :, :]
Cld_2018_4_6_month = Cld_2017_2019_4_6_month[84:168, :, :]
Cld_2019_4_6_month = Cld_2017_2019_4_6_month[168:252, :, :]

#################################################################################
####### Filter_data_fit_PC1_gap #################################################
#################################################################################


import matplotlib.pyplot as plt

# Use the font and apply the matplotlib style
plt.rc("font", family="Times New Roman")
plt.style.use("seaborn")
# Reuse the same font to ensure that font set properly
plt.rc("font", family="Times New Roman")


filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    start=-1.5, end=4.5, gap=0.05
)

# ------ Segmentation of cloud data within each PC interval ---------------------------------

# region
(
    Cld_2017_match_PC_gap,
    PC_2017_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2017,
    PC_data=PC_2017,
)

(
    Cld_2018_match_PC_gap,
    PC_2018_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2018,
    PC_data=PC_2018,
)

(
    Cld_2019_match_PC_gap,
    PC_2019_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2019,
    PC_data=PC_2019,
)

(
    Cld_2020_match_PC_gap,
    PC_2020_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2020,
    PC_data=PC_2020,
)

(
    Cld_2010_2019_match_PC_gap,
    PC_2010_2019_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2010_2019,
    PC_data=PC_2010_2019,
)
# -------------- 2010-2019 4-6 month version ------------------------------------------------------------
# (
#     Cld_2020_match_PC_gap,
#     PC_2020_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2020_4_6_month, PC_data=PC_2020_4_6_month,
# )

# (
#     Cld_2010_2019_match_PC_gap,
#     PC_2010_2019_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2010_2019_4_6_month, PC_data=PC_2010_2019_4_6_month,
# )
# -------------- 2017-2019 4-6 month version ------------------------------------------------------------
# (
#     Cld_2017_match_PC_gap,
#     PC_2017_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2017_4_6_month, PC_data=PC_2017_4_6_month,
# )

# (
#     Cld_2018_match_PC_gap,
#     PC_2018_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2018_4_6_month, PC_data=PC_2018_4_6_month,
# )

# (
#     Cld_2019_match_PC_gap,
#     PC_2019_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2019_4_6_month, PC_data=PC_2019_4_6_month,
# )

# (
#     Cld_2020_match_PC_gap,
#     PC_2020_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2020_4_6_month, PC_data=PC_2020_4_6_month,
# )

# (
#     Cld_2010_2019_match_PC_gap,
#     PC_2010_2019_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2017_2019_4_6_month,
#     PC_data=PC_2017_2019_4_6_month,
# )

# endregion

# -----  Plot filtered data to verify ---------------------------------

filter_data_fit_PC1_gap_plot.plot_PC1_Cld(
    90, 110, PC_2010_2019_match_PC_gap, Cld_2010_2019_match_PC_gap
)

# ----- Plot global mean data to verify ---------------------------------

filter_data_fit_PC1_gap_plot.plot_All_year_mean_PC1_Cld(
    PC_all, Cld_all
)

# ----- Plot correlation between PC1 and Cld ---------------------------------

Correlation = filter_data_fit_PC1_gap_plot.calc_correlation_PC1_Cld(
    PC_all, Cld_all
)

filter_data_fit_PC1_gap_plot.plot_correlation_PC1_Cld(Correlation)

# ----- Plot filtered data with boxplot ---------------------------------


# Box plot of cld data from 2010 to 2019
box_plot = Box_plot(Cld_2010_2019_match_PC_gap, "2010to2019")
box_plot.plot_box_plot()
# Box plot of cld data 2020 only
box_plot = Box_plot(Cld_2020_match_PC_gap, "2020only")
box_plot.plot_box_plot()
# Box plot of cld data 2019 only
box_plot = Box_plot(Cld_2019_match_PC_gap, "2019only")
box_plot.plot_box_plot()
# Box plot of cld data 2018 only
box_plot = Box_plot(Cld_2018_match_PC_gap, "2018only")
box_plot.plot_box_plot()
# Box plot of cld data 2017 only
box_plot = Box_plot(Cld_2017_match_PC_gap, "2017only")
box_plot.plot_box_plot()


# region

# ----- Plot 2018-2020 Cld data& PC data ---------------------------------
# Bad atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    14,
    22,
    PC_2018_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2018_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=5,
)

# Good atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    52,
    60,
    PC_2018_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2018_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=20,
)

# ----- Plot 2019-2020 Cld data& PC data ---------------------------------
# Bad atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    14,
    22,
    PC_2019_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2019_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=5,
)

# Good atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    52,
    60,
    PC_2019_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2019_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=20,
)

# endregion

# ----- Plot mean-2020 Cld data& PC data ---------------------------------
# Bad atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    # 18,
    # 28,
    14,
    18,
    PC_2010_2019_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2010_2019_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=3,
)

# Good atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    # 46,
    # 56,
    56,
    60,
    PC_2010_2019_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2010_2019_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=18,
)

# ---------- Comparing the difference between other years and 2020 -------------------
# ---------- In the giving PC1 gap, and the giving atmospheric gap -------------------


# Calculate cld anomaly in fixed bad atmospheric conditions
# region
Cld_all_match_PC_gap_mean_sub_2020_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=14,
    end=18,
    # start=18,
    # end=28,
)
Cld_all_match_PC_gap_mean_sub_2017_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2017_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=14,
    end=18,
    # start=18,
    # end=28,
)
Cld_all_match_PC_gap_mean_sub_2018_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2018_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=14,
    end=18,
    # start=18,
    # end=28,
)
Cld_all_match_PC_gap_mean_sub_2019_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2019_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=14,
    end=18,
    # start=18,
    # end=28,
)
# endregion

# Calculate cld anomaly in fixed good atmospheric conditions
# region
Cld_all_match_PC_gap_mean_sub_2020_good = (
    compare_cld_between_2020_others(
        Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap,
        Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
        start=56,
        end=60,
    )
)
Cld_all_match_PC_gap_mean_sub_2017_good = (
    compare_cld_between_2020_others(
        Cld_all_match_PC_gap_2020=Cld_2017_match_PC_gap,
        Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
        start=56,
        end=60,
    )
)
Cld_all_match_PC_gap_mean_sub_2018_good = (
    compare_cld_between_2020_others(
        Cld_all_match_PC_gap_2020=Cld_2018_match_PC_gap,
        Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
        start=56,
        end=60,
    )
)
Cld_all_match_PC_gap_mean_sub_2019_good = (
    compare_cld_between_2020_others(
        Cld_all_match_PC_gap_2020=Cld_2019_match_PC_gap,
        Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
        start=56,
        end=60,
    )
)
# endregion

PC_all_match_PC_gap_mean_sub_2020_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=PC_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=PC_2010_2019_match_PC_gap,
    # start=14,
    # end=18,
    start=18,
    end=28,
)

# Good atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    # start=56,
    # end=60,
    start=46,
    end=56,
)
PC_all_match_PC_gap_mean_sub_2020_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=PC_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=PC_2010_2019_match_PC_gap,
    # start=56,
    # end=60,
    start=46,
    end=56,
)

# Plot the difference between the two years
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    # 56,
    # 60,
    46,
    56,
    PC_all_match_PC_gap_mean_sub_2020_good,
    Cld_all_match_PC_gap_mean_sub_2020_good,
    pc_max=0.02,
    cld_max=25,
)

# Plot 2017-2020 cld anormally in bad atmospheric conditions
# region
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2020_bad,
    cld_max=4,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2019_bad,
    cld_max=4,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2018_bad,
    cld_max=4,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2017_bad,
    cld_max=4,
)
# endregion

# Plot 2017-2020 cld anormally in good atmospheric conditions
# region
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    46,
    56,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2020_good,
    cld_max=23,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    46,
    56,
    Cld_all_match_PC_gap_mean_sub_2019_good,
    cld_max=23,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    46,
    56,
    Cld_all_match_PC_gap_mean_sub_2018_good,
    cld_max=23,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    46,
    56,
    Cld_all_match_PC_gap_mean_sub_2017_good,
    cld_max=23,
)
# endregion

# Plot 2020 Cld anormaly without atmospheric control
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    np.nanmean(PC_2017_2019_4_6_month, axis=0)
    - np.nanmean(PC_2020_4_6_month, axis=0),
    np.nanmean(Cld_2017_2019_4_6_month, axis=0)
    - np.nanmean(Cld_2020_4_6_month, axis=0),
    pc_max=1,
    cld_max=20,
)

# ---------- Compare the Cld for giving atmospheric condiction by latitude -------------------

plt.style.use("seaborn-whitegrid")
plt.rc("font", family="Times New Roman")


def compare_cld_between_mean_and_other_year_by_Lat(
    Cld_data_2020, Cld_data_2017, Cld_data_2018, Cld_data_2019
):
    """
    Plot mean cld anormaly from lat 30 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_2017[120:150, :], axis=0),
        linewidth=3,
        label="2017",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_2018[120:150, :], axis=0),
        linewidth=3,
        label="2018",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_2019[120:150, :], axis=0),
        linewidth=3,
        label="2019",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_2020[120:150, :], axis=0),
        color="red",
        label="2020",
        linewidth=3,
    )

    ax.set_facecolor("white")
    ax.legend()
    # adjust the legend font size
    for text in ax.get_legend().get_texts():
        plt.setp(text, color="black", fontsize=20)

    x_ticks_mark = [
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
    ]

    x_ticks = [-120, -60, 0, 60, 120]
    plt.xlim([-180, 180])
    plt.xlabel("Longitude", size=23, weight="bold")
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20, weight="bold")
    plt.yticks(fontsize=20, weight="bold")
    plt.ylabel("HCF difference (%)", size=20, weight="bold")
    # plt.ylim(0, 100)
    plt.show()


compare_cld_between_mean_and_other_year_by_Lat(
    # np.nanmean(Cld_all_match_PC_gap_mean_sub_2020_bad, axis=0),
    # np.nanmean(Cld_all_match_PC_gap_mean_sub_2019_bad, axis=0),
    # np.nanmean(Cld_all_match_PC_gap_mean_sub_2018_bad, axis=0),
    # np.nanmean(Cld_all_match_PC_gap_mean_sub_2017_bad, axis=0),
    # Cld_data_2020=Cld_all_match_PC_gap_mean_sub_2020_bad,
    # Cld_data_2017=Cld_all_match_PC_gap_mean_sub_2017_bad,
    # Cld_data_2018=Cld_all_match_PC_gap_mean_sub_2018_bad,
    # Cld_data_2019=Cld_all_match_PC_gap_mean_sub_2019_bad,
    Cld_data_2020=Cld_all_match_PC_gap_mean_sub_2020_good,
    Cld_data_2017=Cld_all_match_PC_gap_mean_sub_2017_good,
    Cld_data_2018=Cld_all_match_PC_gap_mean_sub_2018_good,
    Cld_data_2019=Cld_all_match_PC_gap_mean_sub_2019_good,
)
