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

    Code for PCA-HCF analyze specially created for coefficient of variation, CV
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2022-06-28
    
    Including the following parts:
        
        1) Read the Pre-calculated PC1 and HCF data 

        2) Filter the data to fit the PC1 gap like -1.5 ~ 3.5
        
        3) Calculate the coefficient of variation for each PC1 gap
        
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
from numba import jit
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import norm, zscore
from sklearn.decomposition import PCA

# ----------  importing dcmap from my util ----------#
from muqy_20220413_util_useful_functions import dcmap as dcmap
from muqy_20220628_uti_PCA_CLD_analysis_function import *

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

#################################################################################
############## Filter_data_fit_PC1_gap ##########################################
#################################################################################

# --------- Divide 2017-2019 data into 2017,2018,2019 data ---------#
PC_2017_4_6_month = PC_2017_2019_4_6_month[:84, :, :]
PC_2018_4_6_month = PC_2017_2019_4_6_month[84:168, :, :]
PC_2019_4_6_month = PC_2017_2019_4_6_month[168:252, :, :]

Cld_2017_4_6_month = Cld_2017_2019_4_6_month[:84, :, :]
Cld_2018_4_6_month = Cld_2017_2019_4_6_month[84:168, :, :]
Cld_2019_4_6_month = Cld_2017_2019_4_6_month[168:252, :, :]


filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    start=-1.5, end=4.5, gap=0.05
)

# ------ Segmentation of cloud data within each PC interval ---------------------------------

# region

# ---------------- Compare hcf between 2011 & 2012 -----------------
# ---------------- Check if volcanal affects the function PCA
(
    Cld_2012_match_PC_gap,
    PC_2012_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2012, PC_data=PC_2012,
)

(
    Cld_2011_match_PC_gap,
    PC_2011_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2011, PC_data=PC_2011,
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


