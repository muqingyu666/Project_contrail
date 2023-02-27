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
# PC data and Cld data shape are all (11,12,28,180,360)
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


# --------- transform PC and Cld data to 3-7 month data ---------
# --------- in order to isolate the contrail cirrus ---------

#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################

# region
PC_2020_3_7_month = PC_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cldarea_2020_3_7_month = Cldarea_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cldtau_2020_3_7_month = Cldtau_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
CldtauL_2020_3_7_month = CldtauL_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cldemissir_2020_3_7_month = Cldemissir_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
Cldicerad_2020_3_7_month = Cldicerad_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cldphase_2020_3_7_month = Cldphase_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)
Cldpress_top_2020_3_7_month = Cldpress_top_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
Cldtemp_top_2020_3_7_month = Cldtemp_top_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
Cldhgth_top_2020_3_7_month = Cldhgth_top_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
Cldpress_base_2020_3_7_month = Cldpress_base_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
Cldtemp_base_2020_3_7_month = Cldtemp_base_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
Cldeff_press_2020_3_7_month = Cldeff_press_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
Cldeff_temp_2020_3_7_month = Cldeff_temp_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
Cldeff_hgth_2020_3_7_month = Cldeff_hgth_2020.reshape(
    12, 28, 180, 360
)[2:7, :, :, :].reshape(5 * 28, 180, 360)
IWP_2020_3_7_month = IWP_2020.reshape(12, 28, 180, 360)[
    2:7, :, :, :
].reshape(5 * 28, 180, 360)

PC_2010_2019_3_7_month = (
    PC_2010_2019.reshape(10, 12, 28, 180, 360)[:, 2:7, :, :, :]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldarea_2010_2019_3_7_month = (
    Cldarea_2010_2019.reshape(10, 12, 28, 180, 360)[:, 2:7, :, :, :]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldtau_2010_2019_3_7_month = (
    Cldtau_2010_2019.reshape(10, 12, 28, 180, 360)[:, 2:7, :, :, :]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
CldtauL_2010_2019_3_7_month = (
    CldtauL_2010_2019.reshape(10, 12, 28, 180, 360)[:, 2:7, :, :, :]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldemissir_2010_2019_3_7_month = (
    Cldemissir_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldicerad_2010_2019_3_7_month = (
    Cldicerad_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldphase_2010_2019_3_7_month = (
    Cldphase_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldpress_top_2010_2019_3_7_month = (
    Cldpress_top_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldtemp_top_2010_2019_3_7_month = (
    Cldtemp_top_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldhgth_top_2010_2019_3_7_month = (
    Cldhgth_top_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldpress_base_2010_2019_3_7_month = (
    Cldpress_base_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldtemp_base_2010_2019_3_7_month = (
    Cldtemp_base_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldeff_press_2010_2019_3_7_month = (
    Cldeff_press_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldeff_temp_2010_2019_3_7_month = (
    Cldeff_temp_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
Cldeff_hgth_2010_2019_3_7_month = (
    Cldeff_hgth_2010_2019.reshape(10, 12, 28, 180, 360)[
        :, 2:7, :, :, :
    ]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)
IWP_2010_2019_3_7_month = (
    IWP_2010_2019.reshape(10, 12, 28, 180, 360)[:, 2:7, :, :, :]
    .reshape(10, 5 * 28, 180, 360)
    .reshape(10 * 5 * 28, 180, 360)
)

# endregion

# ------ Segmentation of cloud data within each PC interval ---------------------------------
# make a aux data for the segmentation
Cld_2010 = Cldarea.reshape(11, 12, 28, 180, 360)[
    0, :, :, :, :
].reshape(-1, 180, 360)
filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=Cld_2010, start=-2.5, end=5.5, gap=0.05
)

# region
(
    Cldarea_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldarea_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldtau_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtau_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    CldtauL_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=CldtauL_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldemissir_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldemissir_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldicerad_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldicerad_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldpress_top_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldpress_top_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldtemp_top_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtemp_top_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldhgth_top_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldhgth_top_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldpress_base_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldpress_base_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldtemp_base_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtemp_base_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)
(
    Cldeff_press_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_press_2020_3_7_month, PC_data=PC_2020_3_7_month
)
(
    Cldeff_temp_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_temp_2020_3_7_month, PC_data=PC_2020_3_7_month
)
(
    Cldeff_hgth_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_hgth_2020_3_7_month, PC_data=PC_2020_3_7_month
)
(
    IWP_2020_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=IWP_2020_3_7_month,
    PC_data=PC_2020_3_7_month,
)

(
    Cldarea_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldarea_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldtau_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtau_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    CldtauL_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=CldtauL_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldemissir_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldemissir_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldicerad_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldicerad_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldpress_top_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldpress_top_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldtemp_top_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtemp_top_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldhgth_top_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldhgth_top_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldpress_base_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldpress_base_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldtemp_base_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtemp_base_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldeff_press_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_press_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldeff_temp_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_temp_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)
(
    Cldeff_hgth_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_hgth_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)

(
    IWP_2010_2019_match_PC_gap_3_7_month,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=IWP_2010_2019_3_7_month,
    PC_data=PC_2010_2019_3_7_month,
)

# all data
(
    Cldarea_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldarea.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldtau_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtau.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    CldtauL_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=CldtauL.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldemissir_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldemissir.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldicerad_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldicerad.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldpress_top_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldpress_top.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldtemp_top_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtemp_top.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldhgth_top_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldhgth_top.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldpress_base_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldpress_base.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldtemp_base_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldtemp_base.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldeff_press_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_press.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldeff_temp_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_temp.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    Cldeff_hgth_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cldeff_hgth.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)
(
    IWP_match_PC_gap,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=IWP.reshape(3696, 180, 360),
    PC_data=PC_all.reshape(3696, 180, 360),
)

# endregion

filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldarea_match_PC_gap, savefig_str="Cld_area (%)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldtau_match_PC_gap, savefig_str="Cld_tau"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    CldtauL_match_PC_gap, savefig_str="Cld_tauL"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldemissir_match_PC_gap, savefig_str="Cld_emissir"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldicerad_match_PC_gap, savefig_str="Cld_icerad (um)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldpress_top_match_PC_gap, savefig_str="Cld_press_top (hPa)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldtemp_top_match_PC_gap, savefig_str="Cld_temp_top (K)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldhgth_top_match_PC_gap, savefig_str="Cld_hgth_top (km)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldpress_base_match_PC_gap, savefig_str="Cld_press_base (hPa)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldtemp_base_match_PC_gap, savefig_str="Cld_temp_base (K)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    IWP_match_PC_gap, savefig_str="Cld_IWP (g m-2)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldeff_press_match_PC_gap, savefig_str="Cld_eff_press (hPa)"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldeff_temp_match_PC_gap, savefig_str="Cld_eff_temp (K))"
)
filter_data_fit_PC1_gap_plot.plot_box_plot(
    Cldeff_hgth_match_PC_gap, savefig_str="Cld_eff_hgth (km)"
)


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

######### Plot correlation between pc1 and cld ########################

plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    Correlation_4_para,
    # P_value_4_para,
    cld_min=0,
    cld_max=1,
    cld_name="Corr",
    cmap_file="/RAID01/data/muqy/color/color_b2g2y2r.txt",
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
    Cld_data_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    step=5,
)

# expired code
compare_cld_between_PC_condition_by_Lat_smoothed(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_bad_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    Cld_data_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    step=5,
)

compare_cld_between_PC_condition_by_Lat_smoothed(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_moderate_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    Cld_data_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    step=5,
)

compare_cld_between_PC_condition_by_Lat_smoothed(
    Cld_data_PC_condition=Cld_all_match_PC_gap_mean_sub_2020_good_filtered,
    Cld_data_aux=np.nanmean(Cld_2020_3_7_month, axis=0)
    - np.nanmean(Cld_2010_2019_3_7_month, axis=0),
    Cld_data_name="Cldicerad Diff (" + r"$\mu$" + r"m)",
    step=5,
)
