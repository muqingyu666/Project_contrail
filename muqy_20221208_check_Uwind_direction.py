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

    Code to check the correlation between PC1 and atmospheric variables
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2022-12-08
    
    Including the following parts:
        
        1) Read the PC1 and atmospheric variables data
        
        2) Plot the correlation between PC1 and atmospheric variables
        
"""

import glob
import os

import numpy as np
import pandas as pd
import scipy
import xarray as xr
from muqy_20220413_util_useful_functions import dcmap as dcmap
from muqy_20220519_sactter_plot import (
    scatter_plot_simulated_observed as scatter_plot,
)
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import norm, zscore

# 1) Read the PC1 and atmospheric variables data
# atmos para path
atmos_path = (
    "/RAID01/data/muqy/ERA5_RH_Z_data/2010_2020_300hPa_5_paras.nc"
)


(
    RelativeH_300,
    Temperature_300,
    Wvelocity_300,
    Stability_300,
    Uwind_300,
) = read_atmos_from_netcdf(atmos_path)

# reshape atmos to each year
RelativeH_300 = RelativeH_300.reshape(11, 12, 28, 180, 360).reshape(
    11, 336, 180, 360
)
Temperature_300 = Temperature_300.reshape(
    11, 12, 28, 180, 360
).reshape(11, 336, 180, 360)
Wvelocity_300 = Wvelocity_300.reshape(11, 12, 28, 180, 360).reshape(
    11, 336, 180, 360
)
Stability_300 = Stability_300.reshape(11, 12, 28, 180, 360).reshape(
    11, 336, 180, 360
)
Uwind_300 = Uwind_300.reshape(11, 12, 28, 180, 360).reshape(
    11, 336, 180, 360
)

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
    PC_para_num=2, CERES_Cld_dataset_num=0
)
# PC_para_num:
# 0 : 4para PC1
# 1 : 5para PC1
# 2 : 4para PC1 each grid point
# # 0 for Cldarea dataset, 1 for Cldicerad dataset
# 2 for Cldtau dataset, 3 for Cldtau_lin dataset, 4 for IWP dataset
# 5 for Cldemissirad dataset

PC_all_4_para = PC_all_4_para.reshape(11, 336, 180, 360)
Cld_all = Cld_all.reshape(11, 336, 180, 360)


# 2) Calc and Plot the correlation between PC1 and atmospheric variables
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


##############################################################################
#### plot main uwind direction ################################################
##############################################################################

for year in range(11):
    plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
        Cld_match_PC_gap=np.nanmean(Uwind_300, axis=1)[year],
        cld_min=-28,
        cld_max=28,
        cld_name="Uwind_300",
        cmap_file="/RAID01/data/muqy/color/test.txt",
    )

##############################################################################
######## RH - PC1 correlation test ########################################
##############################################################################


def plot_full_hemisphere_self_cmap(
    data,
    min,
    max,
    title,
    cb_label,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(11, 7),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=min,
        vmax=max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)
    ax1.set_title(title, fontsize=24)

    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.65,
        extend="both",
    )
    cb2.set_label(label=cb_label, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


plot_full_hemisphere_self_cmap(
    data=np.nanmean(RelativeH_300, axis=(0, 1)),
    min=10,
    max=90,
    title="mean RH",
    cb_label="RH (%)",
)

plot_full_hemisphere_self_cmap(
    data=np.nanmean(PC_all_4_para, axis=(1))[0],
    min=-0.5,
    max=0.5,
    title="mean PC1",
    cb_label="PC1",
)


plot_full_hemisphere_self_cmap(
    data=np.nanmean(Cld_all, axis=(1))[0],
    min=0,
    max=40,
    title="mean HCF",
    cb_label="HCF",
)

plot_full_hemisphere_self_cmap(
    data=np.nanmean(Stability_300, axis=(1))[0],
    min=0,
    max=0.0002,
    title="mean HCF",
    cb_label="HCF",
)


##############################################################################
######### Calc the correlation between PC1 and atmospheric variables #########
##############################################################################

# empty array to store the correlation
Corr_RH = np.zeros((11, 180, 360))
Corr_T = np.zeros((11, 180, 360))
Corr_W = np.zeros((11, 180, 360))
Corr_S = np.zeros((11, 180, 360))

# main loop
for year in range(11):
    Corr_RH[year], _ = calc_correlation_pvalue_PC1_Cld(
        PC_all_4_para[year], RelativeH_300[year]
    )
    Corr_T[year], _ = calc_correlation_pvalue_PC1_Cld(
        PC_all_4_para[year], Temperature_300[year]
    )
    Corr_W[year], _ = calc_correlation_pvalue_PC1_Cld(
        PC_all_4_para[year], Wvelocity_300[year]
    )
    Corr_S[year], _ = calc_correlation_pvalue_PC1_Cld(
        PC_all_4_para[year], Stability_300[year]
    )


# Calc the 11 years mean correlation
Corr_RH_all, _ = calc_correlation_pvalue_PC1_Cld(
    np.nanmean(PC_all_4_para, axis=0),
    np.nanmean(RelativeH_300, axis=0),
)
Corr_T_all, _ = calc_correlation_pvalue_PC1_Cld(
    np.nanmean(PC_all_4_para, axis=0),
    np.nanmean(Temperature_300, axis=0),
)
Corr_W_all, _ = calc_correlation_pvalue_PC1_Cld(
    np.nanmean(PC_all_4_para, axis=0),
    np.nanmean(Wvelocity_300, axis=0),
)
Corr_S_all, _ = calc_correlation_pvalue_PC1_Cld(
    np.nanmean(PC_all_4_para, axis=0),
    np.nanmean(Stability_300, axis=0),
)

################################################################################
### calc the correlation between HCF and atmospheric variables ##################
################################################################################

Cld_all = Cld_all.reshape(11, 336, 180, 360)

# empty array to store the correlation
Corr_RH_hcf = np.zeros((11, 180, 360))
Corr_T_hcf = np.zeros((11, 180, 360))
Corr_W_hcf = np.zeros((11, 180, 360))
Corr_S_hcf = np.zeros((11, 180, 360))

# main loop
for year in range(11):
    Corr_RH_hcf[year], _ = calc_correlation_pvalue_PC1_Cld(
        Cld_all[year], RelativeH_300[year]
    )
    Corr_T_hcf[year], _ = calc_correlation_pvalue_PC1_Cld(
        Cld_all[year], Temperature_300[year]
    )
    Corr_W_hcf[year], _ = calc_correlation_pvalue_PC1_Cld(
        Cld_all[year], Wvelocity_300[year]
    )
    Corr_S_hcf[year], _ = calc_correlation_pvalue_PC1_Cld(
        Cld_all[year], Stability_300[year]
    )


# Calc the 11 years mean correlation
Corr_RH_all_hcf, _ = calc_correlation_pvalue_PC1_Cld(
    np.nanmean(Cld_all, axis=0),
    np.nanmean(RelativeH_300, axis=0),
)
Corr_T_all_hcf, _ = calc_correlation_pvalue_PC1_Cld(
    np.nanmean(Cld_all, axis=0),
    np.nanmean(Temperature_300, axis=0),
)
Corr_W_all_hcf, _ = calc_correlation_pvalue_PC1_Cld(
    np.nanmean(Cld_all, axis=0),
    np.nanmean(Wvelocity_300, axis=0),
)
Corr_S_all_hcf, _ = calc_correlation_pvalue_PC1_Cld(
    np.nanmean(Cld_all, axis=0),
    np.nanmean(Stability_300, axis=0),
)

# Calc the 11 years corr between PC1 and HCF
Corr_PC1_HCF, _ = calc_correlation_pvalue_PC1_Cld(
    Cld_all.reshape(11 * 336, 180, 360),
    PC_all_4_para.reshape(11 * 336, 180, 360),
)

# plot the correlation


def plot_corr_full_hemisphere_self_cmap(
    Corr,
    # p_value,
    min,
    max,
    var_name,
    title,
    time,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    time_lst = [
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
    ]

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    lons, lats = np.meshgrid(lon, lat)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(11, 7),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Corr,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=min,
        vmax=max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)
    ax1.set_title(title, fontsize=24)

    # dot the significant area
    # dot_area = np.where(p_value < 0.00000005)
    # dot = ax1.scatter(
    #     lons[dot_area],
    #     lats[dot_area],
    #     color="k",
    #     s=3,
    #     linewidths=0,
    #     transform=ccrs.PlateCarree(),
    # )

    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.65,
        extend="both",
    )
    cb2.set_label(label="Corr", size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    os.makedirs("corr_hcf", exist_ok=True)
    plt.savefig(
        "corr_hcf/" + title + "_" + str(time_lst[time]) + ".png",
        dpi=300,
        facecolor="w",
    )


def plot_corr_tropical_self_cmap(
    Corr,
    # p_value,
    min,
    max,
    var_name,
    title,
    time,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    time_lst = [
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
    ]

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    lons, lats = np.meshgrid(lon, lat)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

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
    ax1.set_facecolor("silver")
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Corr,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=min,
        vmax=max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)
    ax1.set_title(title, fontsize=24)
    ax1.set_extent([-180, 180, -30, 30], ccrs.PlateCarree())

    # dot the significant area
    # dot_area = np.where(p_value < 0.00000005)
    # dot = ax1.scatter(
    #     lons[dot_area],
    #     lats[dot_area],
    #     color="k",
    #     s=3,
    #     linewidths=0,
    #     transform=ccrs.PlateCarree(),
    # )

    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.65,
        extend="both",
    )
    cb2.set_label(label="Corr", size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    os.makedirs("corr_hcf_trop", exist_ok=True)
    plt.savefig(
        "corr_hcf_trop/"
        + title
        + "_"
        + str(time_lst[time])
        + ".png",
        dpi=300,
        facecolor="w",
    )


# plot corr between PC1 and Cld full atmosphere
plot_corr_full_hemisphere_self_cmap(
    Corr_PC1_HCF, -1, 1, "PC1", "PC1-HCF", 0
)

# plot corr for full hemisphere
for time in range(11):
    plot_corr_full_hemisphere_self_cmap(
        Corr_RH[time], -1, 1, "RH", "PC1-RH", time
    )
    plot_corr_full_hemisphere_self_cmap(
        Corr_T[time], -1, 1, "T", "PC1-Temp", time
    )
    plot_corr_full_hemisphere_self_cmap(
        Corr_W[time], -1, 1, "W", "PC1-Wvelo", time
    )
    plot_corr_full_hemisphere_self_cmap(
        Corr_S[time], -1, 1, "S", "PC1-Unstab", time
    )


plot_corr_full_hemisphere_self_cmap(
    Corr_RH_all, -1, 1, "RH", "PC1-RH-all", 0
)
plot_corr_full_hemisphere_self_cmap(
    Corr_T_all, -1, 1, "T", "PC1-Temp-all", 0
)
plot_corr_full_hemisphere_self_cmap(
    Corr_W_all, -1, 1, "W", "PC1-Wvelo-all", 0
)
plot_corr_full_hemisphere_self_cmap(
    Corr_S_all, -1, 1, "S", "PC1-Unstab-all", 0
)

# plot corr for full hemisphere HCF&atmos paras
for time in range(11):
    plot_corr_full_hemisphere_self_cmap(
        Corr_RH_hcf[time], -1, 1, "RH", "HCF-RH", time
    )
    plot_corr_full_hemisphere_self_cmap(
        Corr_T_hcf[time], -1, 1, "T", "HCF-Temp", time
    )
    plot_corr_full_hemisphere_self_cmap(
        Corr_W_hcf[time], -1, 1, "W", "HCF-Wvelo", time
    )
    plot_corr_full_hemisphere_self_cmap(
        Corr_S_hcf[time], -1, 1, "S", "HCF-Unstab", time
    )

plot_corr_full_hemisphere_self_cmap(
    Corr_RH_all_hcf, -1, 1, "RH", "HCF-RH-all", 0
)
plot_corr_full_hemisphere_self_cmap(
    Corr_T_all_hcf, -1, 1, "T", "HCF-Temp-all", 0
)
plot_corr_full_hemisphere_self_cmap(
    Corr_W_all_hcf, -1, 1, "W", "HCF-Wvelo-all", 0
)
plot_corr_full_hemisphere_self_cmap(
    Corr_S_all_hcf, -1, 1, "S", "HCF-Unstab-all", 0
)

# plot corr for tropical
for time in range(11):
    plot_corr_tropical_self_cmap(
        Corr_RH[time], -1, 1, "RH", "PC1-RH", time
    )
    plot_corr_tropical_self_cmap(
        Corr_T[time], -1, 1, "T", "PC1-Temp", time
    )
    plot_corr_tropical_self_cmap(
        Corr_W[time], -1, 1, "W", "PC1-Wvelo", time
    )
    plot_corr_tropical_self_cmap(
        Corr_S[time], -1, 1, "S", "PC1-Unstab", time
    )

plot_corr_tropical_self_cmap(
    Corr_RH_all, -1, 1, "RH", "PC1-RH-all", 0
)
plot_corr_tropical_self_cmap(
    Corr_T_all, -1, 1, "T", "PC1-Temp-all", 0
)
plot_corr_tropical_self_cmap(
    Corr_W_all, -1, 1, "W", "PC1-Wvelo-all", 0
)
plot_corr_tropical_self_cmap(
    Corr_S_all, -1, 1, "S", "PC1-Unstab-all", 0
)

# plot corr for tropical HCF&atmos paras
for time in range(11):
    plot_corr_tropical_self_cmap(
        Corr_RH_hcf[time], -1, 1, "RH", "HCF-RH", time
    )
    plot_corr_tropical_self_cmap(
        Corr_T_hcf[time], -1, 1, "T", "HCF-Temp", time
    )
    plot_corr_tropical_self_cmap(
        Corr_W_hcf[time], -1, 1, "W", "HCF-Wvelo", time
    )
    plot_corr_tropical_self_cmap(
        Corr_S_hcf[time], -1, 1, "S", "HCF-Unstab", time
    )

plot_corr_tropical_self_cmap(
    Corr_RH_all_hcf, -1, 1, "RH", "HCF-RH-all", 0
)
plot_corr_tropical_self_cmap(
    Corr_T_all_hcf, -1, 1, "T", "HCF-Temp-all", 0
)
plot_corr_tropical_self_cmap(
    Corr_W_all_hcf, -1, 1, "W", "HCF-Wvelo-all", 0
)
plot_corr_tropical_self_cmap(
    Corr_S_all_hcf, -1, 1, "S", "HCF-Unstab-all", 0
)

############################################################
##### Filter the atmos para between each PC gap ############
############################################################

# empty array to store the filtered data
RelativeH_filtered = np.zeros((11, 160, 180, 360))
Temperature_filtered = np.zeros((11, 160, 180, 360))
Wvelocity_filtered = np.zeros((11, 160, 180, 360))
Stability_filtered = np.zeros((11, 160, 180, 360))

filter_atmos_fit_PC1 = FilterAtmosDataFitPCgap(
    Cld_data=Cld_2010, start=-2.5, end=5.5, gap=0.05
)

# main loop to filter the data
for time in range(11):
    RelativeH_filtered[
        time
    ] = filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_atmos(
        Atmos_data=RelativeH_300[time], PC_data=PC_all_4_para[time]
    )
    Temperature_filtered[
        time
    ] = filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_atmos(
        Atmos_data=Temperature_300[time],
        PC_data=PC_all_4_para[time],
    )
    Wvelocity_filtered[
        time
    ] = filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_atmos(
        Atmos_data=Wvelocity_300[time], PC_data=PC_all_4_para[time]
    )
    Stability_filtered[
        time
    ] = filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_atmos(
        Atmos_data=Stability_300[time], PC_data=PC_all_4_para[time]
    )

filter_atmos_fit_PC1_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=Cld_2010, start=-2.5, end=5.5, gap=0.05
)

for time in range(8, 11):
    filter_atmos_fit_PC1_plot.plot_Cld_simple_test_tropical(
        start=100,
        end=160,
        Cld_match_PC_gap=RelativeH_filtered[time],
        cld_min=50,
        cld_max=100,
        cld_name="RH (%)",
        cmap_file="/RAID01/data/muqy/color/test.txt",
    )

for time in range(8, 11):
    filter_atmos_fit_PC1_plot.plot_Cld_simple_test_tropical(
        start=100,
        end=160,
        Cld_match_PC_gap=Temperature_filtered[time],
        cld_min=237,
        cld_max=248,
        cld_name="Temperture (K)",
        cmap_file="/RAID01/data/muqy/color/test.txt",
    )

for time in range(8, 11):
    filter_atmos_fit_PC1_plot.plot_Cld_simple_test_tropical(
        start=100,
        end=160,
        Cld_match_PC_gap=Wvelocity_filtered[time],
        cld_min=-0.5,
        cld_max=0.5,
        cld_name="Wvelocity (m/s)",
        cmap_file="/RAID01/data/muqy/color/test.txt",
    )

for time in range(8, 11):
    filter_atmos_fit_PC1_plot.plot_Cld_simple_test_tropical(
        start=100,
        end=160,
        Cld_match_PC_gap=Stability_filtered[time],
        cld_min=-0.05,
        cld_max=0.05,
        cld_name="Unstability",
        cmap_file="/RAID01/data/muqy/color/test.txt",
    )


def plot_atms_tropics(atms_data, lat_start, lat_end, atms_name):
    # This code plots the 3 year mean of the PC1 for the 3 years of the dataset
    # The PC1 is the first principal component of the dataset
    # This is used to see the evolution of the principal component over the 3 years
    # The function is called in the main function

    fig = plt.figure(figsize=(9, 2))
    ax = fig.add_subplot(111)
    ax.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(
            atms_data[10, :, lat_start:lat_end, :], axis=(1, 2)
        ),
        label="2020",
        color="red",
        linewidth=0.7,
    )
    ax.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(
            atms_data[9, :, lat_start:lat_end, :], axis=(1, 2)
        ),
        label="2019",
        color="blue",
        linewidth=0.7,
    )
    ax.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(
            atms_data[8, :, lat_start:lat_end, :], axis=(1, 2)
        ),
        label="2018",
        color="green",
        linewidth=0.7,
    )
    ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel(atms_name)
    plt.show()


def plot_atms_each_climate(atms_data, atms_name):
    # This code plots the 3 year mean of the PC1 for the 3 years of the dataset
    # The PC1 is the first principal component of the dataset
    # This is used to see the evolution of the principal component over the 3 years
    # The function is called in the main function

    fig = plt.figure(figsize=(10, 6.5))
    ax1 = fig.add_subplot(611)
    ax1.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[10, :, 150:, :], axis=(1, 2)),
        label="2020",
        color="red",
        linewidth=0.7,
    )
    ax1.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[9, :, 150:, :], axis=(1, 2)),
        label="2019",
        color="blue",
        linewidth=0.7,
    )
    ax1.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[8, :, 150:, :], axis=(1, 2)),
        label="2018",
        color="green",
        linewidth=0.7,
    )
    ax1.set_ylabel(atms_name)
    plt.tick_params("x", labelbottom=False)

    ax2 = fig.add_subplot(612, sharex=ax1)
    ax2.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[10, :, 120:150, :], axis=(1, 2)),
        label="2020",
        color="red",
        linewidth=0.7,
    )
    ax2.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[9, :, 120:150, :], axis=(1, 2)),
        label="2019",
        color="blue",
        linewidth=0.7,
    )
    ax2.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[8, :, 120:150, :], axis=(1, 2)),
        label="2018",
        color="green",
        linewidth=0.7,
    )
    ax2.set_ylabel(atms_name)
    plt.tick_params("x", labelbottom=False)

    ax3 = fig.add_subplot(613, sharex=ax1)
    ax3.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[10, :, 90:120, :], axis=(1, 2)),
        label="2020",
        color="red",
        linewidth=0.7,
    )
    ax3.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[9, :, 90:120, :], axis=(1, 2)),
        label="2019",
        color="blue",
        linewidth=0.7,
    )
    ax3.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[8, :, 90:120, :], axis=(1, 2)),
        label="2018",
        color="green",
        linewidth=0.7,
    )
    ax3.set_ylabel(atms_name)
    plt.tick_params("x", labelbottom=False)

    ax4 = fig.add_subplot(614, sharex=ax1)
    ax4.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[10, :, 60:90, :], axis=(1, 2)),
        label="2020",
        color="red",
        linewidth=0.7,
    )
    ax4.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[9, :, 60:90, :], axis=(1, 2)),
        label="2019",
        color="blue",
        linewidth=0.7,
    )
    ax4.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[8, :, 60:90, :], axis=(1, 2)),
        label="2018",
        color="green",
        linewidth=0.7,
    )
    ax4.set_ylabel(atms_name)
    plt.tick_params("x", labelbottom=False)

    ax5 = fig.add_subplot(615, sharex=ax1)
    ax5.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[10, :, 30:60, :], axis=(1, 2)),
        label="2020",
        color="red",
        linewidth=0.7,
    )
    ax5.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[9, :, 30:60, :], axis=(1, 2)),
        label="2019",
        color="blue",
        linewidth=0.7,
    )
    ax5.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[8, :, 30:60, :], axis=(1, 2)),
        label="2018",
        color="green",
        linewidth=0.7,
    )
    ax5.set_ylabel(atms_name)
    plt.tick_params("x", labelbottom=False)

    ax6 = fig.add_subplot(616, sharex=ax1)
    ax6.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[10, :, 0:30, :], axis=(1, 2)),
        label="2020",
        color="red",
        linewidth=0.7,
    )
    ax6.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[9, :, 0:30, :], axis=(1, 2)),
        label="2019",
        color="blue",
        linewidth=0.7,
    )
    ax6.plot(
        np.arange(-2.5, 5.5, 0.05),
        np.nanmean(atms_data[8, :, 0:30, :], axis=(1, 2)),
        label="2018",
        color="green",
        linewidth=0.7,
    )
    ax6.set_ylabel(atms_name)
    ax6.set_xlabel("PC1")

    plt.show()


plot_atms_each_climate(RelativeH_filtered, "RH (%)")
plot_atms_each_climate(Temperature_filtered, "Temperature (K)")
plot_atms_each_climate(Wvelocity_filtered, "Wvelocity (m/s)")
plot_atms_each_climate(Stability_filtered, "Unstability")


plot_atms_tropics(RelativeH_filtered, 0, 180, "RH (%)")
plot_atms_tropics(Temperature_filtered, 0, 180, "Temperature (K)")
plot_atms_tropics(Wvelocity_filtered, 0, 180, "Wvelocity (m/s)")
plot_atms_tropics(Stability_filtered, 0, 180, "Unstability")


plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    np.nanmean(RelativeH_filtered[10, 0:160, :, :], axis=0),
    # p_value,
    cld_min=0,
    cld_max=100,
    cld_name="RH (%)",
)

plot_Cld_no_mean_simple_partial_self_cmap(
    np.nanmean(RelativeH_filtered[10, 0:160, 30:90, :], axis=0),
    # p_value,
    cld_min=0,
    cld_max=100,
    cld_name="RH (%)",
    lon=np.linspace(0, 359, 360),
    lat=np.linspace(-60, -1, 60),
    cmap_file="/RAID01/data/muqy/color/test.txt",
)

plot_Cld_no_mean_simple_tropical_self_cmap(
    np.nanmean(RelativeH_filtered[10, 0:160, :, :], axis=0),
    # p_value,
    cld_min=0,
    cld_max=100,
    cld_name="RH (%)",
)
