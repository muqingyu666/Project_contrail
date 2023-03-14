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
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import norm, zscore

(
    PC_rotate_PCA,
    Cld_all,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/2010_2020_4_parameters_300hPa_PC1_varimax_rotate_PCA.nc",
    CERES_Cld_dataset_num=0,
)

PC_factor_analysis, _ = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/2010_2020_4_parameters_300hPa_PC1_factor_analysis.nc",
    CERES_Cld_dataset_num=0,
)

(
    PC_normal,
    _,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/2010_2020_4_parameters_300hPa_PC1_metpy_unstab.nc",
    CERES_Cld_dataset_num=0,
)

# extract data except polar region
PC_rotate_PCA = PC_rotate_PCA.reshape(11, 12, 28, 180, 360)
PC_factor_analysis = PC_factor_analysis.reshape(
    11, 12, 28, 180, 360
)
PC_normal = PC_normal.reshape(11, 12, 28, 180, 360)

Cld_all = Cld_all.reshape(11, 12, 28, 180, 360)


# 2) Calc and Plot the correlation between PC1 and atmospheric variables
def calc_correlation_pvalue_PC1_Cld(PC_data, Cld_data):
    Correlation = np.empty((Cld_data.shape[1], Cld_data.shape[2]))
    P_value = np.empty((Cld_data.shape[1], Cld_data.shape[2]))

    for i in range(Cld_data.shape[1]):
        for j in range(Cld_data.shape[2]):
            Correlation[i, j], P_value[i, j] = stats.pearsonr(
                pd.Series(PC_data[:, i, j]),
                pd.Series(Cld_data[:, i, j]),
            )

    return Correlation, P_value


################################################################################
### calc the correlation between HCF and atmospheric variables ##################
################################################################################

# Calc the 11 years corr between PC1 and HCF
Corr_PC1_HCF_rotate_PCA, _ = calc_correlation_pvalue_PC1_Cld(
    Cld_all.reshape(-1, 180, 360),
    PC_rotate_PCA.reshape(-1, 180, 360),
)
Corr_PC1_HCF_factor_analysis, _ = calc_correlation_pvalue_PC1_Cld(
    Cld_all.reshape(-1, 180, 360),
    PC_factor_analysis.reshape(-1, 180, 360),
)
Corr_PC1_HCF_normal, _ = calc_correlation_pvalue_PC1_Cld(
    Cld_all.reshape(-1, 180, 360), PC_normal.reshape(-1, 180, 360)
)


# -------------------------- plot the correlation -----------------------------
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
    cb2.set_label(label=var_name, size=24)
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


plot_corr_full_hemisphere_self_cmap(
    Corr_PC1_HCF_rotate_PCA, -1, 1, "Corr", "PC1-HCF global", 0
)

plot_corr_full_hemisphere_self_cmap(
    Corr_PC1_HCF_factor_analysis, -1, 1, "Corr", "PC1-HCF global", 0
)


plot_corr_full_hemisphere_self_cmap(
    Corr_PC1_HCF_normal, -1, 1, "Corr", "PC1-HCF global", 0
)


# plot the PC1-HCF correlation difference except polar region
# cause we only conduct PCA except polar region
plot_corr_full_hemisphere_self_cmap(
    Corr_PC1_HCF_normal - Corr_PC1_HCF_rotate_PCA,
    -0.1,
    0.1,
    "Corr",
    "Normal corr minus rotate one",
    0,
)

plot_corr_full_hemisphere_self_cmap(
    Corr_PC1_HCF_normal - Corr_PC1_HCF_factor_analysis,
    -0.3,
    0.3,
    "Corr",
    "Normal corr minus factor analysis",
    0,
)


############################################################
##### Filter the atmos para between each PC gap ############
############################################################

filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=Cld_all.reshape(-1, 120, 360),
    start=-2.5,
    end=5.5,
    gap=0.05,
)

(
    Cld_match_PC_gap_global_grid,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_all.reshape(-1, 120, 360),
    PC_data=PC_all_global_grid.reshape(-1, 120, 360),
)
(
    Cld_match_PC_gap_except_polar,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_all.reshape(-1, 120, 360),
    PC_data=PC_all_except_polar.reshape(-1, 120, 360),
)

(
    Cld_match_PC_gap_global_grid,
    _,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap_new(
    Cld_data=Cld_all.reshape(-1, 180, 360),
    PC_data=PC_all_global_grid.reshape(-1, 180, 360),
)


class Box_plot(object):
    """
    Plot boxplot of Cld data match each PC1 interval

    """

    def __init__(self, Cld_match_PC_gap):
        """
        Initialize the class

        Parameters
        ----------
        Cld_match_PC_gap : Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
        """
        # Input array must be in shape of (PC1_gap, lat, lon)
        self.Cld_match_PC_gap = Cld_match_PC_gap

    def Convert_pandas(self):
        gap_num = self.Cld_match_PC_gap.shape[0]
        Box = np.empty(
            (
                self.Cld_match_PC_gap.shape[1]
                * self.Cld_match_PC_gap.shape[2],
                gap_num,
            )
        )

        for i in range(gap_num):
            Box[:, i] = self.Cld_match_PC_gap[i, :, :].reshape(
                self.Cld_match_PC_gap.shape[1]
                * self.Cld_match_PC_gap.shape[2]
            )

        Box = pd.DataFrame(Box)
        # Number of PC1 interval is set in np.arange(start, end, interval gap)
        Box.columns = np.round(np.arange(-2.5, 5.5, 0.05), 3)

        return Box

    def plot_box_plot(self):
        """
        Plot boxplot of Cld data match each PC1 interval
        Main plot function
        """
        Box = self.Convert_pandas()

        fig, ax = plt.subplots(figsize=(18, 10))
        flierprops = dict(
            marker="o",
            markersize=7,
            markeredgecolor="grey",
        )
        Box.boxplot(
            # sym="o",
            flierprops=flierprops,
            whis=[10, 90],
            meanline=None,
            showmeans=True,
            notch=True,
        )
        plt.xlabel("PC1", size=26, weight="bold")
        plt.ylabel("HCF (%)", size=26, weight="bold")
        # plt.xticks((np.round(np.arange(-1.5, 3.5, 0.5), 2)),fontsize=26, weight="bold", )
        plt.yticks(
            fontsize=26,
            weight="bold",
        )
        # plt.savefig(
        #     "Box_plot_PC1_Cld.png",
        #     dpi=500,
        #     facecolor=fig.get_facecolor(),
        #     edgecolor="none",
        # )
        plt.show()


box_plot = Box_plot(Cld_match_PC_gap=Cld_match_PC_gap_except_polar)
box_plot.plot_box_plot()
box_plot = Box_plot(Cld_match_PC_gap=Cld_match_PC_gap_global_grid)
box_plot.plot_box_plot()


def error_bar_plot(data):
    """
    Plot error bar of Cld data match each PC1 interval

    Args:
        data (array): Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
    """
    # Input array must be in shape of (PC1_gap, lat, lon)
    # reshape data to (PC1_gap, lat*lon)
    data = data.reshape(data.shape[0], -1)

    # Calculate mean and std of each PC1 interval
    data_y = np.round(np.nanmean(data, axis=1), 3)
    data_x = np.round(np.arange(-2.5, 5.5, 0.05), 3)
    data_std = np.nanstd(data, axis=1)

    # Create a figure instance
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.errorbar(data_x, data_y, yerr=data_std, fmt="-o")

    # Add labels and title
    plt.xlabel("PC1")
    plt.ylabel("HCF (%)")
    # plt.title('')

    # Display the plot
    plt.show()


def error_fill_plot(data):
    """
    Plot error bar of Cld data match each PC1 interval

    Args:
        data (array): Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
    """
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

    # Create a figure instance
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(
        data_x,
        data_y,
        linewidth=3,
        color="#A3AECC",
    )
    plt.fill_between(
        data_x, data_up, data_down, facecolor="#A3AECC", alpha=0.5
    )

    # Add labels and title
    plt.xlabel("PC1")
    plt.ylabel("HCF (%)")
    # plt.title('')

    # Display the plot
    plt.show()


error_fill_plot(Cld_match_PC_gap_global_grid)

error_bar_plot(Cld_match_PC_gap_global_grid)

# ---------------------------------------------------------------
