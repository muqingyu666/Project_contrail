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
    PC_all_normal,
    # PC_all_global_grid,
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
    PC_path="/RAID01/data/PC_data/2010_2020_4_parameters_300hPa_PC1_metpy_unstab.nc",
    # PC_path="/RAID01/data/PC_data/2010_2020_4_parameters_300hPa_PC1_except_2010.nc",
    CERES_Cld_dataset_num=0,
)

(
    PC_all_each_year,
    # PC_all_global_grid,
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
    PC_path="/RAID01/data/PC_data/2010_2020_4_parameters_300hPa_PC1_each_year.nc",
    # PC_path="/RAID01/data/PC_data/2010_2020_4_parameters_300hPa_PC1_except_2010.nc",
    CERES_Cld_dataset_num=0,
)

(
    PC_all_except_2010,
    Cld_all,
) = read_PC1_CERES_from_netcdf_except_2010(
    PC_path="/RAID01/data/PC_data/2010_2020_4_parameters_300hPa_PC1_except_2010.nc",
    CERES_Cld_dataset_num=0,
)

PC_all_except_2010 = PC_all_except_2010.reshape(10, 336, 180, 360)

# reshape Cld to each year and year-month
Cld_all = Cld_all.reshape(11, 336, 180, 360)


# 2) Calc and Plot the correlation between PC1 and atmospheric variables
def calc_correlation_pvalue_PC1_Cld(PC_data, Cld_data):
    """Calculate the correlation and p-value between PC1 and cloud cover (Cld)."""

    # Pre-allocate empty arrays to store results
    Correlation_flat, P_value_flat = np.empty(
        (180 * 360)
    ), np.empty((180 * 360))
    Correlation, P_value = np.empty((180, 360)), np.empty(
        (180, 360)
    )

    # Reshape the data to a 2D array
    PC_data_flat = PC_data.reshape(PC_data.shape[0], -1)
    Cld_data_flat = Cld_data.reshape(PC_data.shape[0], -1)

    # Calculate the correlation and p-value for each grid point
    for grid in range(PC_data_flat.shape[1]):
        Correlation_flat[grid], P_value_flat[grid] = stats.pearsonr(
            PC_data_flat[:, grid], Cld_data_flat[:, grid]
        )

    # Reshape the results to a 2D array
    Correlation = Correlation_flat.reshape(180, 360)
    P_value = P_value_flat.reshape(180, 360)

    return Correlation, P_value


################################################################################
### calc the correlation between HCF and atmospheric variables ##################
################################################################################

Cld_all_except_2010 = Cld_all[1:, :, :, :]


# create empty array to store the correlation and p-value
Corr_PC_Cld_except_2010 = np.empty((180, 360))
# 11 means 11 years
Corr_PC_Cld_all_year = np.empty((11, 180, 360))
Corr_PC_Cld_each_year = np.empty((11, 180, 360))

(
    Corr_PC_Cld_except_2010,
    _,
) = calc_correlation_pvalue_PC1_Cld(
    Cld_all_except_2010.reshape(-1, 180, 360),
    PC_all_except_2010.reshape(-1, 180, 360),
)

(
    Corr_PC_Cld_all_year,
    _,
) = calc_correlation_pvalue_PC1_Cld(
    Cld_all.reshape(-1, 180, 360),
    PC_all_normal.reshape(-1, 180, 360),
)

for year in range(11):
    (
        Corr_PC_Cld_all_year[year],
        _,
    ) = calc_correlation_pvalue_PC1_Cld(
        Cld_all.reshape(11, 336, 180, 360)[year, :, :, :].reshape(
            -1, 180, 360
        ),
        PC_all_normal.reshape(11, 336, 180, 360)[
            year, :, :, :
        ].reshape(-1, 180, 360),
    )


for year in range(11):
    (
        Corr_PC_Cld_each_year[year],
        _,
    ) = calc_correlation_pvalue_PC1_Cld(
        Cld_all.reshape(11, 336, 180, 360)[year, :, :, :].reshape(
            -1, 180, 360
        ),
        PC_all_each_year.reshape(11, 336, 180, 360)[
            year, :, :, :
        ].reshape(-1, 180, 360),
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


def plot_full_hemisphere_self_cmap_muti(
    data,
    min,
    max,
    title,
    cb_label,
    save_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    """
    Plot the data on the full hemisphere

    Parameters
    ----------
    data : numpy.ndarray
        The data to be plotted
    min : float
        The minimum value of the data
    max : float
        The maximum value of the data
    title : str
        The title of the figure
    cb_label : str
        The label of the colorbar
    cmap_file : str, optional
        The path of the color map file, by default "/RAID01/data/muqy/color/test.txt"
    """
    # set the font
    plt.rcParams.update({"font.family": "Times New Roman"})

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over((146 / 255, 21 / 255, 25 / 255))
    # set color using RGB values
    cmap.set_under((27 / 255, 44 / 255, 98 / 255))

    # create the subplots
    fig, axs = plt.subplots(
        ncols=4,
        nrows=3,
        figsize=(28, 12),
        sharex=True,
        sharey=True,
        # constrained_layout=True,
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=0)
        },
    )
    axs = axs.flatten()

    for year in range(11):
        b = axs[year].pcolormesh(
            lon,
            lat,
            data[year],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=min,
            vmax=max,
        )
        axs[year].coastlines(resolution="50m", lw=0.9)
        # axs[year].set_title(title_list[year], fontsize=24)

        gl = axs[year].gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cb = fig.colorbar(
        b,
        cax=cax,
        location="right",
        pad=0.05,
        shrink=0.65,
        extend="both",
    )
    cb.set_label(label=cb_label, size=24)
    cb.ax.tick_params(labelsize=24)

    os.makedirs("/RAID01/data/python_fig", exist_ok=True)
    plt.savefig(
        "/RAID01/data/python_fig/" + save_name + ".png",
        dpi=300,
        facecolor="w",
        bbox_inches="tight",
    )


plot_corr_full_hemisphere_self_cmap(
    Corr_PC_Cld_except_2010 - Corr_PC_Cld_all_year,
    -0.1,
    0.1,
    "Corr",
    "PC1-HCF except 2010",
    0,
)

plot_corr_full_hemisphere_self_cmap(
    Corr_PC_Cld_all_year,
    -1,
    1,
    "Corr",
    "PC1-HCF each",
    0,
)

for year in range(11):
    plot_corr_full_hemisphere_self_cmap(
        Corr_PC_Cld_each_year[year] - Corr_PC_Cld_all_year[year],
        -1,
        1,
        "Corr",
        "PC1-HCF each",
        year,
    )

plot_full_hemisphere_self_cmap_muti(
    data=Corr_PC_Cld_each_year - Corr_PC_Cld_all_year,
    min=-0.5,
    max=0.5,
    title=r"$\Delta$" + "Corr",
    cb_label=r"$\Delta$" + "Corr",
    save_name="CorrDifference",
    cmap_file="/RAID01/data/muqy/color/Corr_color.txt",
)

############################################################
##### Filter the atmos para between each PC gap ############
############################################################


#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################
PC_all_global_grid = PC_all_global_grid.reshape(11 * 336, 180, 360)

# form a 2010->2019 dataset
PC_2010_2019 = np.concatenate(
    [globals()[f"PC_{year}"] for year in range(2010, 2020)],
    axis=0,
)

Cld_2010_2019 = np.concatenate(
    [globals()[f"Cld_{year}"] for year in range(2010, 2020)],
    axis=0,
)

# ------ Segmentation of cloud data within each PC interval ---------------------------------
filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    Cld_data=Cld_all, start=-1.5, end=6, gap=0.5
)

(
    Cld_all_match_PC_gap,
    PC_all_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_all,
    PC_data=PC_all_global_grid,
)

# ---------------------------------------------------------------
# ----------- Boxplot of PC1 and HCF ----------------------------
# --------------------------------------------------------------


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
        Box = np.zeros(
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


box_plot = Box_plot(Cld_match_PC_gap=Cld_all_match_PC_gap)
box_plot.plot_box_plot()

# ---------------------------------------------------------------


def plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    data,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

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
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)

    gl = ax1.gridlines(
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
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    data=np.nanmean(Cld_all_match_PC_gap[60:90], axis=0),
    cld_min=0,
    cld_max=10,
    cld_name="HCF (%)",
)
