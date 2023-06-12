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

# import modules
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
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)
from PIL import Image
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
# now we read IWP and other cld data (not IWP) from netcdf file
(
    # pc
    PC_all,
    PC_years,
    # cld
    Cld_all,
    Cld_years,
    # iwp
    IWP_data,
    IWP_years,
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_num=0)

# 0 for Cldarea dataset, 1 for Cldicerad dataset
# 2 for Cldtau dataset, 3 for Cldtau_lin dataset, 4 for IWP dataset
# 5 for Cldemissirad dataset

# use the 2010-2020 PC1 only
PC_all = PC_all[-11:]

# read in the correlation coefficient and pvalue
data = xr.open_dataset(
    "corr_data/correlation_pvalue_PC1_Cldarea.nc"
)
correlation = data["correlation"].values
pvalue = data["p_values"].values

# Mask the PC1 and Cldarea data where the correlation coefficients are less than 0.45
# Assuming PC_years is a dictionary with 11 arrays and correlation has shape (180, 360)
mask = correlation < 0.45

for year in range(2017, 2021):
    PC_years[year] = np.ma.masked_array(
        PC_years[year],
        mask=np.repeat(
            mask[np.newaxis, ...], 12 * 28, axis=0
        ).reshape(12, 28, 180, 360),
    )
    Cld_years[year] = np.ma.masked_array(
        Cld_years[year],
        mask=np.repeat(
            mask[np.newaxis, ...], 12 * 28, axis=0
        ).reshape(12, 28, 180, 360),
    )


# Extract the PC1 and Cldarea data for each year
def extract_PC1_CERES_each_year(
    PC_years: dict[int, np.ndarray],
    Cld_years: dict[int, np.ndarray],
    IWP_years: dict[int, np.ndarray],
) -> None:
    """
    Extracts data for principal component 1 (PC1) and cloud data (CERES) for each year from 2010 to 2020,
    and assigns these data to global variables with names that include the year.

    Parameters:
    -----------
    PC_years: dict[int, np.ndarray]
        A dictionary containing the principal component 1 data for each year from 2010 to 2020. The keys
        are integers representing years, and the values are numpy arrays containing the PC1 data.
    Cld_years: dict[int, np.ndarray]
        A dictionary containing the CERES cloud data for each year from 2010 to 2020. The keys are integers
        representing years, and the values are numpy arrays containing the CERES data.

    Returns:
    --------
    None
    """
    for year in range(2017, 2021):
        globals()[f"PC_{year}"] = PC_years[year]

    for year in range(2017, 2021):
        globals()[f"Cld_{year}"] = Cld_years[year]

    for year in range(2017, 2021):
        globals()[f"IWP_{year}"] = IWP_years[year]


extract_PC1_CERES_each_year(PC_years, Cld_years, IWP_years)

#########################################
##### start seperate time test ##########
#########################################

# ------ Segmentation of cloud data within each PC interval ---------------------------------
#### triout for IWP constrain the same time with PC1 gap constrain ####
# first we need to divide IWP data into n intervals
# what i think is 6 parts of IWP, 0-1, 1-2, 2-3, 3-4, 4-5, 5-6
dividePC = DividePCByDataVolume(
    dataarray_main=IWP_data,
    n=5,
    start=-2.5,
    end=5.5,
    gap=0.05,
)
IWP_gap = dividePC.main_gap()

# region
filter_data_fit_PC1_gap_IWP_constrain = (
    Filter_data_fit_PC1_gap_plot_IWP_constrain(
        Cld_data=Cld_2018.reshape(-1, 180, 360),
        start=-2.5,
        end=5.5,
        gap=0.05,
    )
)

Cld_2020_match_PC_gap_IWP_constrain_mean = np.empty(
    (5, 160, 180, 360)
)
PC_2020_match_PC_gap_IWP_constrain_mean = np.empty(
    (5, 160, 180, 360)
)

for i in range(0, len(IWP_gap) - 1):
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    (
        Cld_2020_match_PC_gap_IWP_constrain_mean[i],
        PC_2020_match_PC_gap_IWP_constrain_mean[i],
    ) = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_new(
        Cld_data=Cld_all.reshape(-1, 180, 360),
        PC_data=PC_all.reshape(-1, 180, 360),
        IWP_data=IWP_data.reshape(-1, 180, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

# endregion

#######################################################################
## Use box plot to quantify the cld distribution within each PC interval
## Under the IWP constrain ####
#######################################################################

# -----------------------------------------------------------
# plot box plot for every variable
# -----------------------------------------------------------
for i in range(0, len(IWP_gap) - 1):
    filter_data_fit_PC1_gap_IWP_constrain.plot_box_plot(
        Cld_match_PC_gap=Cld_2020_match_PC_gap_IWP_constrain_mean[
            i
        ],
        savefig_str=f"box_plot_IWP_constrain_{IWP_gap[i]}_{IWP_gap[i+1]}",
    )


# -----------------------------------------------------------
# plot fill between plot for every variable
# -----------------------------------------------------------
# plot fill between plot for every variable
# fill between area is 2 times std
def error_fill_plot(data, xlabel, savefig_str):
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
    fig, ax = plt.subplots(figsize=(7, 5))

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
    plt.ylabel(xlabel)

    # save figure
    os.makedirs(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/",
        exist_ok=True,
    )
    plt.savefig(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/"
        + savefig_str,
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()


# fill between plot for every variable
for i in range(0, len(IWP_gap) - 1):
    error_fill_plot(
        data=Cld_2020_match_PC_gap_IWP_constrain_mean[i],
        xlabel="Cld Area (%)",
        savefig_str="Cld_area_IWP_constrain"
        + f"{np.round(IWP_gap[i],2)}_{np.round(IWP_gap[i+1],2)}"
        + ".png",
    )


# ------------------------------------------------------------
# Plot 3D error fill plot for different IWP conditions
# ------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D


def error_fill_3d_plot(
    data_list, xlabel, ylabel, zlabel, legend_labels, savefig_str
):
    """
    Create a 3D error fill plot with different colors for each IWP condition

    Args:
        data_list (list): List of data arrays for different IWP conditions
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        legend_labels (list): List of legend labels for each IWP condition
        savefig_str (str): String for saving the figure
    """
    # Create a figure instance
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Loop through data_list to create error fill plots for each IWP condition
    for i, data in enumerate(data_list):
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

        # Create IWP condition coordinate on y-axis
        iwp_condition = np.ones_like(data_x) * i

        # Plot the mean line and fill between up and down limits for each IWP condition
        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=colors[i % len(colors)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=colors[i % len(colors)],
                alpha=0.2,
            ),
            zs=i,
            zdir="y",
        )

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # Add legend
    custom_lines = [
        plt.Line2D([0], [0], color=colors[i % len(colors)], lw=3)
        for i in range(len(data_list))
    ]
    ax.legend(custom_lines, legend_labels)

    # Set the viewing angle
    ax.view_init(elev=10, azim=-65)
    ax.dist = 12

    # Turn off the grid lines
    ax.grid(False)

    # Save figure
    os.makedirs(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/",
        exist_ok=True,
    )
    plt.savefig(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/"
        + savefig_str,
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()


# Call the error_fill_3d_plot function

# Create a list of data arrays for different IWP conditions
cld_data_list = [
    Cld_2020_match_PC_gap_IWP_constrain_mean[0],
    Cld_2020_match_PC_gap_IWP_constrain_mean[1],
    Cld_2020_match_PC_gap_IWP_constrain_mean[2],
    Cld_2020_match_PC_gap_IWP_constrain_mean[3],
    Cld_2020_match_PC_gap_IWP_constrain_mean[4],
    # Add more data arrays as needed
]

# Define the labels for each IWP condition
iwp_legend_labels = [
    "IWP Condition 1",
    "IWP Condition 2",
    "IWP Condition 3",
    "IWP Condition 4",
    "IWP Condition 5"
    # Add more labels as needed
]

# Call the error_fill_3d_plot function with your data
error_fill_3d_plot(
    data_list=cld_data_list,
    xlabel="PC1",
    ylabel="IWP Conditions",
    zlabel="Cld Area (%)",
    legend_labels=iwp_legend_labels,
    savefig_str="Cld_area_IWP_constrain_3D",
)

# ------------------------------------------------------------------
# Plot 4D plot to demenstrate the constrain of PC1, IWP, Dust AOD
# ------------------------------------------------------------------


def plot_4d_IWP_PC1_AOD_constrain_HCF(
    data_list, xlabel, ylabel, zlabel, legend_labels, savefig_str
):
    """
    Create a 4D plot

    Args:
        data_list (list): List of data arrays for different IWP conditions
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        legend_labels (list): List of legend labels for each IWP condition
        savefig_str (str): String for saving the figure
    """
    # Create a figure instance
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Loop through data_list to create error fill plots for each IWP condition
    for i, data in enumerate(data_list):
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

        # Create IWP condition coordinate on y-axis
        iwp_condition = np.ones_like(data_x) * i

        # Plot the mean line and fill between up and down limits for each IWP condition
        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=colors[i % len(colors)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=colors[i % len(colors)],
                alpha=0.2,
            ),
            zs=i,
            zdir="y",
        )

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # Add legend
    custom_lines = [
        plt.Line2D([0], [0], color=colors[i % len(colors)], lw=3)
        for i in range(len(data_list))
    ]
    ax.legend(custom_lines, legend_labels)

    # Set the viewing angle
    ax.view_init(elev=10, azim=-65)
    ax.dist = 12

    # Turn off the grid lines
    ax.grid(False)

    # Save figure
    os.makedirs(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/",
        exist_ok=True,
    )
    plt.savefig(
        "/RAID01/data/python_fig/fill_between_plot_cld_var/"
        + savefig_str,
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()


#######################################################################
################# Filter data within PC interval ######################
####### Using 10% lowest and highest PC values in each interval #######
#######################################################################

#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################

# region
(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_1_2_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_1_2_month_median
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_1_2_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_1_2_month_median
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_1_2_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_1_2_month_mean
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_1_2_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_1_2_month_mean
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_3_4_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_3_4_month_median
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_3_4_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_3_4_month_median
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_3_4_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_3_4_month_mean
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_3_4_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_3_4_month_mean
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_5_6_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_5_6_month_median
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_5_6_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_5_6_month_median
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_5_6_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_5_6_month_mean
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_5_6_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_5_6_month_mean
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_7_8_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_7_8_month_median
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_7_8_month_median,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_7_8_month_median
)

(
    Cld_lowermost_error_2017_2019,
    Cld_highermost_error_2017_2019,
    Cld_2017_2019_match_PC_gap_filtered_7_8_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2017_2019_match_PC_gap_7_8_month_mean
)
(
    Cld_lowermost_error_2020,
    Cld_highermost_error_2020,
    Cld_2020_match_PC_gap_filtered_7_8_month_mean,
) = filter_data_PC1_gap_lowermost_highermost_error(
    Cld_2020_match_PC_gap_7_8_month_mean
)
# endregion

#######################################################################
#######################################################################
#######################################################################
###### We only analyze the april to july cld and pc1 data #############
###### In order to extract the contrail maximum signal ################
#######################################################################

# region
# -----------------------------------------------------------------------------
# divide the PC1 data into 6 parts

concatenated_data = np.concatenate(
    (
        PC_2020_1_2_month[:, 110:140, :],
        PC_2020_3_4_month[:, 110:140, :],
        PC_2020_5_6_month[:, 110:140, :],
        PC_2020_7_8_month[:, 110:140, :],
    ),
    axis=0,
)

dividePC = DividePCByDataVolume(
    dataarray_main=concatenated_data,
    n=5,
    start=-2.5,
    end=5.5,
    gap=0.05,
)
gap_num_min, gap_num_max = dividePC.gap_number_for_giving_main_gap()

# retrieve the sparse PC_constrain data
# dimension 1 means atmospheric condiations
# 0 means bad, 1 means moderate, 2 means good
Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean = np.empty(
    (3, 360)
)
Cld_delta_2020_match_PC_gap_filtered_3_4_month_mean = np.empty(
    (3, 360)
)
Cld_delta_2020_match_PC_gap_filtered_5_6_month_mean = np.empty(
    (3, 360)
)
Cld_delta_2020_match_PC_gap_filtered_7_8_month_mean = np.empty(
    (3, 360)
)
# 1-2 month
for i in range(0, 3):
    Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean[
        i
    ] = np.nanmean(
        Cld_2020_match_PC_gap_filtered_1_2_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    ) - np.nanmean(
        Cld_2017_2019_match_PC_gap_filtered_1_2_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    )
    # 3-4 month
    Cld_delta_2020_match_PC_gap_filtered_3_4_month_mean[
        i
    ] = np.nanmean(
        Cld_2020_match_PC_gap_filtered_3_4_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    ) - np.nanmean(
        Cld_2017_2019_match_PC_gap_filtered_3_4_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    )
    # 5-6 month
    Cld_delta_2020_match_PC_gap_filtered_5_6_month_mean[
        i
    ] = np.nanmean(
        Cld_2020_match_PC_gap_filtered_5_6_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    ) - np.nanmean(
        Cld_2017_2019_match_PC_gap_filtered_5_6_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    )
    # 7-8 month
    Cld_delta_2020_match_PC_gap_filtered_7_8_month_mean[
        i
    ] = np.nanmean(
        Cld_2020_match_PC_gap_filtered_7_8_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    ) - np.nanmean(
        Cld_2017_2019_match_PC_gap_filtered_7_8_month_mean[
            int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
        ],
        axis=(0, 1),
    )


# Improved version
# define lists for atmospheric conditions and calculation type
atm_conditions = ["bad", "moderate", "good"]
calc_types = ["median", "mean"]

# create empty dictionaries to store the results
results_median = {}
results_mean = {}

# -----------------------------------------------------------------------------
# 1 - 2 month

# iterate over atmospheric conditions and calculation types
for i, atm in enumerate(atm_conditions):
    for calc_type in calc_types:
        # create the variable names dynamically
        var_name = f"{atm}_filtered_{calc_type}"
        cld_2020_var = (
            f"Cld_2020_match_PC_gap_filtered_1_2_month_{calc_type}"
        )
        cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_1_2_month_{calc_type}"
        gap_min_var = gap_num_min[i]
        gap_max_var = gap_num_max[i]

        # call the compare_cld_between_2020_others function
        results = compare_cld_between_2020_others(
            Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
            Cld_all_match_PC_gap_others=locals()[cld_others_var],
            start=int(gap_min_var),
            end=int(gap_max_var),
        )

        # store the results in the appropriate dictionary
        if calc_type == "median":
            results_median[var_name] = results
        elif calc_type == "mean":
            results_mean[var_name] = results

# assign the results to variables
Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median = (
    results_median["bad_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median = (
    results_median["moderate_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median = (
    results_median["good_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean = (
    results_mean["bad_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean = (
    results_mean["moderate_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean = (
    results_mean["good_filtered_mean"]
)

# ---------------------------------------------------------------------------------
# 3 - 4 month
# iterate over atmospheric conditions and calculation types
for i, atm in enumerate(atm_conditions):
    for calc_type in calc_types:
        # create the variable names dynamically
        var_name = f"{atm}_filtered_{calc_type}"
        cld_2020_var = (
            f"Cld_2020_match_PC_gap_filtered_3_4_month_{calc_type}"
        )
        cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_3_4_month_{calc_type}"
        gap_min_var = gap_num_min[i]
        gap_max_var = gap_num_max[i]

        # call the compare_cld_between_2020_others function
        results = compare_cld_between_2020_others(
            Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
            Cld_all_match_PC_gap_others=locals()[cld_others_var],
            start=int(gap_min_var),
            end=int(gap_max_var),
        )

        # store the results in the appropriate dictionary
        if calc_type == "median":
            results_median[var_name] = results
        elif calc_type == "mean":
            results_mean[var_name] = results

# assign the results to variables
Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median = (
    results_median["bad_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median = (
    results_median["moderate_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median = (
    results_median["good_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean = (
    results_mean["bad_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean = (
    results_mean["moderate_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean = (
    results_mean["good_filtered_mean"]
)

# --------------------------------------------------------------------------------------------
# 5 - 6 month
# iterate over atmospheric conditions and calculation types
for i, atm in enumerate(atm_conditions):
    for calc_type in calc_types:
        # create the variable names dynamically
        var_name = f"{atm}_filtered_{calc_type}"
        cld_2020_var = (
            f"Cld_2020_match_PC_gap_filtered_5_6_month_{calc_type}"
        )
        cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_5_6_month_{calc_type}"
        gap_min_var = gap_num_min[i]
        gap_max_var = gap_num_max[i]

        # call the compare_cld_between_2020_others function
        results = compare_cld_between_2020_others(
            Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
            Cld_all_match_PC_gap_others=locals()[cld_others_var],
            start=int(gap_min_var),
            end=int(gap_max_var),
        )

        # store the results in the appropriate dictionary
        if calc_type == "median":
            results_median[var_name] = results
        elif calc_type == "mean":
            results_mean[var_name] = results

Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median = (
    results_median["bad_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median = (
    results_median["moderate_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median = (
    results_median["good_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean = (
    results_mean["bad_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean = (
    results_mean["moderate_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean = (
    results_mean["good_filtered_mean"]
)

# --------------------------------------------------------------------------------------------
# 7 - 8 month
# iterate over atmospheric conditions and calculation types
for i, atm in enumerate(atm_conditions):
    for calc_type in calc_types:
        # create the variable names dynamically
        var_name = f"{atm}_filtered_{calc_type}"
        cld_2020_var = (
            f"Cld_2020_match_PC_gap_filtered_7_8_month_{calc_type}"
        )
        cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_7_8_month_{calc_type}"
        gap_min_var = gap_num_min[i]
        gap_max_var = gap_num_max[i]

        # call the compare_cld_between_2020_others function
        results = compare_cld_between_2020_others(
            Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
            Cld_all_match_PC_gap_others=locals()[cld_others_var],
            start=int(gap_min_var),
            end=int(gap_max_var),
        )

        # store the results in the appropriate dictionary
        if calc_type == "median":
            results_median[var_name] = results
        elif calc_type == "mean":
            results_mean[var_name] = results

Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median = (
    results_median["bad_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median = (
    results_median["moderate_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median = (
    results_median["good_filtered_median"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean = (
    results_mean["bad_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean = (
    results_mean["moderate_filtered_mean"]
)
Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean = (
    results_mean["good_filtered_mean"]
)
# --------------------------------------------------------------------------------------------
# endregion

########################################################################
###### Draw the Cld difference between 2020 and 2010-2019 ###############
###### But by latitude this time, 20N-60N, Contrail region ##############
########################################################################

#########################################################
############ smoothed data ##############################
#########################################################
# ------------------------------------------------------------
# plot the aviation fill between version
# ------------------------------------------------------------
y_lim_lst = [[-4, 6.5], [-4.5, 6.5], [-5.2, 8]]

# 1-2 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
    - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="January-February",
    step=5,
)

# 3-4 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
    - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="March-April",
    step=5,
)

# 5-6 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
    - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="May-June",
    step=5,
)

# 7-8 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
    - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="July-August",
    step=5,
)

# no compare between PC1 and no PC1 constrain
# ------------------------------------------------------------
# plot the aviation fill between version
# ------------------------------------------------------------
y_lim_lst = [[-1.15, 1.15], [-1.15, 1.15], [-1.15, 1.15]]

# 1-2 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
    - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="January-February",
    step=10,
)

# 3-4 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
    - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="March-April",
    step=10,
)

# 5-6 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
    - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="May-June",
    step=5,
)

# 7-8 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
    - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="July-August",
    step=10,
)


# zscores
# ------------------------------------------------------------
# plot the aviation fill between version
# ------------------------------------------------------------
y_lim_lst = [[-1.15, 1.15], [-1.15, 1.15], [-1.15, 1.15]]

# 1-2 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
    - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="January-February",
    step=10,
)

# 3-4 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
    - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="March-April",
    step=10,
)

# 5-6 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
    - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="May-June",
    step=5,
)

# 7-8 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
    Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
    - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="July-August",
    step=10,
)

# --------------------------------------------
# plot the improve version
# --------------------------------------------
y_lim_lst = [[-4, 6.5], [-4.5, 6.5], [-5.2, 8]]

# 1-2 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
    Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="January-February",
    step=5,
)

# 3-4 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
    Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="March-April",
    step=5,
)

# 5-6 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
    Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="May-June",
    step=5,
)

# 7-8 month
compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
    Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
    Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
    Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
    Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
    Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
    Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
    Cld_data_name=r"$\Delta$" + "HCF(%)",
    y_lim_lst=y_lim_lst,
    title="July-August",
    step=5,
)
