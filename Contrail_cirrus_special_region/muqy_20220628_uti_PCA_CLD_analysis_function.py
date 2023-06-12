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

    Useful functions for PCA-HCF analyze, including the 
    calculation of CV
    
    Owner: Mu Qingyu
    version 1.0
        version 2.0 : Filter_data_fit_PC1_gap_plot now must pass a Cld_Data object
            to automatically get the lat and lon information
    Created: 2022-06-28
    
    Including the following parts:
        
        1) Read the Pre-calculated PC1 and HCF data 
        
        2) Filter the data to fit the PC1 gap like -1.5 ~ 3.5
        
        3) Plot boxplot to show the distribution of HCF data
        
        4) Compare HCF of different years in the same PC1 condition
                
        5) All sort of test code
        
"""

import glob

# ------------ PCA analysis ------------
# ------------ Start import ------------
import os
import warnings
from statistics import mean

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np

# from numba import jit
import pandas as pd
import seaborn as sns
import shapely.geometry as sgeom
import xarray as xr
from matplotlib import rcParams, ticker
from matplotlib.dates import DateFormatter, MonthLocator
from metpy.units import units

# ----------  importing dcmap from my util ----------#
from muqy_20220413_util_useful_functions import dcmap as dcmap
from scipy import stats
from scipy.stats import norm, zscore

# Use the font and apply the matplotlib style
mpl.rc("font", family="Times New Roman")
mpl.style.use("seaborn-v0_8-ticks")
# Reuse the same font to ensure that font set properly
# plt.rc("font", family="Times New Roman")

# ----------  done importing  ----------#


# define the function to read the data
def read_nc_data(file_path):
    """
    用于读取nc文件的函数
    :param file_path: nc文件的路径
    :return: 读取的数据
    """
    data = xr.open_dataset(file_path)
    return data


# def nan_array_normalize(arr):
#     """
#     Normalize the array, and fill the nan with 0
#     :param arr: the array to be normalized
#     :return: normalized array
#     """
#     from sklearn.preprocessing import MinMaxScaler

#     # Get the original shape of the data
#     original_shape = arr.shape

#     arr_no_nan = np.nan_to_num(arr, nan=0)

#     scaler = MinMaxScaler(feature_range=(-1, 1))

#     scaled_data = scaler.fit_transform(arr_no_nan.reshape(-1, 1)).reshape(original_shape)

#     return scaled_data


def nan_array_normalize(arr):
    """
    Normalize the array, and fill the nan with 0
    :param arr: the array to be normalized
    :return: normalized array
    """

    arr_no_nan = np.nan_to_num(arr, nan=0)

    scaled_data = stats.zscore(arr_no_nan)

    return scaled_data


# ---------- Read PCA&CLD data from netcdf file --------


def read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name):
    """
    Read the PC1 and CERES data from the netcdf file
    you can choose the PC1 and CERES data you want to read
    PC1 data :
    [0] : 4-para PC1
    [1] : 5-para PC1

    Cld data :
    ["Cldarea",
        "Cldicerad",
        "Cldtau",
        "Cldtau_lin",
        "IWP",
        "Cldemissir"]
    using dataset_num

    Args:
        PC_para_num (int): the number of PC1 parameter
        0 : 4para PC1
        1 : 5para PC1

        CERES_Cld_dataset_num (int): 0-5, specify the CERES dataset
        0 - Cldarea
        1 - Cldicerad
        2 - Cldtau
        3 - Cldtau_lin
        4 - IWP
        5 - Cldemissir

    Returns:
        Specified CERES dataset and PC1 data
    """
    # Read data from netcdf file
    print("Reading data from netcdf file...")
    data_cld = xr.open_dataset(
        "/RAID01/data/Cld_data/2010_2020_CERES_high_cloud_data.nc"
    )
    data_pc = xr.open_dataset(
        "/RAID01/data/PC_data/1990_2020_4_parameters_300hPa_PC1.nc"
    )

    PC_all = np.array(data_pc.PC1)

    # Arrange data from all years
    PC_all = PC_all.reshape(31, 12, 28, 180, 360)

    # -------------------------------------------------
    Cld_data = data_cld[CERES_Cld_dataset_name].values
    
    IWP_data = data_cld["IWP"].values

    print("Done loading netcdf file.")

    Cld_all = Cld_data.reshape(11, 12, 28, 180, 360)
    IWP_data = IWP_data.reshape(11, 12, 28, 180, 360)

    Cld_all[Cld_all == -999] = np.nan
    IWP_data[IWP_data == -999] = np.nan

    # -------------------------------------------------
    Cld_years = {}
    IWP_years = {}   
    PC_years = {}
    
    for i, year in enumerate(range(2010, 2021)):
        PC_years[year] = PC_all[-(11 - i)].astype(np.float32)

    for i, year in enumerate(range(2010, 2021)):
        Cld_years[year] = Cld_all[i].astype(np.float32)
        
    for i, year in enumerate(range(2010, 2021)):
        IWP_years[year] = IWP_data[i].astype(np.float32)

    return (
        # pc
        PC_all,
        PC_years,
        # cld
        Cld_all,
        Cld_years,
        # iwp
        IWP_data,
        IWP_years,
    )


def read_PC1_CERES_20_50_lat_band_from_netcdf(
    CERES_Cld_dataset_name,
):
    """
    Read the PC1 and CERES data from the netcdf file
    you can choose the PC1 and CERES data you want to read
    PC1 data :
    [0] : 4-para PC1
    [1] : 5-para PC1

    Cld data :
    ["Cldarea",
        "Cldicerad",
        "Cldtau",
        "Cldtau_lin",
        "IWP",
        "Cldemissir"]
    using dataset_num

    Args:
        PC_para_num (int): the number of PC1 parameter
        0 : 4para PC1
        1 : 5para PC1

        CERES_Cld_dataset_num (int): 0-5, specify the CERES dataset
        0 - Cldarea
        1 - Cldicerad
        2 - Cldtau
        3 - Cldtau_lin
        4 - IWP
        5 - Cldemissir

    Returns:
        Specified CERES dataset and PC1 data
    """
    # Read data from netcdf file
    print("Reading data from netcdf file...")
    data_cld = xr.open_dataset(
        "/RAID01/data/Cld_data/2010_2020_CERES_high_cloud_data.nc"
    )
    data_pc = xr.open_dataset(
        "/RAID01/data/PC_data/1990_2020_4_parameters_300hPa_PC1.nc"
    )

    print("Done loading netcdf file.")

    PC_all = np.array(data_pc.PC1)

    # Arrange data from all years
    PC_all = PC_all.reshape(31, 12, 28, 180, 360)[
        :, :, :, 110:140, :
    ]

    PC_years = {}
    for i, year in enumerate(range(2017, 2021)):
        PC_years[year] = PC_all[-(11 - i)].astype(np.float32)

    # -------------------------------------------------
    Cld_data = np.array(
        data_cld[CERES_Cld_dataset_name]
    )[:,:,110:140,:]
    IWP_data = np.array(data_cld["IWP"])[:,:,110:140,:]

    Cld_all = Cld_data.reshape(11, 12, 28, 30, 360)
    IWP_data = IWP_data.reshape(11, 12, 28, 30, 360)

    Cld_all[Cld_all == -999] = np.nan
    IWP_data[IWP_data == -999] = np.nan

    Cld_years = {}
    IWP_years = {}
    for i, year in enumerate(range(2017, 2021)):
        Cld_years[year] = Cld_all[i].astype(np.float32)
        IWP_years[year] = IWP_data[i].astype(np.float32)

    return (
        # pc
        PC_all.astype(np.float32),
        PC_years,
        # cld
        Cld_all.astype(np.float32),
        Cld_years,
        # iwp
        IWP_data.astype(np.float32),
        IWP_years,
    )


def read_PC1_CERES_20_60_lat_band_from_netcdf(
    CERES_Cld_dataset_name,
):
    """
    Read the PC1 and CERES data from the netcdf file
    you can choose the PC1 and CERES data you want to read
    PC1 data :
    [0] : 4-para PC1
    [1] : 5-para PC1

    Cld data :
    ["Cldarea",
        "Cldicerad",
        "Cldtau",
        "Cldtau_lin",
        "IWP",
        "Cldemissir"]
    using dataset_num

    Args:
        PC_para_num (int): the number of PC1 parameter
        0 : 4para PC1
        1 : 5para PC1

        CERES_Cld_dataset_num (int): 0-5, specify the CERES dataset
        0 - Cldarea
        1 - Cldicerad
        2 - Cldtau
        3 - Cldtau_lin
        4 - IWP
        5 - Cldemissir

    Returns:
        Specified CERES dataset and PC1 data
    """
    # Read data from netcdf file
    print("Reading data from netcdf file...")
    data_cld = xr.open_dataset(
        "/RAID01/data/Cld_data/2010_2020_CERES_high_cloud_data.nc"
    )
    data_pc = xr.open_dataset(
        "/RAID01/data/PC_data/1990_2020_4_parameters_300hPa_PC1.nc"
    )

    print("Done loading netcdf file.")

    PC_all = np.array(data_pc.PC1)

    # Arrange data from all years
    PC_all = PC_all.reshape(31, 12, 28, 180, 360)[
        :, :, :, 110:150, :
    ]

    PC_years = {}
    for i, year in enumerate(range(2016, 2021)):
        PC_years[year] = PC_all[-(11 - i)].astype(np.float32)

    # -------------------------------------------------
    Cld_data = np.array(
        data_cld[CERES_Cld_dataset_name]
    )[:,:,110:150,:]
    IWP_data = np.array(data_cld["IWP"])[:,:,110:150,:]

    Cld_all = Cld_data.reshape(11, 12, 28, 40, 360)
    IWP_data = IWP_data.reshape(11, 12, 28, 40, 360)

    Cld_all[Cld_all == -999] = np.nan
    IWP_data[IWP_data == -999] = np.nan

    Cld_years = {}
    IWP_years = {}
    for i, year in enumerate(range(2016, 2021)):
        Cld_years[year] = Cld_all[i].astype(np.float32)
        IWP_years[year] = IWP_data[i].astype(np.float32)

    return (
        # pc
        PC_all.astype(np.float32),
        PC_years,
        # cld
        Cld_all.astype(np.float32),
        Cld_years,
        # iwp
        IWP_data.astype(np.float32),
        IWP_years,
    )


def read_PC1_CERES_20_60_lat_band_from_netcdf_2015_2020(
    CERES_Cld_dataset_name,
):
    """
    Read the PC1 and CERES data from the netcdf file
    you can choose the PC1 and CERES data you want to read
    PC1 data :
    [0] : 4-para PC1
    [1] : 5-para PC1

    Cld data :
    ["Cldarea",
        "Cldicerad",
        "Cldtau",
        "Cldtau_lin",
        "IWP",
        "Cldemissir"]
    using dataset_num

    Args:
        PC_para_num (int): the number of PC1 parameter
        0 : 4para PC1
        1 : 5para PC1

        CERES_Cld_dataset_num (int): 0-5, specify the CERES dataset
        0 - Cldarea
        1 - Cldicerad
        2 - Cldtau
        3 - Cldtau_lin
        4 - IWP
        5 - Cldemissir

    Returns:
        Specified CERES dataset and PC1 data
    """
    # Read data from netcdf file
    print("Reading data from netcdf file...")
    data_cld = xr.open_dataset(
        "/RAID01/data/Cld_data/2010_2020_CERES_high_cloud_data.nc"
    )
    data_pc = xr.open_dataset(
        "/RAID01/data/PC_data/1990_2020_4_parameters_300hPa_PC1.nc"
    )

    print("Done loading netcdf file.")

    PC_all = np.array(data_pc.PC1)

    # Arrange data from all years
    PC_all = PC_all.reshape(31, 12, 28, 180, 360)[
        :, :, :, 110:150, :
    ]

    PC_years = {}
    for i, year in enumerate(range(2015, 2021)):
        PC_years[year] = PC_all[-(11 - i)].astype(np.float32)

    # -------------------------------------------------
    Cld_data = np.array(
        data_cld[CERES_Cld_dataset_name]
    )[:,:,110:150,:]
    IWP_data = np.array(data_cld["IWP"])[:,:,110:150,:]

    Cld_all = Cld_data.reshape(11, 12, 28, 40, 360)
    IWP_data = IWP_data.reshape(11, 12, 28, 40, 360)

    Cld_all[Cld_all == -999] = np.nan
    IWP_data[IWP_data == -999] = np.nan

    Cld_years = {}
    IWP_years = {}
    for i, year in enumerate(range(2015, 2021)):
        Cld_years[year] = Cld_all[i].astype(np.float32)
        IWP_years[year] = IWP_data[i].astype(np.float32)

    return (
        # pc
        PC_all.astype(np.float32),
        PC_years,
        # cld
        Cld_all.astype(np.float32),
        Cld_years,
        # iwp
        IWP_data.astype(np.float32),
        IWP_years,
    )


#########################################################
######### moving average function #######################
#########################################################


def np_move_avg(a, n, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode)


#########################################################
###### simple plot func ##################################
#########################################################

# ------------------------------------------------------------------------
# Plot actual aviation #
# ------------------------------------------------------------------------


def plot_statistic_var(var_data, var_name):
    # This code plots the 3 year mean of the PC1 for the 3 years of the dataset
    # The PC1 is the first principal component of the dataset
    # This is used to see the evolution of the principal component over the 3 years
    # The function is called in the main function
    var_2023 = var_data["2023 7-day moving average"].values
    var_2022 = var_data["2022 7-day moving average"].values
    var_2021 = var_data["2021 7-day moving average"].values
    var_2020 = var_data["2020 7-day moving average"].values
    var_2019 = var_data["2019 7-day moving average"].values

    # Convert day of year to datetime
    dates = pd.date_range(
        start="1/1/2023", end="12/31/2023", freq="D"
    )
    dates = dates + pd.Timedelta("6H")
    dates = dates.strftime("%Y-%m-%d %H:%M:%S")

    fig = plt.figure(figsize=(12, 3.5), constrained_layout=True)
    ax1 = fig.add_subplot(111)
    # ax1.plot(
    #     np.arange(1, 367, 1),
    #     var_2023,
    #     label="2023",
    #     color=(123 / 255, 152 / 255, 201 / 255),
    #     linewidth=2,
    # )
    # ax1.plot(
    #     np.arange(1, 367, 1),
    #     var_2022,
    #     label="2022",
    #     color=(162 / 255, 141 / 255, 190 / 255),
    #     linewidth=2,
    # )
    # ax1.plot(
    #     np.arange(1, 367, 1),
    #     var_2021,
    #     label="2021",
    #     color=(136 / 255, 187 / 255, 203 / 255),
    #     linewidth=2,
    # )
    ax1.plot(
        np.arange(1, 367, 1),
        var_2020,
        label="2020",
        color=(230 / 255, 165 / 255, 85 / 255),
        linewidth=2,
    )
    ax1.plot(
        np.arange(1, 367, 1),
        var_2019,
        label="2019",
        color=(152 / 255, 152 / 255, 152 / 255),
        linewidth=2,
    )

    # set the x-axis ticks as month names
    date_format = DateFormatter("%b")
    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(date_format)

    ax1.set_ylabel(var_name, fontsize=14)
    ax1.set_xlabel("Month of the year", fontsize=14)
    ax1.set_ylim(
        0.65 * np.nanmin(var_2020), 1.05 * np.nanmax(var_2022)
    )

    # set the x-axis ticks font size
    ax1.tick_params(axis="x", labelsize=14)

    plt.legend()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(
        "/RAID01/data/python_fig/fig1.png",
        dpi=300,
        facecolor="w",
        edgecolor="w",
    )

    plt.show()


def plot_statistic_var_difference(var_data, var_name):
    # This code plots the 3 year mean of the PC1 for the 3 years of the dataset
    # The PC1 is the first principal component of the dataset
    # This is used to see the evolution of the principal component over the 3 years
    # The function is called in the main function
    var_2023 = var_data["2023 7-day moving average"].values
    var_2022 = var_data["2022 7-day moving average"].values
    var_2021 = var_data["2021 7-day moving average"].values
    var_2020 = var_data["2020 7-day moving average"].values
    var_2019 = var_data["2019 7-day moving average"].values

    # Convert day of year to datetime
    dates = pd.date_range(
        start="1/1/2023", end="12/31/2023", freq="D"
    )
    dates = dates + pd.Timedelta("6H")
    dates = dates.strftime("%Y-%m-%d %H:%M:%S")

    fig = plt.figure(figsize=(12, 3.5), constrained_layout=True)
    ax1 = fig.add_subplot(111)

    ax1.plot(
        np.arange(1, 367, 1),
        var_2019 - var_2020,
        label="2019 - 2020 aviation number",
        color=(152 / 255, 152 / 255, 152 / 255),
        linewidth=2,
    )

    # set the x-axis ticks as month names
    date_format = DateFormatter("%b")
    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(date_format)

    ax1.set_ylabel(var_name, fontsize=14)
    ax1.set_xlabel("Month of the year", fontsize=14)
    # ax1.set_ylim(
    #     0.65 * np.nanmin(var_2020), 1.05 * np.nanmax(var_2022)
    # )

    # set the x-axis ticks font size
    ax1.tick_params(axis="x", labelsize=14)

    plt.legend()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(
        "/RAID01/data/python_fig/2019_2020_aviation_difference.png",
        dpi=300,
        facecolor="w",
        edgecolor="w",
    )

    plt.show()


def plot_statistic_var_difference_fill_time(var_data, var_name):
    # This code plots the 3 year mean of the PC1 for the 3 years of the dataset
    # The PC1 is the first principal component of the dataset
    # This is used to see the evolution of the principal component over the 3 years
    # The function is called in the main function
    var_2023 = var_data["2023 7-day moving average"].values
    var_2022 = var_data["2022 7-day moving average"].values
    var_2021 = var_data["2021 7-day moving average"].values
    var_2020 = var_data["2020 7-day moving average"].values
    var_2019 = var_data["2019 7-day moving average"].values

    # Define the x-axis limits for the different regions
    min_val = np.nanmin(var_2019 - var_2020)

    # Define the x-axis limits for the different regions
    mar_limits = (61, 90)  # March 1 to March 31
    apr_limits = (91, 120)  # April 1 to April 30
    may_limits = (121, 151)  # May 1 to May 31
    jun_limits = (152, 181)  # June 1 to June 30
    jul_limits = (182, 212)  # July 1 to July 31
    aug_limits = (213, 243)  # August 1 to August 31

    # Convert day of year to datetime
    dates = pd.date_range(
        start="1/1/2023", end="12/31/2023", freq="D"
    )
    dates = dates + pd.Timedelta("6H")
    dates = dates.strftime("%Y-%m-%d %H:%M:%S")

    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    ax1 = fig.add_subplot(111)

    ax1.plot(
        np.arange(1, 367, 1),
        var_2019 - var_2020,
        label="2019 - 2020 aviation number",
        color=(152 / 255, 152 / 255, 152 / 255),
        linewidth=2,
    )

    # set the x-axis ticks as month names
    date_format = DateFormatter("%b")
    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(date_format)

    ax1.set_ylabel(var_name, fontsize=14)
    ax1.set_xlabel("Month of the year", fontsize=14)

    # set the x-axis ticks font size
    ax1.tick_params(axis="x", labelsize=14)

    # Fill between March and April with color1
    ax1.fill_between(
        np.arange(mar_limits[0], apr_limits[1] + 1),
        var_2019[mar_limits[0] - 1 : apr_limits[1]]
        - var_2020[mar_limits[0] - 1 : apr_limits[1]],
        min_val,
        color="#243D7A",
        alpha=0.4,
    )

    # Fill between May and June with color2
    ax1.fill_between(
        np.arange(may_limits[0], jun_limits[1] + 1),
        var_2019[may_limits[0] - 1 : jun_limits[1]]
        - var_2020[may_limits[0] - 1 : jun_limits[1]],
        min_val,
        color="#88ADD5",
        alpha=0.4,
    )

    # Fill between July and August with color3
    ax1.fill_between(
        np.arange(jul_limits[0], aug_limits[1] + 1),
        var_2019[jul_limits[0] - 1 : aug_limits[1]]
        - var_2020[jul_limits[0] - 1 : aug_limits[1]],
        min_val,
        color="#CFD5E2",
        alpha=0.4,
    )

    plt.legend()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(
        "/RAID01/data/python_fig/2019_2020_aviation_difference_fill_time.png",
        dpi=300,
        facecolor="w",
        edgecolor="w",
    )

    plt.show()


def plot_lat_lon_mean_of_flight(var_data, var_name):
    # transform the data to a numpy array
    data = var_data

    # Calculate the mean along the latitude and longitude dimensions
    # You can mean by time as well
    lat_mean = np.mean(data, axis=(1, 2))
    lon_mean = np.mean(data, axis=(0, 2))

    # Create a new figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the latitude mean
    ax1.plot(lat_mean)
    ax1.set_xlabel("Latitude Mean")
    ax1.set_ylabel("Data Mean")
    ax1.set_title("Latitude Mean of Data")
    ax1.set_xticks(np.arange(-90, 91, 30))
    ax1.set_xticklabels(
        [
            "90$^\circ$S",
            "60$^\circ$S",
            "30$^\circ$S",
            "0$^\circ$",
            "30$^\circ$N",
            "60$^\circ$N",
            "90$^\circ$N",
        ]
    )

    # Plot the longitude mean
    ax2.plot(lon_mean)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Data Mean")
    ax2.set_title("Longitude Mean of Data")
    ax2.set_xticks(np.arange(-180, 181, 60))
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]
    ax2.set_xticklabels(x_ticks_mark)
    ax2.set_yticks(np.arange(-90, 91, 30))
    ax2.set_yticklabels(
        [
            "90$^\circ$S",
            "60$^\circ$S",
            "30$^\circ$S",
            "0$^\circ$",
            "30$^\circ$N",
            "60$^\circ$N",
            "90$^\circ$N",
        ]
    )

    # Show the plot
    plt.show()


def plot_global_and_lat_lon_mean(
    var_data,
    var_name,
    var_min,
    var_max,
    time_spec,
    cmap_file="/RAID01/data/muqy/color/Var_color.txt",
):
    """plot the spatial distribution of the variable
    globally using cartopy, with latitude and longitude mean line plots
    """

    lon = np.linspace(-180, 179, 360)
    lat = np.linspace(-90, 89, 180)

    lons, lats = np.meshgrid(lon, lat)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("white")

    fig = plt.figure(figsize=(15, 8.3))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1, 9],
        height_ratios=[6, 1],
        wspace=0.15,
        hspace=0.2,
    )

    # Set the font to Times New Roman
    plt.rcParams.update({"font.family": "Times New Roman"})

    # Create the global plot
    ax = fig.add_subplot(
        gs[0, 1], projection=ccrs.PlateCarree(central_longitude=0)
    )

    ax.set_facecolor("silver")
    b = ax.pcolormesh(
        lon,
        lat,
        np.nanmean(var_data, axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=var_min,
        vmax=var_max,
    )
    ax.coastlines(resolution="50m", lw=0.9)

    # Add the gridlines and colorbar
    gl = ax.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False

    plt.title(
        "Aviation Distance Difference " + time_spec,
        fontsize=24,
        # pad=20,
    )
    # colorbar

    cax = fig.add_axes([0.96, 0.29, 0.015, 0.58])
    cb = plt.colorbar(
        b,
        orientation="vertical",
        extend="both",
        aspect=45,
        cax=cax,
    )

    cb.set_label(label=var_name, size=24)
    cb.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    # mean along the latitude and longitude dimensions
    lat_mean = np.mean(var_data, axis=(0, 2))
    lon_mean = np.mean(var_data, axis=(0, 1))

    # Plot the latitude mean in the left subplot
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(
        lat_mean, np.linspace(-90, 89, len(lat_mean)), color="black"
    )
    ax.set_xlim(var_min, var_max * 0.5)
    ax.set_ylim(-90, 89)
    ax.set_xticks([])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ax.yaxis.set_tick_params(width=2)
    # ax.set_ylabel("Latitude", fontsize=18)
    ax.invert_xaxis()
    ax.set_yticks([])
    ax.fill_betweenx(
        np.linspace(-90, 89, len(lat_mean)),
        lat_mean,
        var_min,
        color="#2878B5",
        alpha=0.4,
    )

    # Plot the longitude mean in the bottom subplot
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(
        np.linspace(-180, 179, len(lon_mean)),
        lon_mean,
        color="black",
    )
    ax.set_xlim(-180, 179)
    ax.set_ylim(var_min, var_max * 0.2)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([])
    # ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.xaxis.set_tick_params(width=2)
    # ax.set_xlabel("Longitude", fontsize=18)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.fill_between(
        np.linspace(-180, 179, len(lon_mean)),
        lon_mean,
        var_min,
        color="#2878B5",
        alpha=0.4,
    )

    # plt.title(
    #     "Aviation Distance Difference " + time_spec,
    #     fontsize=24,
    #     # pad=20,
    # )
    plt.savefig(
        "/RAID01/data/python_fig/Aviation Distance Difference "
        + time_spec
        + ".png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
        edgecolor="w",
    )
    plt.show()


# -----------------------------------------------------------------------------------
# Plot PC and Cld data
# -----------------------------------------------------------------------------------


def plot_Cld_no_mean_simple_full_hemisphere_self_cmap(
    Cld_match_PC_gap,
    # p_value,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
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
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)

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
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


def plot_Cld_no_mean_simple_partial_self_cmap(
    Cld_match_PC_gap,
    # p_value,
    cld_min,
    cld_max,
    cld_name,
    lon=np.linspace(0, 359, 360),
    lat=np.linspace(-90, 89, 180),
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
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
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)

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
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


def plot_Cld_no_mean_simple_tropical_self_cmap(
    Cld_match_PC_gap,
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
    ax1.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
    ax1.set_facecolor("silver")
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_match_PC_gap,
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

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


def plot_Cld_no_mean_simple_north_polar_self_cmap(
    Cld_match_PC_gap,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

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
        figsize=(14, 11),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.NorthPolarStereo(),
    )
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())
    ax1.coastlines(resolution="50m", lw=0.9)
    # gl = ax1.gridlines(
    #     linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    # )
    gl = ax1.gridlines(
        linestyle="--",
        ylocs=np.arange(65, 90, 5),
        xlocs=np.arange(-180, 180, 30),
        draw_labels=True,
    )
    gl.xlabels_top = True
    # gl.ylabels_right = True

    gl.xformatter = LongitudeFormatter()

    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1.set_boundary(circle, transform=ax1.transAxes)


def plot_Cld_no_mean_simple_sourth_polar_self_cmap(
    Cld_match_PC_gap,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

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
        figsize=(14, 11),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.SouthPolarStereo(),
    )
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
    ax1.coastlines(resolution="50m", lw=0.9)
    # gl = ax1.gridlines(
    #     linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    # )
    gl = ax1.gridlines(
        linestyle="--",
        ylocs=np.arange(-90, -65, 5),
        xlocs=np.arange(-180, 180, 30),
        draw_labels=True,
    )
    gl.xlabels_top = True
    # gl.ylabels_right = True

    gl.xformatter = LongitudeFormatter()

    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1.set_boundary(circle, transform=ax1.transAxes)


def plot_Cld_no_mean_simple_sourth_polar_half_hemisphere(
    Cld_match_PC_gap,
    cld_min,
    cld_max,
    cld_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
):
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, -1, 90)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    # cmap.set_under("white")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(14, 11),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.SouthPolarStereo(),
    )
    # ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        Cld_match_PC_gap,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=cld_min,
        vmax=cld_max,
    )
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.coastlines(resolution="50m", lw=1.7)
    # gl = ax1.gridlines(
    #     linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    # )
    gl = ax1.gridlines(
        linestyle="--",
        ylocs=np.arange(-90, -50, 5),
        xlocs=np.arange(-180, 180, 30),
        draw_labels=True,
    )
    gl.xlabels_top = True
    # gl.ylabels_right = True

    gl.xformatter = LongitudeFormatter()

    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.8,
        extend="both",
    )
    cb2.set_label(label=cld_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1.set_boundary(circle, transform=ax1.transAxes)


def compare_cld_between_PC_condition_by_Lat(
    Cld_data_PC_condition, Cld_data_aux
):
    """
    Plot mean cld anormaly from lat 20 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    Cld_data_PC_condition = np.concatenate(
        (
            Cld_data_PC_condition[:, 180:],
            Cld_data_PC_condition[:, :180],
        ),
        axis=1,
    )
    Cld_data_aux = np.concatenate(
        (Cld_data_aux[:, 180:], Cld_data_aux[:, :180]), axis=1
    )

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_PC_condition[110:150, :], axis=0),
        color="Blue",
        linewidth=2,
        label="With PC constraint",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    ax.set_facecolor("white")
    ax.legend()
    # adjust the legend font size
    for text in ax.get_legend().get_texts():
        plt.setp(text, color="black", fontsize=20)

    # x_ticks_mark = [
    #     "60$^\circ$E",
    #     "120$^\circ$E",
    #     "180$^\circ$",
    #     "120$^\circ$W",
    #     "60$^\circ$W",
    # ]
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    plt.xlim([-180, 180])
    plt.xlabel("Longitude", size=23, weight="bold")
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20, weight="bold")
    plt.yticks(fontsize=20, weight="bold")
    plt.ylabel("HCF difference (%)", size=20, weight="bold")
    # plt.ylim(0, 100)
    plt.show()


def compare_cld_between_PC_condition_by_Lat_smoothed(
    Cld_data_PC_condition,
    Cld_data_aux,
    Cld_data_name,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    Cld_data_PC_condition = np.concatenate(
        (
            Cld_data_PC_condition[:, 180:],
            Cld_data_PC_condition[:, :180],
        ),
        axis=1,
    )
    Cld_data_aux = np.concatenate(
        (Cld_data_aux[:, 180:], Cld_data_aux[:, :180]), axis=1
    )

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition[110:150, :], axis=0),
            step,
            mode="same",
        ),
        color="Blue",
        linewidth=2,
        label="With PC constraint",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    ax.set_facecolor("white")
    ax.legend()
    # adjust the legend font size
    for text in ax.get_legend().get_texts():
        plt.setp(text, color="black", fontsize=20)

    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    plt.xlim([-180, 180])
    plt.xlabel("Longitude", size=23, weight="bold")
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20, weight="bold")
    plt.yticks(fontsize=20, weight="bold")
    plt.ylabel(Cld_data_name, size=20, weight="bold")
    # plt.ylim(0, 100)
    plt.show()


def compare_cld_between_PC_condition_by_each_Lat_smoothed(
    Cld_data_PC_condition_0,
    Cld_data_PC_condition_1,
    Cld_data_PC_condition_2,
    Cld_data_aux,
    Cld_data_name,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    Cld_data_PC_condition_0 = np.concatenate(
        (
            Cld_data_PC_condition_0[:, 180:],
            Cld_data_PC_condition_0[:, :180],
        ),
        axis=1,
    )
    Cld_data_PC_condition_1 = np.concatenate(
        (
            Cld_data_PC_condition_1[:, 180:],
            Cld_data_PC_condition_1[:, :180],
        ),
        axis=1,
    )
    Cld_data_PC_condition_2 = np.concatenate(
        (
            Cld_data_PC_condition_2[:, 180:],
            Cld_data_PC_condition_2[:, :180],
        ),
        axis=1,
    )
    Cld_data_aux = np.concatenate(
        (Cld_data_aux[:, 180:], Cld_data_aux[:, :180]), axis=1
    )

    fig, axs = plt.subplots(
        figsize=(18, 16), nrows=3, ncols=1, sharex=True
    )

    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_0[110:150, :], axis=0),
            step,
            mode="same",
        ),
        color="#9DC3E7",
        linewidth=2.5,
        label="Bad Atmospheric Condition",
        alpha=1,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_1[110:150, :], axis=0),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition",
        alpha=0.95,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_2[110:150, :], axis=0),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition",
        alpha=0.95,
    )

    # Auxiliary lines representing the data without PC1 constrain and 0
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    for axs in axs:
        axs.set_facecolor("white")
        axs.legend(prop={"size": 20})
        # axs.set_yticks(fontsize=20, weight="bold")
        axs.tick_params(axis="y", labelsize=20)
        axs.set_ylabel(Cld_data_name, size=20)

    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    # adjust the legend font size
    # for text in axs[0].get_legend().get_texts():
    #     plt.setp(text, color="black", fontsize=20)

    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    plt.xlim([-180, 180])
    plt.xlabel("Longitude", size=23)
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20)
    plt.show()


def compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill(
    Cld_data_PC_condition_0,
    Cld_data_PC_condition_1,
    Cld_data_PC_condition_2,
    Cld_data_aux,
    Cld_data_name,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    Cld_data_PC_condition_0 = np.concatenate(
        (
            Cld_data_PC_condition_0[:, 180:],
            Cld_data_PC_condition_0[:, :180],
        ),
        axis=1,
    )
    Cld_data_PC_condition_1 = np.concatenate(
        (
            Cld_data_PC_condition_1[:, 180:],
            Cld_data_PC_condition_1[:, :180],
        ),
        axis=1,
    )
    Cld_data_PC_condition_2 = np.concatenate(
        (
            Cld_data_PC_condition_2[:, 180:],
            Cld_data_PC_condition_2[:, :180],
        ),
        axis=1,
    )
    Cld_data_aux = np.concatenate(
        (Cld_data_aux[:, 180:], Cld_data_aux[:, :180]), axis=1
    )

    fig, axs = plt.subplots(
        figsize=(18, 15), nrows=3, ncols=1, sharex=True
    )

    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_0[110:150, :], axis=0),
            step,
            mode="same",
        ),
        color="#9DC3E7",
        linewidth=2.5,
        label="Bad Atmospheric Condition",
        alpha=1,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_1[110:150, :], axis=0),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition",
        alpha=0.95,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(Cld_data_PC_condition_2[110:150, :], axis=0),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition",
        alpha=0.95,
    )

    # Auxiliary lines representing the data without PC1 constrain and 0
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    # adjust the subplots to make it more compact
    fig.subplots_adjust(hspace=0.05)

    # set the universal x axis parameters
    # set the xlabel to the longitude axis
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    # set the xticks label
    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    # set the xlim
    plt.xlim([-180, 180])
    # set the xticks label
    plt.xlabel("Longitude", size=23)
    # set the xticks
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20)

    # set the y axis parameters
    for axs in axs:
        # get the y axis limits first
        y_limits = axs.get_ylim()

        # set the background color
        axs.set_facecolor("white")
        # set the legend
        axs.legend(prop={"size": 20})
        # axs.set_yticks(fontsize=20, weight="bold")
        axs.tick_params(axis="y", labelsize=20)
        axs.set_ylabel(Cld_data_name, size=20)

        # fill the USA aviation area
        axs.fill_betweenx(
            np.linspace(y_limits[0], y_limits[1], 100),
            -110,
            -70,
            color="grey",
            alpha=0.2,
        )
        # fill the Euro aviation area
        axs.fill_betweenx(
            np.linspace(y_limits[0], y_limits[1], 100),
            -10,
            40,
            color="grey",
            alpha=0.2,
        )
        # fill the Asia aviation area
        axs.fill_betweenx(
            np.linspace(y_limits[0], y_limits[1], 100),
            100,
            130,
            color="grey",
            alpha=0.2,
        )

    plt.savefig(
        "/RAID01/data/python_fig/" + Cld_data_name + ".png",
        dpi=300,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
    )
    plt.show()


def compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_z(
    Cld_data_PC_condition_0_mean,
    Cld_data_PC_condition_1_mean,
    Cld_data_PC_condition_2_mean,
    Cld_data_PC_condition_0_median,
    Cld_data_PC_condition_1_median,
    Cld_data_PC_condition_2_median,
    Cld_data_aux,
    Cld_data_name,
    title,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60
    Now we try 20 to 50 to reduce noise

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    def shift_array_columns(arr, n):
        """Shift the columns of an array by n positions."""
        return np.concatenate((arr[:, n:], arr[:, :n]), axis=1)

    # Usage, flip the array 180 degrees
    Cld_data_PC_condition_0_mean = stats.zscore(
        shift_array_columns(Cld_data_PC_condition_0_mean, 180)
    )
    Cld_data_PC_condition_1_mean = stats.zscore(
        shift_array_columns(Cld_data_PC_condition_1_mean, 180)
    )
    Cld_data_PC_condition_2_mean = stats.zscore(
        shift_array_columns(Cld_data_PC_condition_2_mean, 180)
    )

    Cld_data_PC_condition_0_median = stats.zscore(
        shift_array_columns(Cld_data_PC_condition_0_median, 180)
    )
    Cld_data_PC_condition_1_median = stats.zscore(
        shift_array_columns(Cld_data_PC_condition_1_median, 180)
    )
    Cld_data_PC_condition_2_median = stats.zscore(
        shift_array_columns(Cld_data_PC_condition_2_median, 180)
    )

    Cld_data_aux = stats.zscore(
        shift_array_columns(Cld_data_aux, 180)
    )

    # plot the figure
    fig, axs = plt.subplots(
        figsize=(18, 15), nrows=3, ncols=1, sharex=True
    )

    # plot lines
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_0_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9DC3E7",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Mean)",
        alpha=1,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_0_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9DC3E7",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Median)",
        linestyle="--",
        alpha=1,
    )

    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_1_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_1_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_2_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_2_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    # Auxiliary lines representing the data without PC1 constrain and 0
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    # set the title
    axs[0].set_title(title, size=26, y=1.02)

    # adjust the subplots to make it more compact
    fig.subplots_adjust(hspace=0.05)

    # set the universal x axis parameters
    # set the xlabel to the longitude axis
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    # set the xticks label
    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    # set the xlim
    plt.xlim([-180, 180])
    # set the xticks label
    plt.xlabel("Longitude", size=23)
    # set the xticks
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20)

    # set the y axis parameters
    for axs in axs:
        # get the y axis limits first
        y_limits = axs.get_ylim()

        # set the background color
        axs.set_facecolor("white")
        # set the legend
        axs.legend(prop={"size": 20})
        # axs.set_yticks(fontsize=20, weight="bold")
        axs.tick_params(axis="y", labelsize=20)
        axs.set_ylabel(Cld_data_name, size=20)

        # fill the USA aviation area
        axs.fill_betweenx(
            np.linspace(y_limits[0], y_limits[1], 100),
            -110,
            -70,
            color="grey",
            alpha=0.2,
        )
        # fill the Euro aviation area
        axs.fill_betweenx(
            np.linspace(y_limits[0], y_limits[1], 100),
            -10,
            40,
            color="grey",
            alpha=0.2,
        )
        # fill the Asia aviation area
        axs.fill_betweenx(
            np.linspace(y_limits[0], y_limits[1], 100),
            100,
            130,
            color="grey",
            alpha=0.2,
        )

    plt.savefig(
        "/RAID01/data/python_fig/"
        + Cld_data_name
        + title
        + "_mean_median.png",
        dpi=300,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
    )
    plt.show()


def compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
    Cld_data_PC_condition_0_mean,
    Cld_data_PC_condition_1_mean,
    Cld_data_PC_condition_2_mean,
    Cld_data_PC_condition_0_median,
    Cld_data_PC_condition_1_median,
    Cld_data_PC_condition_2_median,
    Cld_data_aux,
    Cld_data_name,
    y_lim_lst,
    title,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60
    Now we try 20 to 50 to reduce noise

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Test the aviation fly distance data from 2019 and 2020
    # Data is from "10.1029/2021AV000546"
    aviation_data_2019 = read_nc_data(
        file_path="/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Aviation/data/flight/kmflown_cruising_2019.nc"
    )
    aviation_data_2020 = read_nc_data(
        file_path="/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Aviation/data/flight/kmflown_cruising_2020.nc"
    )

    aviation_kmflown_2019 = aviation_data_2019["cruise"]
    aviation_kmflown_2020 = aviation_data_2020["cruise"]

    # ------------------ get the difference between 2019 and 2020 ------------------
    difference_2020_2019 = np.empty((240, 180, 360))
    difference_2020_2019 = np.array(
        aviation_kmflown_2020[:240, :, :]
    ) - np.array(aviation_kmflown_2019[:240, :, :])

    # get the difference between 2019 and 2020 in jan and feb only
    difference_2020_2019_jan_feb = difference_2020_2019[:60, :, :]
    difference_2020_2019_mar_apr = difference_2020_2019[
        60:120, :, :
    ]
    difference_2020_2019_may_jun = difference_2020_2019[
        120:180, :, :
    ]
    difference_2020_2019_jul_aug = difference_2020_2019[
        180:240, :, :
    ]

    # 1 - 2 month
    aviation_kmflown_2020_2019_1_2_month = np.nanmean(
        difference_2020_2019_jan_feb[:, 110:140, :],
        axis=(0, 1),
    )
    # 3 - 4 month
    aviation_kmflown_2020_2019_3_4_month = np.nanmean(
        difference_2020_2019_mar_apr[:, 110:140, :],
        axis=(0, 1),
    )
    # 5 - 6 month
    aviation_kmflown_2020_2019_5_6_month = np.nanmean(
        difference_2020_2019_may_jun[:, 110:140, :],
        axis=(0, 1),
    )
    # 7 - 8 month
    aviation_kmflown_2020_2019_7_8_month = np.nanmean(
        difference_2020_2019_jul_aug[:, 110:140, :],
        axis=(0, 1),
    )

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    def shift_array_columns(arr, n):
        """Shift the columns of an array by n positions."""
        return np.concatenate((arr[:, n:], arr[:, :n]), axis=1)

    # Usage, flip the array 180 degrees
    Cld_data_PC_condition_0_mean = shift_array_columns(
        Cld_data_PC_condition_0_mean, 180
    )
    Cld_data_PC_condition_1_mean = shift_array_columns(
        Cld_data_PC_condition_1_mean, 180
    )
    Cld_data_PC_condition_2_mean = shift_array_columns(
        Cld_data_PC_condition_2_mean, 180
    )

    Cld_data_PC_condition_0_median = shift_array_columns(
        Cld_data_PC_condition_0_median, 180
    )
    Cld_data_PC_condition_1_median = shift_array_columns(
        Cld_data_PC_condition_1_median, 180
    )
    Cld_data_PC_condition_2_median = shift_array_columns(
        Cld_data_PC_condition_2_median, 180
    )

    Cld_data_aux = shift_array_columns(Cld_data_aux, 180)

    # plot the figure
    fig, axs = plt.subplots(
        figsize=(18, 15), nrows=3, ncols=1, sharex=True
    )

    # plot lines
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_0_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#82A6B1",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Mean)",
        alpha=1,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_0_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#82A6B1",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Median)",
        linestyle="--",
        alpha=1,
    )

    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_1_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_1_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_2_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_2_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    # Auxiliary lines representing the data without PC1 constrain and 0
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_aux[110:150, :], axis=0),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    # set the title
    axs[0].set_title(title, size=26, y=1.02)

    # set ylim for each subplot
    # axs[0].set_ylim(y_lim_lst[0])
    # axs[1].set_ylim(y_lim_lst[1])
    # axs[2].set_ylim(y_lim_lst[2])

    # adjust the subplots to make it more compact
    fig.subplots_adjust(hspace=0.05)

    # set the universal x axis parameters
    # set the xlabel to the longitude axis
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    # set the xticks label
    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    # set the xlim
    plt.xlim([-180, 180])
    # set the xticks label
    plt.xlabel("Longitude", size=23)
    # set the xticks
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20)

    # set the y axis parameters
    for axs in axs:
        # get the y axis limits first
        y1_min, y1_max = axs.get_ylim()

        # set the background color
        axs.set_facecolor("white")
        # set the legend
        axs.legend(prop={"size": 20})
        # axs.set_yticks(fontsize=20, weight="bold")
        axs.tick_params(axis="y", labelsize=20)
        axs.set_ylabel(Cld_data_name, size=20)

        # plot the secondary y axis
        axs_sec_y = axs.twinx()

        # fill between the line of aviation_kmflown_2019_2020_1_2_month and the y axis 0
        if title == "January-February":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_1_2_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_1_2_month,
                0,
                where=aviation_kmflown_2020_2019_1_2_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_1_2_month,
                0,
                where=aviation_kmflown_2020_2019_1_2_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "March-April":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_3_4_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_3_4_month,
                0,
                where=aviation_kmflown_2020_2019_3_4_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_3_4_month,
                0,
                where=aviation_kmflown_2020_2019_3_4_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "May-June":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_5_6_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_5_6_month,
                0,
                where=aviation_kmflown_2020_2019_5_6_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_5_6_month,
                0,
                where=aviation_kmflown_2020_2019_5_6_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "July-August":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_7_8_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_7_8_month,
                0,
                where=aviation_kmflown_2020_2019_7_8_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_7_8_month,
                0,
                where=aviation_kmflown_2020_2019_7_8_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )

    plt.savefig(
        "/RAID01/data/python_fig/"
        + Cld_data_name
        + title
        + "_mean_median_fill_aviation.png",
        dpi=300,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
    )
    plt.show()


def compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
    Cld_data_PC_condition_0_mean,
    Cld_data_PC_condition_1_mean,
    Cld_data_PC_condition_2_mean,
    Cld_data_PC_condition_0_median,
    Cld_data_PC_condition_1_median,
    Cld_data_PC_condition_2_median,
    Cld_data_aux,
    Cld_data_name,
    y_lim_lst,
    title,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60
    Now we try 20 to 50 to reduce noise

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Test the aviation fly distance data from 2019 and 2020
    # Data is from "10.1029/2021AV000546"
    aviation_data_2019 = read_nc_data(
        file_path="/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Aviation/data/flight/kmflown_cruising_2019.nc"
    )
    aviation_data_2020 = read_nc_data(
        file_path="/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Aviation/data/flight/kmflown_cruising_2020.nc"
    )

    aviation_kmflown_2019 = aviation_data_2019["cruise"]
    aviation_kmflown_2020 = aviation_data_2020["cruise"]

    # ------------------ get the difference between 2019 and 2020 ------------------
    difference_2020_2019 = np.empty((240, 180, 360))
    difference_2020_2019 = np.array(
        aviation_kmflown_2020[:240, :, :]
    ) - np.array(aviation_kmflown_2019[:240, :, :])

    # get the difference between 2019 and 2020 in jan and feb only
    difference_2020_2019_jan_feb = difference_2020_2019[:60, :, :]
    difference_2020_2019_mar_apr = difference_2020_2019[
        60:120, :, :
    ]
    difference_2020_2019_may_jun = difference_2020_2019[
        120:180, :, :
    ]
    difference_2020_2019_jul_aug = difference_2020_2019[
        180:240, :, :
    ]

    # 1 - 2 month
    aviation_kmflown_2020_2019_1_2_month = np.nanmean(
        difference_2020_2019_jan_feb[:, 110:140, :],
        axis=(0, 1),
    )
    # 3 - 4 month
    aviation_kmflown_2020_2019_3_4_month = np.nanmean(
        difference_2020_2019_mar_apr[:, 110:140, :],
        axis=(0, 1),
    )
    # 5 - 6 month
    aviation_kmflown_2020_2019_5_6_month = np.nanmean(
        difference_2020_2019_may_jun[:, 110:140, :],
        axis=(0, 1),
    )
    # 7 - 8 month
    aviation_kmflown_2020_2019_7_8_month = np.nanmean(
        difference_2020_2019_jul_aug[:, 110:140, :],
        axis=(0, 1),
    )

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    def shift_array_columns(arr, n):
        """Shift the columns of an array by n positions."""
        return np.concatenate((arr[:, n:], arr[:, :n]), axis=1)

    # Usage, flip the array 180 degrees
    Cld_data_PC_condition_0_mean = shift_array_columns(
        Cld_data_PC_condition_0_mean, 180
    )
    Cld_data_PC_condition_1_mean = shift_array_columns(
        Cld_data_PC_condition_1_mean, 180
    )
    Cld_data_PC_condition_2_mean = shift_array_columns(
        Cld_data_PC_condition_2_mean, 180
    )

    Cld_data_PC_condition_0_median = shift_array_columns(
        Cld_data_PC_condition_0_median, 180
    )
    Cld_data_PC_condition_1_median = shift_array_columns(
        Cld_data_PC_condition_1_median, 180
    )
    Cld_data_PC_condition_2_median = shift_array_columns(
        Cld_data_PC_condition_2_median, 180
    )

    Cld_data_aux = shift_array_columns(Cld_data_aux, 180)

    # plot the figure
    fig, axs = plt.subplots(
        figsize=(18, 15), nrows=3, ncols=1, sharex=True
    )

    # plot lines
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_0_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#82A6B1",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Mean)",
        alpha=1,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_0_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#82A6B1",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Median)",
        linestyle="--",
        alpha=1,
    )

    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_1_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_1_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_2_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_2_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    # Auxiliary lines representing the data without PC1 constrain and 0
    # axs[0].plot(
    #     np.linspace(-180, 179, 360),
    #     np.nanmean(Cld_data_aux[110:150, :], axis=0),
    #     color="#D76364",
    #     label="Without PC constraint",
    #     linewidth=2.5,
    # )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    # axs[1].plot(
    #     np.linspace(-180, 179, 360),
    #     np.nanmean(Cld_data_aux[110:150, :], axis=0),
    #     color="#D76364",
    #     label="Without PC constraint",
    #     linewidth=2.5,
    # )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    # axs[2].plot(
    #     np.linspace(-180, 179, 360),
    #     np.nanmean(Cld_data_aux[110:150, :], axis=0),
    #     color="#D76364",
    #     label="Without PC constraint",
    #     linewidth=2.5,
    # )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    # set the title
    axs[0].set_title(title, size=26, y=1.02)

    # set ylim for each subplot
    # axs[0].set_ylim(y_lim_lst[0])
    # axs[1].set_ylim(y_lim_lst[1])
    # axs[2].set_ylim(y_lim_lst[2])

    # adjust the subplots to make it more compact
    fig.subplots_adjust(hspace=0.05)

    # set the universal x axis parameters
    # set the xlabel to the longitude axis
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    # set the xticks label
    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    # set the xlim
    plt.xlim([-180, 180])
    # set the xticks label
    plt.xlabel("Longitude", size=23)
    # set the xticks
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20)

    # set the y axis parameters
    for axs in axs:
        # get the y axis limits first
        y1_min, y1_max = axs.get_ylim()

        # set the background color
        axs.set_facecolor("white")
        # set the legend
        axs.legend(prop={"size": 20})
        # axs.set_yticks(fontsize=20, weight="bold")
        axs.tick_params(axis="y", labelsize=20)
        axs.set_ylabel(Cld_data_name, size=20)

        # plot the secondary y axis
        axs_sec_y = axs.twinx()

        # fill between the line of aviation_kmflown_2019_2020_1_2_month and the y axis 0
        if title == "January-February":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_1_2_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_1_2_month,
                0,
                where=aviation_kmflown_2020_2019_1_2_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_1_2_month,
                0,
                where=aviation_kmflown_2020_2019_1_2_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "March-April":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_3_4_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_3_4_month,
                0,
                where=aviation_kmflown_2020_2019_3_4_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_3_4_month,
                0,
                where=aviation_kmflown_2020_2019_3_4_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "May-June":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_5_6_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_5_6_month,
                0,
                where=aviation_kmflown_2020_2019_5_6_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_5_6_month,
                0,
                where=aviation_kmflown_2020_2019_5_6_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "July-August":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_7_8_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_7_8_month,
                0,
                where=aviation_kmflown_2020_2019_7_8_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_7_8_month,
                0,
                where=aviation_kmflown_2020_2019_7_8_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )

    plt.savefig(
        "/RAID01/data/python_fig/"
        + Cld_data_name
        + title
        + "_mean_median_fill_aviation_no_compare_2017_2019.png",
        dpi=300,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
    )
    plt.show()


def compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
    Cld_data_PC_condition_0_mean,
    Cld_data_PC_condition_1_mean,
    Cld_data_PC_condition_2_mean,
    Cld_data_PC_condition_0_median,
    Cld_data_PC_condition_1_median,
    Cld_data_PC_condition_2_median,
    Cld_data_aux,
    Cld_data_name,
    y_lim_lst,
    title,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60
    Now we try 20 to 50 to reduce noise

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Test the aviation fly distance data from 2019 and 2020
    # Data is from "10.1029/2021AV000546"
    aviation_data_2019 = read_nc_data(
        file_path="/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Aviation/data/flight/kmflown_cruising_2019.nc"
    )
    aviation_data_2020 = read_nc_data(
        file_path="/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Aviation/data/flight/kmflown_cruising_2020.nc"
    )

    aviation_kmflown_2019 = aviation_data_2019["cruise"]
    aviation_kmflown_2020 = aviation_data_2020["cruise"]

    # ------------------ get the difference between 2019 and 2020 ------------------
    difference_2020_2019 = np.empty((240, 180, 360))
    difference_2020_2019 = np.array(
        aviation_kmflown_2020[:240, :, :]
    ) - np.array(aviation_kmflown_2019[:240, :, :])

    # get the difference between 2019 and 2020 in jan and feb only
    difference_2020_2019_jan_feb = difference_2020_2019[:60, :, :]
    difference_2020_2019_mar_apr = difference_2020_2019[
        60:120, :, :
    ]
    difference_2020_2019_may_jun = difference_2020_2019[
        120:180, :, :
    ]
    difference_2020_2019_jul_aug = difference_2020_2019[
        180:240, :, :
    ]

    # 1 - 2 month
    aviation_kmflown_2020_2019_1_2_month = np.nanmean(
        difference_2020_2019_jan_feb[:, 110:140, :],
        axis=(0, 1),
    )
    # 3 - 4 month
    aviation_kmflown_2020_2019_3_4_month = np.nanmean(
        difference_2020_2019_mar_apr[:, 110:140, :],
        axis=(0, 1),
    )
    # 5 - 6 month
    aviation_kmflown_2020_2019_5_6_month = np.nanmean(
        difference_2020_2019_may_jun[:, 110:140, :],
        axis=(0, 1),
    )
    # 7 - 8 month
    aviation_kmflown_2020_2019_7_8_month = np.nanmean(
        difference_2020_2019_jul_aug[:, 110:140, :],
        axis=(0, 1),
    )

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    def shift_array_columns(arr, n):
        """Shift the columns of an array by n positions."""
        return np.concatenate((arr[:, n:], arr[:, :n]), axis=1)

    # Usage, flip the array 180 degrees
    Cld_data_PC_condition_0_mean = shift_array_columns(
        Cld_data_PC_condition_0_mean, 180
    )

    Cld_data_PC_condition_1_mean = shift_array_columns(
        Cld_data_PC_condition_1_mean, 180
    )

    Cld_data_PC_condition_2_mean = shift_array_columns(
        Cld_data_PC_condition_2_mean, 180
    )

    Cld_data_PC_condition_0_median = shift_array_columns(
        Cld_data_PC_condition_0_median, 180
    )

    Cld_data_PC_condition_1_median = shift_array_columns(
        Cld_data_PC_condition_1_median, 180
    )

    Cld_data_PC_condition_2_median = shift_array_columns(
        Cld_data_PC_condition_2_median, 180
    )

    Cld_data_aux = shift_array_columns(Cld_data_aux, 180)

    # plot the figure
    fig, axs = plt.subplots(
        figsize=(18, 15), nrows=3, ncols=1, sharex=True
    )

    # plot lines
    axs[0].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np_move_avg(
                np.nanmean(
                    Cld_data_PC_condition_0_mean[110:140, :], axis=0
                ),
                step,
                mode="same",
            )
        ),
        color="#82A6B1",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Mean)",
        alpha=1,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np_move_avg(
                np.nanmean(
                    Cld_data_PC_condition_0_median[110:140, :],
                    axis=0,
                ),
                step,
                mode="same",
            )
        ),
        color="#82A6B1",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Median)",
        linestyle="--",
        alpha=1,
    )

    axs[1].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np_move_avg(
                np.nanmean(
                    Cld_data_PC_condition_1_mean[110:140, :], axis=0
                ),
                step,
                mode="same",
            )
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np_move_avg(
                np.nanmean(
                    Cld_data_PC_condition_1_median[110:140, :],
                    axis=0,
                ),
                step,
                mode="same",
            )
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    axs[2].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np_move_avg(
                np.nanmean(
                    Cld_data_PC_condition_2_mean[110:140, :], axis=0
                ),
                step,
                mode="same",
            )
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np_move_avg(
                np.nanmean(
                    Cld_data_PC_condition_2_median[110:140, :],
                    axis=0,
                ),
                step,
                mode="same",
            )
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    # Auxiliary lines representing the data without PC1 constrain and 0
    axs[0].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np.nanmean(Cld_data_aux[110:150, :], axis=0)
        ),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np.nanmean(Cld_data_aux[110:150, :], axis=0)
        ),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        nan_array_normalize(
            np.nanmean(Cld_data_aux[110:150, :], axis=0)
        ),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    # set the title
    axs[0].set_title(title, size=26, y=1.02)

    # set ylim for each subplot
    axs[0].set_ylim(y_lim_lst[0])
    axs[1].set_ylim(y_lim_lst[1])
    axs[2].set_ylim(y_lim_lst[2])

    # adjust the subplots to make it more compact
    fig.subplots_adjust(hspace=0.05)

    # set the universal x axis parameters
    # set the xlabel to the longitude axis
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    # set the xticks label
    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    # set the xlim
    plt.xlim([-180, 180])
    # set the xticks label
    plt.xlabel("Longitude", size=23)
    # set the xticks
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20)

    # set the y axis parameters
    for axs in axs:
        # get the y axis limits first
        y1_min, y1_max = axs.get_ylim()

        # set the background color
        axs.set_facecolor("white")
        # set the legend
        axs.legend(prop={"size": 20})
        # axs.set_yticks(fontsize=20, weight="bold")
        axs.tick_params(axis="y", labelsize=20)
        axs.set_ylabel(Cld_data_name, size=20)

        # plot the secondary y axis
        axs_sec_y = axs.twinx()

        # fill between the line of aviation_kmflown_2019_2020_1_2_month and the y axis 0
        if title == "January-February":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_1_2_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_1_2_month,
                0,
                where=aviation_kmflown_2020_2019_1_2_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_1_2_month,
                0,
                where=aviation_kmflown_2020_2019_1_2_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "March-April":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_3_4_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_3_4_month,
                0,
                where=aviation_kmflown_2020_2019_3_4_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_3_4_month,
                0,
                where=aviation_kmflown_2020_2019_3_4_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "May-June":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_5_6_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_5_6_month,
                0,
                where=aviation_kmflown_2020_2019_5_6_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_5_6_month,
                0,
                where=aviation_kmflown_2020_2019_5_6_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "July-August":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_7_8_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_7_8_month,
                0,
                where=aviation_kmflown_2020_2019_7_8_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_7_8_month,
                0,
                where=aviation_kmflown_2020_2019_7_8_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )

    plt.savefig(
        "/RAID01/data/python_fig/"
        + Cld_data_name
        + title
        + "_mean_median_fill_aviation_no_compare_2017_2019.png",
        dpi=300,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
    )
    plt.show()


def compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
    Cld_data_PC_condition_0_mean,
    Cld_data_PC_condition_1_mean,
    Cld_data_PC_condition_2_mean,
    Cld_data_PC_condition_0_median,
    Cld_data_PC_condition_1_median,
    Cld_data_PC_condition_2_median,
    Cld_data_aux_proses,
    Cld_data_name,
    y_lim_lst,
    title,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60
    Now we try 20 to 50 to reduce noise

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Test the aviation fly distance data from 2019 and 2020
    # Data is from "10.1029/2021AV000546"
    aviation_data_2019 = read_nc_data(
        file_path="/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Aviation/data/flight/kmflown_cruising_2019.nc"
    )
    aviation_data_2020 = read_nc_data(
        file_path="/RAID01/data/muqy/PYTHON_CODE/Highcloud_Contrail/Aviation/data/flight/kmflown_cruising_2020.nc"
    )

    aviation_kmflown_2019 = aviation_data_2019["cruise"]
    aviation_kmflown_2020 = aviation_data_2020["cruise"]

    # ------------------ get the difference between 2019 and 2020 ------------------
    difference_2020_2019 = np.empty((240, 180, 360))
    difference_2020_2019 = np.array(
        aviation_kmflown_2020[:240, :, :]
    ) - np.array(aviation_kmflown_2019[:240, :, :])

    # get the difference between 2019 and 2020 in jan and feb only
    difference_2020_2019_jan_feb = difference_2020_2019[:60, :, :]
    difference_2020_2019_mar_apr = difference_2020_2019[
        60:120, :, :
    ]
    difference_2020_2019_may_jun = difference_2020_2019[
        120:180, :, :
    ]
    difference_2020_2019_jul_aug = difference_2020_2019[
        180:240, :, :
    ]

    # 1 - 2 month
    aviation_kmflown_2020_2019_1_2_month = np.nanmean(
        difference_2020_2019_jan_feb[:, 110:140, :],
        axis=(0, 1),
    )
    # 3 - 4 month
    aviation_kmflown_2020_2019_3_4_month = np.nanmean(
        difference_2020_2019_mar_apr[:, 110:140, :],
        axis=(0, 1),
    )
    # 5 - 6 month
    aviation_kmflown_2020_2019_5_6_month = np.nanmean(
        difference_2020_2019_may_jun[:, 110:140, :],
        axis=(0, 1),
    )
    # 7 - 8 month
    aviation_kmflown_2020_2019_7_8_month = np.nanmean(
        difference_2020_2019_jul_aug[:, 110:140, :],
        axis=(0, 1),
    )

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    def shift_array_columns(arr, n):
        """Shift the columns of an array by n positions."""
        return np.concatenate((arr[:, n:], arr[:, :n]), axis=1)

    # Usage, flip the array 180 degrees
    Cld_data_PC_condition_0_mean = shift_array_columns(
        Cld_data_PC_condition_0_mean, 180
    )
    Cld_data_PC_condition_1_mean = shift_array_columns(
        Cld_data_PC_condition_1_mean, 180
    )
    Cld_data_PC_condition_2_mean = shift_array_columns(
        Cld_data_PC_condition_2_mean, 180
    )

    Cld_data_PC_condition_0_median = shift_array_columns(
        Cld_data_PC_condition_0_median, 180
    )
    Cld_data_PC_condition_1_median = shift_array_columns(
        Cld_data_PC_condition_1_median, 180
    )
    Cld_data_PC_condition_2_median = shift_array_columns(
        Cld_data_PC_condition_2_median, 180
    )

    Cld_data_aux_proses = shift_array_columns(
        Cld_data_aux_proses, 180
    )

    # plot the figure
    fig, axs = plt.subplots(
        figsize=(18, 15), nrows=3, ncols=1, sharex=True
    )

    # plot lines
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_0_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#82A6B1",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Mean)",
        alpha=1,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_0_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#82A6B1",
        linewidth=2.5,
        label="Bad Atmospheric Condition (Median)",
        linestyle="--",
        alpha=1,
    )

    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_1_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_1_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#5F97D2",
        linewidth=2.5,
        label="Moderate Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_2_mean[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Mean)",
        alpha=0.95,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            np.nanmean(
                Cld_data_PC_condition_2_median[110:140, :], axis=0
            ),
            step,
            mode="same",
        ),
        color="#9394E7",
        linewidth=2.5,
        label="Good Atmospheric Condition (Median)",
        linestyle="--",
        alpha=0.95,
    )

    # Auxiliary lines representing the data without PC1 constrain and 0
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            Cld_data_aux_proses[0],
            step,
            mode="same",
        ),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[0].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            Cld_data_aux_proses[1],
            step,
            mode="same",
        ),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[1].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np_move_avg(
            Cld_data_aux_proses[2],
            step,
            mode="same",
        ),
        color="#D76364",
        label="Without PC constraint",
        linewidth=2.5,
    )
    axs[2].plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        # label="Without PC constraint",
        linewidth=3,
    )

    # set the title
    axs[0].set_title(title, size=26, y=1.02)

    # set xlim for each subplot
    # axs[0].set_ylim(y_lim_lst[0])
    # axs[1].set_ylim(y_lim_lst[1])
    # axs[2].set_ylim(y_lim_lst[2])

    # adjust the subplots to make it more compact
    fig.subplots_adjust(hspace=0.05)

    # set the universal x axis parameters
    # set the xlabel to the longitude axis
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    # set the xticks label
    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    # set the xlim
    plt.xlim([-180, 180])
    # set the xticks label
    plt.xlabel("Longitude", size=23)
    # set the xticks
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20)

    # set the y axis parameters
    for axs in axs:
        # get the y axis limits first
        y1_min, y1_max = axs.get_ylim()

        # set the background color
        axs.set_facecolor("white")
        # set the legend
        axs.legend(prop={"size": 20})
        # axs.set_yticks(fontsize=20, weight="bold")
        axs.tick_params(axis="y", labelsize=20)
        axs.set_ylabel(Cld_data_name, size=20)

        # plot the secondary y axis
        axs_sec_y = axs.twinx()

        # fill between the line of aviation_kmflown_2019_2020_1_2_month and the y axis 0
        if title == "January-February":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_1_2_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_1_2_month,
                0,
                where=aviation_kmflown_2020_2019_1_2_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_1_2_month,
                0,
                where=aviation_kmflown_2020_2019_1_2_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "March-April":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_3_4_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_3_4_month,
                0,
                where=aviation_kmflown_2020_2019_3_4_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_3_4_month,
                0,
                where=aviation_kmflown_2020_2019_3_4_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "May-June":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_5_6_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_5_6_month,
                0,
                where=aviation_kmflown_2020_2019_5_6_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_5_6_month,
                0,
                where=aviation_kmflown_2020_2019_5_6_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )
        elif title == "July-August":
            # calculate the ratio of the two y axis
            zoom_ratio = (
                np.nanmax(
                    np.abs(aviation_kmflown_2020_2019_7_8_month)
                )
            ) / (np.abs(y1_min) * 0.9)

            # set the secondary y axis parameters
            axs_sec_y.set_ylim(
                y1_min * zoom_ratio, y1_max * zoom_ratio
            )
            axs_sec_y.set_ylabel("Aviation Distance (km)", size=23)
            axs_sec_y.tick_params(axis="y", labelsize=20)

            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_7_8_month,
                0,
                where=aviation_kmflown_2020_2019_7_8_month >= 0,
                interpolate=True,
                color="#EDBFC6",
                alpha=0.65,
            )
            axs_sec_y.fill_between(
                np.linspace(-180, 179, 360),
                aviation_kmflown_2020_2019_7_8_month,
                0,
                where=aviation_kmflown_2020_2019_7_8_month <= 0,
                interpolate=True,
                color="#545863",
                alpha=0.3,
            )

    plt.savefig(
        "/RAID01/data/python_fig/"
        + Cld_data_name
        + title
        + "_mean_median_fill_aviation_improve.png",
        dpi=300,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
    )
    plt.show()


def compare_cld_anormally_in_spec_Lat_smoothed(
    Cld_data_anormally_each_year,
    year_lst,
    Cld_data_name,
    step,
):
    """
    Plot mean cld anormaly from lat 20 to 60
    Now we try 20 to 50 to reduce noise

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    step : int
        step of moving average
    """

    # Deal data to central in 0 longitude
    mpl.rc("font", family="Times New Roman")

    def shift_array_columns(arr, n):
        """Shift the columns of an array by n positions."""
        return np.concatenate(
            (arr[:, :, n:], arr[:, :, :n]), axis=2
        )

    # Usage, flip the array 180 degrees
    Cld_data_anormally_each_year = shift_array_columns(
        Cld_data_anormally_each_year, 180
    )

    # plot the figure
    fig, axs = plt.subplots(
        figsize=(18, 7),
    )

    color_lst = [
        "#4e639e",
        "#7fbfdd",
        "#dba053",
        "#ff997c",
        "#e54616",
    ]

    # auxiliary line 0
    axs.plot(
        np.linspace(-180, 179, 360),
        np.zeros(360),
        color="Black",
        linewidth=3,
    )

    # lines representing the data without PC1 constrain and 0
    # for each years in the data
    for i, year in enumerate(year_lst):
        axs.plot(
            np.linspace(-180, 179, 360),
            np.nanmean(
                Cld_data_anormally_each_year[i, 110:150, :], axis=0
            ),
            color=color_lst[i],
            label="Cld anormaly " + str(year),
            linewidth=2.5,
        )

    # adjust the subplots to make it more compact
    fig.subplots_adjust(hspace=0.05)

    # set the universal x axis parameters
    # set the xlabel to the longitude axis
    x_ticks_mark = [
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
        "0$^\circ$",
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
    ]

    # set the xticks label
    x_ticks = [-180, -120, -60, 0, 60, 120, 180]
    # set the xlim
    plt.xlim([-180, 180])
    # set the xticks label
    plt.xlabel("Longitude", size=23)
    # set the xticks
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20)

    # set the y axis parameters
    # get the y axis limits first
    y_limits = axs.get_ylim()

    # set the background color
    axs.set_facecolor("white")
    # set the legend
    axs.legend(prop={"size": 20})
    # axs.set_yticks(fontsize=20, weight="bold")
    axs.tick_params(axis="y", labelsize=20)
    axs.set_ylabel(Cld_data_name, size=20)

    # fill the USA aviation area
    axs.fill_betweenx(
        np.linspace(y_limits[0], y_limits[1], 100),
        -110,
        -70,
        color="grey",
        alpha=0.2,
    )
    # fill the Euro aviation area
    axs.fill_betweenx(
        np.linspace(y_limits[0], y_limits[1], 100),
        -10,
        40,
        color="grey",
        alpha=0.2,
    )
    # fill the Asia aviation area
    axs.fill_betweenx(
        np.linspace(y_limits[0], y_limits[1], 100),
        100,
        130,
        color="grey",
        alpha=0.2,
    )

    plt.savefig(
        "/RAID01/data/python_fig/"
        + Cld_data_name
        + "_cld_anoramlly.png",
        dpi=300,
        facecolor="white",
        edgecolor="white",
        bbox_inches="tight",
    )
    plt.show()


#########################################################
########### Filter CLD data to fit PC1 gap #######################################
########################################################


def NUMBA_FILTER_DATA_FIT_PC1_GAP(
    var1,
    var2,
    coef,
    gap_num,
    PC_gap_len,
    latitude_len,
    longitude_len,
    Cld_data,
    PC_data,
):
    """
    Filter the CLD data to fit PC1 gap (Numba version)

    Parameters
    ----------
    var1 : float
        _description_
    var2 : float
        _description_
    coef : float
        _description_
    gap_num : int
        gap number of PC1
    PC_gap_len : int
        length of the PC1 array
    latitude_len : int
        _description_
    longitude_len : int
        _description_
    Cld_data : array in shape
        (PC_gap_len, latitude_len, longitude_len)
        CLD_data
    PC_data : array in shape
        (PC_gap_len, latitude_len, longitude_len)
        PC1_data

    Returns
    -------
    Cld_match_PC_gap: array in shape
        (PC_gap_len, latitude_len, longitude_len)
        filtered CLD data
    PC_match_PC_gap: array in shape
        (PC_gap_len, latitude_len, longitude_len)
        filtered PC data

    """
    Cld_match_PC_gap = np.zeros(
        (PC_gap_len, latitude_len, longitude_len)
    )
    PC_match_PC_gap = np.zeros(
        (PC_gap_len, latitude_len, longitude_len)
    )
    print("Start filtering data")
    for lat in range(latitude_len):
        for lon in range(longitude_len):
            for gap_num in range(PC_gap_len):
                # Filter Cld data with gap, start and end with giving gap
                Cld_match_PC_gap[gap_num, lat, lon] = np.nanmean(
                    Cld_data[:, lat, lon][
                        np.where(
                            (
                                PC_data[:, lat, lon]
                                >= (np.array(gap_num + var1) * coef)
                            )
                            & (
                                PC_data[:, lat, lon]
                                < (np.array(gap_num + var2) * coef)
                            )
                        )
                    ]
                )
                # generate PC match PC gap as well to insure
                PC_match_PC_gap[gap_num, lat, lon] = np.nanmean(
                    PC_data[:, lat, lon][
                        np.where(
                            (
                                PC_data[:, lat, lon]
                                >= (np.array(gap_num + var1) * coef)
                            )
                            & (
                                PC_data[:, lat, lon]
                                < (np.array(gap_num + var2) * coef)
                            )
                        )
                    ]
                )

    return Cld_match_PC_gap, PC_match_PC_gap


class Filter_data_fit_PC1_gap_plot(object):
    def __init__(self, Cld_data, start, end, gap):
        self.start = start
        self.end = end
        self.gap = gap
        self.latitude = [i for i in range(0, Cld_data.shape[1], 1)]
        self.longitude = [i for i in range(0, Cld_data.shape[2], 1)]

    def Filter_data_fit_PC1_gap_new(self, Cld_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data, shape (PC_gap, lat, lon)
        PC_data : numpy.array
            PC data, shape (PC_gap, lat, lon)
        start : int
            min value pf PC, like -1
        end : int
            max value of PC, like 2
        gap : int
            Giving gap of PC, like 0.2

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """

        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num, " ******")
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
            " ******",
        )
        PC_gap = np.arange(int(gap_num))

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )

        for gap_num in PC_gap:
            pc_min = coef * (gap_num + var1)
            pc_max = coef * (gap_num + var2)
            mask = (PC_data >= pc_min) & (PC_data < pc_max)
            Cld_match_PC_gap[gap_num, :, :] = np.nanmean(
                np.where(mask, Cld_data, np.nan), axis=0
            )
            PC_match_PC_gap[gap_num, :, :] = np.nanmean(
                np.where(mask, PC_data, np.nan), axis=0
            )

        return Cld_match_PC_gap, PC_match_PC_gap

    def Filter_data_fit_PC1_gap_new_median(self, Cld_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data, shape (PC_gap, lat, lon)
        PC_data : numpy.array
            PC data, shape (PC_gap, lat, lon)
        start : int
            min value pf PC, like -1
        end : int
            max value of PC, like 2
        gap : int
            Giving gap of PC, like 0.2

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """

        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num, " ******")
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
            " ******",
        )
        PC_gap = np.arange(int(gap_num))

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )

        for gap_num in PC_gap:
            pc_min = coef * (gap_num + var1)
            pc_max = coef * (gap_num + var2)
            mask = (PC_data >= pc_min) & (PC_data < pc_max)
            Cld_match_PC_gap[gap_num, :, :] = np.nanmedian(
                np.where(mask, Cld_data, np.nan), axis=0
            )
            PC_match_PC_gap[gap_num, :, :] = np.nanmedian(
                np.where(mask, PC_data, np.nan), axis=0
            )

        return Cld_match_PC_gap, PC_match_PC_gap

    def Filter_data_fit_PC1_gap_each_day(self, Cld_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data
        PC_data : numpy.array
            PC data
        start : int
            Start PC value, like -1
        end : int
            End PC value, like 2
        gap : int
            Giving gap, like 0.2

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num)
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
        )
        PC_gap = [i for i in range(0, int(gap_num), 1)]

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        for lat in self.latitude:
            for lon in self.longitude:
                for gap_num in PC_gap:
                    # Filter Cld data with gap, start and end with giving gap
                    Cld_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        Cld_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )
                    # generate PC match PC gap as well to insure
                    PC_match_PC_gap[gap_num, lat, lon] = np.nanmean(
                        PC_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )

        return Cld_match_PC_gap, PC_match_PC_gap

    def numba_Filter_data_fit_PC1_gap(self, Cld_data, PC_data):
        """
        Call numba filter function

        Parameters
        ----------
        Cld_data : array in shape
            (PC_gap_len, latitude_len, longitude_len)
            CLD_data
        PC_data : array in shape
            (PC_gap_len, latitude_len, longitude_len)
            PC1_data
            Returns

        -------
        Same as the numba filter function

        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        PC_gap = [i for i in range(0, int(gap_num), 1)]

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num)
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
        )

        return NUMBA_FILTER_DATA_FIT_PC1_GAP(
            var1,
            var2,
            coef,
            gap_num,
            len(PC_gap),
            len(self.latitude),
            len(self.longitude),
            Cld_data,
            PC_data,
        )

    def give_loop_list_for_giving_gap(self):
        """
        Give the loop list for giving gap

        Parameters
        ----------
        start : int
            start of the loop
        end : int
            end of the loop
        gap : int
            gap

        Returns
        -------
        loop_list : list
            loop list
        """
        range = self.end - self.start
        loop_num = range / self.gap

        var1 = (
            self.start
            * (loop_num - 1)
            / ((self.end - self.gap) - self.start)
        )
        coefficient = self.start / var1
        var2 = self.gap / coefficient + var1

        return var1, var2, coefficient, loop_num

    def calc_correlation_PC1_Cld(self, PC_data, Cld_data):
        Correlation = np.zeros((180, 360))

        for i in range(180):
            for j in range(360):
                Correlation[i, j] = pd.Series(
                    PC_data[:, i, j]
                ).corr(
                    pd.Series(Cld_data[:, i, j]),
                    method="pearson",
                )

        return Correlation

    def plot_PC1_Cld(
        self, start, end, PC_match_PC_gap, Cld_match_PC_gap
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        # lat = np.linspace(-90, 89, 180)
        lat = np.linspace(-90, -1, 90)

        print("****** Start plot PC1 ******")
        fig, (ax1, ax2) = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(20, 20),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            211,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000", alpha=1)
        cmap.set_under("#191970", alpha=1)

        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # vmax=1.5,
            # vmin=-1.5,
            cmap=cmap,
        )
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cbar = fig.colorbar(
            a,
            ax=[ax1],
            location="right",
            shrink=0.9,
            extend="both",
            label="PC 1",
        )
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        cbar.ax.tick_params(labelsize=24)

        ax2 = plt.subplot(
            212,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        ax2.coastlines(resolution="50m", lw=0.3)
        ax2.set_global()
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # vmax=30,
            # vmin=0,
            cmap=cmap,
        )
        gl = ax2.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cbar = fig.colorbar(
            b,
            ax=[ax2],
            location="right",
            shrink=0.9,
            extend="both",
            label="HCF (%)",
        )
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        cbar.ax.tick_params(labelsize=24)
        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_PC1_Cld_Difference(
        self,
        start,
        end,
        PC_match_PC_gap,
        Cld_match_PC_gap,
        pc_max,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, -1, 90)

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        print("****** Start plot PC1 ******")
        fig, (ax1, ax2) = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(15, 15),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            211,
            projection=ccrs.PlateCarree(),
        )

        ax1.set_global()

        # norm1 = colors.CenteredNorm(halfrange=pc_max)
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # norm=norm1,
            vmax=pc_max,
            vmin=-pc_max,
            cmap=cmap,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb1 = fig.colorbar(
            a,
            ax=ax1,
            extend="both",
            location="right",
            shrink=0.8,
        )
        # adjust the colorbar label size
        cb1.set_label(label="PC 1", size=24)
        cb1.ax.tick_params(labelsize=24)

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        # set serial number for this subplot
        # ax1.text(
        #     0.05,
        #     0.95,
        #     "PC",
        #     transform=ax1.transAxes,
        #     fontsize=24,
        #     verticalalignment="top",
        # )

        ax2 = plt.subplot(
            212,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        ax2.set_global()
        norm2 = colors.CenteredNorm(halfrange=cld_max)
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            norm=norm2,
            cmap=cmap,
        )
        ax2.coastlines(resolution="50m", lw=0.9)
        gl = ax2.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = fig.colorbar(
            b,
            ax=ax2,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_Difference(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

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
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            norm=norm2,
            cmap=cmap,
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
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

        plt.savefig(
            "/RAID01/data/muqy/PYTHONFIG/Volc_shit/"
            + str(np.round((np.array(start + var1)) * coef, 2))
            + "_PC1_"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2))
            + ".png",
            dpi=250,
            facecolor=fig.get_facecolor(),
            transparent=True,
        )

    def plot_Cld_simple_shit(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

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
            Cld_match_PC_gap,
            transform=ccrs.PlateCarree(),
            norm=norm2,
            cmap=cmap,
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
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

        plt.savefig(
            "/RAID01/data/muqy/PYTHONFIG/Volc_shit/"
            + str(np.round((np.array(start + var1)) * coef, 2))
            + "_PC1_"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2))
            + "after.png",
            dpi=250,
            facecolor=fig.get_facecolor(),
            transparent=True,
        )

    def plot_Cld_simple_test_half_hemisphere(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

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
        mpl.style.use("seaborn-v0_8-ticks")
        mpl.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        # ax1.set_global()
        b = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
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
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_simple_test_full_hemisphere(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
        cld_name,
        cmap_file,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        cmap = dcmap(cmap_file)
        cmap.set_over("#800000")
        cmap.set_under("#191970")
        cmap.set_bad("silver", alpha=0)

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
        ax1.set_facecolor("silver")
        b = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(central_longitude=0),
            cmap=cmap,
            vmin=cld_min,
            vmax=cld_max,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb.set_label(label=cld_name, size=24)
        cb.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.5,
            y=1.11,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_simple_test_tropical(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
        cld_name,
        cmap_file,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        cmap = dcmap(cmap_file)
        cmap.set_over("#800000")
        cmap.set_under("#191970")
        cmap.set_bad("silver", alpha=0)

        print("****** Start plot PC1 ******")
        fig, (ax1) = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(12, 2.5),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        # ax1.set_global()
        ax1.set_extent([-180, 180, -30, 30], ccrs.PlateCarree())
        ax1.set_facecolor("silver")
        b = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(central_longitude=0),
            cmap=cmap,
            vmin=cld_min,
            vmax=cld_max,
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb = fig.colorbar(
            b,
            ax=ax1,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb.set_label(label=cld_name, size=24)
        cb.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.5,
            y=1.11,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_no_mean_full_hemisphere(
        self,
        start,
        end,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
        cld_name,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

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

        ax1.set_facecolor("silver")
        # ax1.set_global()
        b = ax1.pcolormesh(
            lon,
            lat,
            Cld_match_PC_gap,
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

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.5,
            y=1.11,
            size=42,
            fontweight="bold",
        )

    def plot_Cld_no_mean_simple_full_hemisphere(
        self,
        Cld_match_PC_gap,
        cld_min,
        cld_max,
        cld_name,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

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
        b = ax1.pcolormesh(
            lon,
            lat,
            Cld_match_PC_gap,
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

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        # plt.suptitle(
        #     str(np.round((np.array(start + var1)) * coef, 2))
        #     + "<=PC1<"
        #     + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
        #     x=0.5,
        #     y=1.11,
        #     size=42,
        #     fontweight="bold",
        # )

    def plot_PC1_Cld_test(
        self,
        start,
        end,
        Cld_data,
        cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        print("****** Start plot PC1 ******")
        fig, ax1 = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(15, 8),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        # norm1 = colors.CenteredNorm(halfrange=pc_max)
        a = ax1.pcolormesh(
            lon,
            lat,
            Cld_data,
            transform=ccrs.PlateCarree(),
            # norm=norm1,
            vmax=cld_max,
            vmin=-cld_max,
            cmap=cmap,
        )
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb1 = fig.colorbar(
            a,
            ax=ax1,
            extend="both",
            location="right",
            shrink=0.7,
        )
        # adjust the colorbar label size
        cb1.set_label(label="PC 1", size=24)
        cb1.ax.tick_params(labelsize=24)

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(np.round((np.array((end - 1) + var2)) * coef, 2)),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_All_year_mean_PC1_Cld(self, PC_data, Cld_data):
        # plot all year mean PC1 and Cld
        # ! Input PC_data and Cld_data must be the same shape
        # ! [time, lat, lon]
        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)
        # lon,lat1 = np.meshgrid(lon,lat1)

        print("****** Start plot all year mean PC1 and Cld ******")
        fig = plt.figure(figsize=(18, 15))
        plt.rc("font", size=10, weight="bold")

        cmap = dcmap("/RAID01/data/muqy/color/test_cld.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("white")

        cmap1 = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap1.set_bad("gray")
        cmap1.set_over("#800000")
        cmap1.set_under("#191970")

        ax1 = plt.subplot(
            2,
            1,
            1,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_data[:, :, :], axis=0),
            linewidth=0,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmax=40,
            vmin=0,
        )
        ax1.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        ax1.set_title(" High Cloud Fraction (HCF) ", size=12)
        fig.colorbar(
            a,
            ax=[ax1],
            location="right",
            shrink=0.9,
            extend="both",
            label="HCF (%)",
        )

        ax2 = plt.subplot(
            2,
            1,
            2,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax2.coastlines(resolution="50m", lw=0.7)
        ax2.set_global()
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_data[:, :, :], axis=0),
            linewidth=0,
            transform=ccrs.PlateCarree(),
            # norm=MidpointNormalize(midpoint=0),
            cmap=cmap1,
            vmax=2,
            vmin=-1,
        )
        ax2.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        ax2.set_title(" Principle Component 1 (PC1) ", size=12)
        fig.colorbar(
            b,
            ax=[ax2],
            location="right",
            shrink=0.9,
            extend="both",
            label="PC1",
        )
        # plt.savefig('PC1_CLDAREA1.pdf')
        # plt.tight_layout()
        plt.show()

    def plot_correlation_PC1_Cld(self, Corr_data):
        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)
        lat1 = np.linspace(0, 69, 70)
        # lon,lat1 = np.meshgrid(lon,lat1)

        fig = plt.figure(figsize=(10, 6))
        plt.rc("font", size=10, weight="bold")

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        ax1 = plt.subplot(
            1,
            1,
            1,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax1.coastlines(resolution="50m", lw=0.3)
        ax1.set_global()
        a = ax1.pcolor(
            lon,
            lat,
            Corr_data,
            # Corr_all,
            # Corr_d,
            linewidth=0,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmax=1,
            vmin=-1,
        )
        ax1.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        ax1.set_title(" PC1-HCF Correlation (Corr) ", size=12)
        fig.colorbar(
            a,
            ax=[ax1],
            location="right",
            shrink=0.9,
            extend="both",
            label="Corr",
        )

        plt.show()

    def Convert_pandas(self, Cld_match_PC_gap):
        gap_num = Cld_match_PC_gap.shape[0]
        Box = np.zeros(
            (
                Cld_match_PC_gap.shape[1]
                * Cld_match_PC_gap.shape[2],
                gap_num,
            )
        )
        # Box = np.zeros((gap_num, 64800, ))

        for i in range(gap_num):
            Box[:, i] = Cld_match_PC_gap[i, :, :].reshape(-1)

        Box = pd.DataFrame(Box)

        # Number of PC1 interval is set in np.arange(start, end, interval gap)
        Box.columns = np.round(
            np.arange(self.start, self.end, self.gap), 3
        )

        return Box

    def plot_box_plot(self, Cld_match_PC_gap, savefig_str):
        """
        Plot boxplot of Cld data match each PC1 interval
        Main plot function
        """
        Box = self.Convert_pandas(Cld_match_PC_gap)

        plt.style.use("seaborn-v0_8-ticks")  # type: ignore
        plt.rc("font", family="Times New Roman")

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
        plt.ylabel(savefig_str, size=26, weight="bold")
        # plt.xticks((np.round(np.arange(-1.5, 3.5, 0.5), 2)),fontsize=26, weight="bold", )
        plt.xticks(rotation=45)
        plt.yticks(
            fontsize=26,
            weight="bold",
        )
        os.makedirs("Box_plot", exist_ok=True)
        plt.savefig(
            "Box_plot/Box_plot_PC1_" + savefig_str + ".png",
            dpi=500,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.show()


class Filter_data_fit_PC1_gap_plot_IWP_constrain(object):
    def __init__(self, Cld_data, start, end, gap):
        self.start = start
        self.end = end
        self.gap = gap
        self.latitude = [i for i in range(0, Cld_data.shape[1], 1)]
        self.longitude = [i for i in range(0, Cld_data.shape[2], 1)]

    def give_loop_list_for_giving_gap(self):
        """
        Give the loop list for giving gap

        Parameters
        ----------
        start : int
            start of the loop
        end : int
            end of the loop
        gap : int
            gap

        Returns
        -------
        loop_list : list
            loop list
        """
        range = self.end - self.start
        loop_num = range / self.gap

        var1 = (
            self.start
            * (loop_num - 1)
            / ((self.end - self.gap) - self.start)
        )
        coefficient = self.start / var1
        var2 = self.gap / coefficient + var1

        return var1, var2, coefficient, loop_num

    def Filter_data_fit_PC1_gap_new(
        self,
        Cld_data,
        PC_data,
        IWP_data,
        IWP_min=0,
        IWP_max=float("inf"),
    ):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data, shape (PC_gap, lat, lon)
        PC_data : numpy.array
            PC data, shape (PC_gap, lat, lon)
        IWP_data : numpy.array
            IWP data, shape (PC_gap, lat, lon)
        IWP_min : float
            Minimum IWP value for the constrain, default is 0
        IWP_max : float
            Maximum IWP value for the constrain, default is infinity
        # ... (other parameters)

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """

        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num, " ******")
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
            " ******",
        )
        PC_gap = np.arange(int(gap_num))

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )

        for gap_num in PC_gap:
            pc_min = coef * (gap_num + var1)
            pc_max = coef * (gap_num + var2)
            mask = (
                (PC_data >= pc_min)
                & (PC_data < pc_max)
                & (IWP_data >= IWP_min)
                & (IWP_data <= IWP_max)
            )
            Cld_match_PC_gap[gap_num, :, :] = np.nanmean(
                np.where(mask, Cld_data, np.nan), axis=0
            )
            PC_match_PC_gap[gap_num, :, :] = np.nanmean(
                np.where(mask, PC_data, np.nan), axis=0
            )

        return Cld_match_PC_gap, PC_match_PC_gap

    def Filter_data_fit_PC1_gap_new_median(self, Cld_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data, shape (PC_gap, lat, lon)
        PC_data : numpy.array
            PC data, shape (PC_gap, lat, lon)
        start : int
            min value pf PC, like -1
        end : int
            max value of PC, like 2
        gap : int
            Giving gap of PC, like 0.2

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """

        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num, " ******")
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
            " ******",
        )
        PC_gap = np.arange(int(gap_num))

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )

        for gap_num in PC_gap:
            pc_min = coef * (gap_num + var1)
            pc_max = coef * (gap_num + var2)
            mask = (PC_data >= pc_min) & (PC_data < pc_max)
            Cld_match_PC_gap[gap_num, :, :] = np.nanmedian(
                np.where(mask, Cld_data, np.nan), axis=0
            )
            PC_match_PC_gap[gap_num, :, :] = np.nanmedian(
                np.where(mask, PC_data, np.nan), axis=0
            )

        return Cld_match_PC_gap, PC_match_PC_gap

    def Convert_pandas(self, Cld_match_PC_gap):
        gap_num = Cld_match_PC_gap.shape[0]
        Box = np.zeros(
            (
                Cld_match_PC_gap.shape[1]
                * Cld_match_PC_gap.shape[2],
                gap_num,
            )
        )
        # Box = np.zeros((gap_num, 64800, ))

        for i in range(gap_num):
            Box[:, i] = Cld_match_PC_gap[i, :, :].reshape(-1)

        Box = pd.DataFrame(Box)

        # Number of PC1 interval is set in np.arange(start, end, interval gap)
        Box.columns = np.round(
            np.arange(self.start, self.end, self.gap), 3
        )

        return Box

    def plot_box_plot(self, Cld_match_PC_gap, savefig_str):
        """
        Plot boxplot of Cld data match each PC1 interval
        Main plot function
        """
        Box = self.Convert_pandas(Cld_match_PC_gap)

        plt.style.use("seaborn-v0_8-ticks")  # type: ignore
        plt.rc("font", family="Times New Roman")

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
        plt.ylabel(savefig_str, size=26, weight="bold")
        # plt.xticks((np.round(np.arange(-1.5, 3.5, 0.5), 2)),fontsize=26, weight="bold", )
        plt.xticks(rotation=45)
        plt.yticks(
            fontsize=26,
            weight="bold",
        )
        os.makedirs("Box_plot", exist_ok=True)
        plt.savefig(
            "Box_plot/Box_plot_PC1_" + savefig_str + ".png",
            dpi=500,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.show()


# class Filter_data_fit_PC1_gap_IWP_AOD_constrain(object):
#     def __init__(self, Cld_data):
#         self.latitude = [i for i in range(0, Cld_data.shape[1], 1)]
#         self.longitude = [i for i in range(0, Cld_data.shape[2], 1)]

#     def Filter_data_fit_gap(
#         self,
#         Cld_data,
#         PC_data,
#         IWP_data,
#         AOD_data,
#         PC_gap,
#         IWP_gap,
#         AOD_gap,
#     ):
#         """
#         Filter data with given PC1, IWP, and AOD constraints

#         Parameters
#         ----------
#         Cld_data : numpy.array
#             Cloud data, shape (PC_gap, lat, lon)
#         PC_data : numpy.array
#             PC data, shape (PC_gap, lat, lon)
#         IWP_data : numpy.array
#             IWP data, shape (PC_gap, lat, lon)
#         AOD_data : numpy.array
#             AOD data, shape (PC_gap, lat, lon)
#         PC_gap : numpy.array
#             PC gap array, shape (n,)
#         IWP_gap : numpy.array
#             IWP gap array, shape (n,)
#         AOD_gap : numpy.array
#             AOD gap array, shape (n,)

#         Returns
#         -------
#         Cld_data_fit : numpy.array
#             Filtered data, Cld data for each PC gap
#             array(AOD_bin, IWP_bin, PC_bin, lat, lon)
#         """

#         pc1_gap_count = len(PC_gap) - 1
#         iwp_gap_count = len(IWP_gap) - 1
#         aod_gap_count = len(AOD_gap) - 1

#         Cld_match_PC_gap = np.zeros(
#             (aod_gap_count, iwp_gap_count, pc1_gap_count, len(self.latitude), len(self.longitude))
#         )
#         PC_match_PC_gap = np.zeros(
#             (aod_gap_count, iwp_gap_count, pc1_gap_count, len(self.latitude), len(self.longitude))
#         )

#         for aod_num in range(aod_gap_count):
#             aod_min = AOD_gap[aod_num]
#             aod_max = AOD_gap[aod_num + 1]
#             for iwp_num in range(iwp_gap_count):
#                 iwp_min = IWP_gap[iwp_num]
#                 iwp_max = IWP_gap[iwp_num + 1]
#                 for pc_num in range(pc1_gap_count):
#                     pc_min = PC_gap[pc_num]
#                     pc_max = PC_gap[pc_num + 1]
#                     mask = (
#                         (PC_data >= pc_min)
#                         & (PC_data < pc_max)
#                         & (IWP_data >= iwp_min)
#                         & (IWP_data < iwp_max)
#                         & (AOD_data >= aod_min)
#                         & (AOD_data < aod_max)
#                     )
#                     Cld_match_PC_gap[aod_num, iwp_num, pc_num, :, :] = np.nanmean(
#                         np.where(mask, Cld_data, np.nan), axis=0
#                     )
#                     PC_match_PC_gap[aod_num, iwp_num, pc_num, :, :] = np.nanmean(
#                         np.where(mask, PC_data, np.nan), axis=0
#                     )

#         return Cld_match_PC_gap, PC_match_PC_gap


import concurrent.futures


class Filter_data_fit_PC1_gap_IWP_AOD_constrain(object):
    def __init__(self, lat, lon):
        self.latitude = lat
        self.longitude = lon

    def process_data(
        self,
        aod_num,
        iwp_num,
        pc_num,
        PC_data,
        IWP_data,
        AOD_data,
        Cld_data,
        PC_gap,
        IWP_gap,
        AOD_gap,
    ):
        aod_min = AOD_gap[aod_num]
        aod_max = AOD_gap[aod_num + 1]
        iwp_min = IWP_gap[iwp_num]
        iwp_max = IWP_gap[iwp_num + 1]
        pc_min = PC_gap[pc_num]
        pc_max = PC_gap[pc_num + 1]
        mask = (
            (PC_data >= pc_min)
            & (PC_data < pc_max)
            & (IWP_data >= iwp_min)
            & (IWP_data < iwp_max)
            & (AOD_data >= aod_min)
            & (AOD_data < aod_max)
        )
        Cld_match_PC_gap = np.nanmean(
            np.where(mask, Cld_data, np.nan), axis=0
        )
        PC_match_PC_gap = np.nanmean(
            np.where(mask, PC_data, np.nan), axis=0
        )

        return (
            aod_num,
            iwp_num,
            pc_num,
            Cld_match_PC_gap,
            PC_match_PC_gap,
        )

    def Filter_data_fit_gap(
        self,
        Cld_data,
        PC_data,
        IWP_data,
        AOD_data,
        PC_gap,
        IWP_gap,
        AOD_gap,
    ):
        pc1_gap_count = len(PC_gap) - 1
        iwp_gap_count = len(IWP_gap) - 1
        aod_gap_count = len(AOD_gap) - 1

        Cld_match_PC_gap = np.empty(
            (
                aod_gap_count,
                iwp_gap_count,
                pc1_gap_count,
                len(self.latitude),
                len(self.longitude),
            )
        )
        PC_match_PC_gap = np.empty(
            (
                aod_gap_count,
                iwp_gap_count,
                pc1_gap_count,
                len(self.latitude),
                len(self.longitude),
            )
        )
        
        # convert data type to float32 to save memory
        Cld_data = Cld_data.astype(np.float32)
        PC_data = PC_data.astype(np.float32)
        IWP_data = IWP_data.astype(np.float32)
        AOD_data = AOD_data.astype(np.float32)
        
        print("Start to process data with multi-threading")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=160
        ) as executor:
            futures = [
                executor.submit(
                    self.process_data,
                    aod_num,
                    iwp_num,
                    pc_num,
                    PC_data,
                    IWP_data,
                    AOD_data,
                    Cld_data,
                    PC_gap,
                    IWP_gap,
                    AOD_gap,
                )
                for aod_num in range(aod_gap_count)
                for iwp_num in range(iwp_gap_count)
                for pc_num in range(pc1_gap_count)
            ]

            for future in concurrent.futures.as_completed(futures):
                (
                    aod_num,
                    iwp_num,
                    pc_num,
                    cld_result,
                    pc_result,
                ) = future.result()
                Cld_match_PC_gap[
                    aod_num, iwp_num, pc_num, :, :
                ] = cld_result
                PC_match_PC_gap[
                    aod_num, iwp_num, pc_num, :, :
                ] = pc_result

        return Cld_match_PC_gap, PC_match_PC_gap


# Divide data into n parts based on the PC1 data volume
class DividePCByDataVolume:
    def __init__(self, dataarray_main, n):
        self.dataarray_main = dataarray_main
        self.n = n

    # divide the data into 5 parts based on the PC1 data volume
    def main_gap(self):
        """
        Get the gap between each main piece

        Parameters:
            dataarray_main: a 2D dataarray of the main pieces
            n: the number of main pieces

        Returns:
            main_gap_num: the gap between each main piece
        """
        # Flatten the data.
        dataarray_main = self.dataarray_main.flatten()

        # mask = (self.dataarray_main >= -2.5) & (
        #     self.dataarray_main <= 5.5
        # )

        # self.dataarray_main = np.where(
        #     mask, self.dataarray_main, np.nan
        # )

        # Get the gap between each main piece.
        main_gap_num = np.empty((self.n + 1))

        for i in range(self.n + 1):
            print(
                "Current main gap number: ",
                i,
                " / ",
                self.n + 1,
                "\n",
            )
            main_gap_num[i] = np.nanpercentile(
                dataarray_main, i * 100 / self.n
            )
        return main_gap_num


# Filter data with gap, start and end with giving gap
# This is for atmos para filter
class FilterAtmosDataFitPCgap(Filter_data_fit_PC1_gap_plot):
    def Filter_data_fit_PC1_gap_atmos(self, Atmos_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Atmos_data : numpy.array
            Atmos data
        PC_data : numpy.array
            PC data
        start : int
            Start PC value, like -1
        end : int
            End PC value, like 2
        gap : int
            Giving gap, like 0.2

        Returns
        -------
        Atmos_data_fit : numpy.array
            Filtered data, Atmos data for each PC gap
            array(PC_gap, lat, lon)
        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num, " ******")
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
            " ******",
        )
        PC_gap = [i for i in range(0, int(gap_num), 1)]

        Atmos_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )

        for lat in self.latitude:
            for lon in self.longitude:
                for gap_num in PC_gap:
                    # Filter Cld data with gap, start and end with giving gap
                    Atmos_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        Atmos_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )

        return Atmos_match_PC_gap


#########################################################
########### Box Plot #######################################
########################################################


class Box_plot(object):
    """
    Plot boxplot of Cld data match each PC1 interval

    """

    def __init__(self, Cld_match_PC_gap, time_str):
        """
        Initialize the class

        Parameters
        ----------
        Cld_match_PC_gap : Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
        time_str : string
            time string like '2010to2019' or "2010to2019_4_6_month" or "2018only"
        """
        # Input array must be in shape of (PC1_gap, lat, lon)
        self.Cld_match_PC_gap = Cld_match_PC_gap
        self.time_str = time_str

    def Convert_pandas(self):
        gap_num = self.Cld_match_PC_gap.shape[0]
        Box = np.zeros(
            (
                self.Cld_match_PC_gap.shape[1]
                * self.Cld_match_PC_gap.shape[2],
                gap_num,
            )
        )
        # Box = np.zeros((gap_num, 64800, ))

        for i in range(gap_num):
            Box[:, i] = self.Cld_match_PC_gap[i, :, :].reshape(-1)

        Box = pd.DataFrame(Box)
        # Number of PC1 interval is set in np.arange(start, end, interval gap)
        Box.columns = np.round(np.arange(-1.5, 4.5, 0.05), 3)

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
        plt.savefig(
            "Box_plot_PC1_Cld_" + self.time_str + ".png",
            dpi=500,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.show()


# ---------- Comparing the difference between other years and 2020 -------------------
# ---------- In the giving PC1 gap, and the giving atmospheric gap -------------------


def compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020,
    Cld_all_match_PC_gap_others,
    start: float,
    end: float,
):
    """
    Loop over the given array to See if the data at each location is nan, if so,
    assign it to nan, if not, subtract the data at that location within two years
    """
    Cld_all_match_PC_gap_2020_sub_others = np.empty(
        (
            Cld_all_match_PC_gap_2020.shape[1],
            Cld_all_match_PC_gap_2020.shape[2],
        )
    )

    Cld_all_match_PC_gap_2020_sub_others[:, :] = np.nan

    for lat in range((Cld_all_match_PC_gap_2020.shape[1])):
        for lon in range((Cld_all_match_PC_gap_2020.shape[2])):
            for gap in range(start, end):
                # condition 1 : if both data are not nan, then subtract
                if (
                    np.isnan(
                        Cld_all_match_PC_gap_2020[gap, lat, lon]
                    )
                    == False
                ) and (
                    np.isnan(
                        Cld_all_match_PC_gap_others[gap, lat, lon]
                    )
                    == False
                ):
                    Cld_all_match_PC_gap_2020_sub_others[
                        lat, lon
                    ] = (
                        Cld_all_match_PC_gap_2020[gap, lat, lon]
                        - Cld_all_match_PC_gap_others[gap, lat, lon]
                    )
                    # exit the loop if the condition is met
                    break
                # condition 2 : if both data are zero, then assign nan
                # elif (
                #     Cld_all_match_PC_gap_2020[gap, lat, lon] == 0
                # ) and (
                #     Cld_all_match_PC_gap_others[gap, lat, lon] == 0
                # ):
                #     Cld_all_match_PC_gap_2020_sub_others[
                #         lat, lon
                #     ] = np.nan

                # condition 3 : if one of the data is nan, then assign nan
                else:
                    Cld_all_match_PC_gap_2020_sub_others[
                        lat, lon
                    ] = np.nan

    return Cld_all_match_PC_gap_2020_sub_others


def compare_cld_between_2020_others_percentile(
    Cld_all_match_PC_gap_2020: np.ndarray,
    Cld_all_match_PC_gap_others: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    rows, cols = Cld_all_match_PC_gap_2020.shape[1:]
    Cld_all_match_PC_gap_2020_sub_others = np.empty((rows, cols))
    Cld_all_match_PC_gap_2020_sub_others[:, :] = np.nan

    epsilon = 1e-8  # Small value to avoid division by zero

    for lat, lon in np.ndindex((rows, cols)):
        for gap in range(start, end):
            value_2020 = Cld_all_match_PC_gap_2020[gap, lat, lon]
            value_others = Cld_all_match_PC_gap_others[
                gap, lat, lon
            ]

            if (
                not np.isnan(value_2020)
                and not np.isnan(value_others)
                and abs(value_others) > epsilon
            ):
                percentage_difference = (
                    (value_2020 - value_others) / value_others
                ) * 100

                # Add checks for outliers, e.g., with a threshold of 200%
                if abs(percentage_difference) < 200:
                    Cld_all_match_PC_gap_2020_sub_others[
                        lat, lon
                    ] = percentage_difference
                else:
                    Cld_all_match_PC_gap_2020_sub_others[
                        lat, lon
                    ] = np.nan

                break

    return Cld_all_match_PC_gap_2020_sub_others


# -------------- Calculate the coefficient of variation (CV) ---------------
# -------------- of the given array ---------------
# -------------- and the Index of dispersion ---------------


def Calculate_coefficient_of_variation(input_array):
    """
    Calculated input array's coefficient of variation

    Parameters
    ----------
    input_array : array
        an array of numnber waiting to be calculated

    Returns
    -------
    float
        the CV of input array in %
    """
    # reshape the array to 1D
    array = np.array(input_array.reshape(-1))

    return (np.nanstd(array) / np.nanmean(array)) * 100


def Calculate_index_of_dispersion(input_array):
    """
    Calculated input array's index of dispersion

    Parameters
    ----------
    input_array : array
        an array of numnber waiting to be calculated

    Returns
    -------
    float
        the Index of dispersion of input array in %
    """
    # reshape the array to 1D
    array = np.array(input_array.reshape(-1))

    return ((np.nanstd(array) ** 2) / np.nanmean(array)) * 100


# ------------- Save calculated data as netcdf file ----------------------------------------------------


def save_PCA_data_as_netcdf(PC_filtered, Cld_filtered):
    """
    Save the PCA and Cld data as netcdf file

    Parameters
    ----------
    PC_filtered : array
        the filtered PCA data
    Cld_filtered : array
        the filtered Cld data
    """
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    PC_all = PC_all.reshape(132, 28, 180, 360)  # PCA

    ds = xr.Dataset(
        {
            "PC1": (
                ("Gap", "Latitude", "Longitude"),
                PC_filtered[:, :, :],
            ),
            "CLD": (
                ("Gap", "Latitude", "Longitude"),
                Cld_filtered[:, :, :],
            ),
        },
        coords={
            "gap_num": ("Gap", np.linspace(-90, 89, 180)),
            "lat": ("Latitude", np.linspace(-90, 89, 180)),
            "lon": ("Longitude", np.linspace(0, 359, 360)),
        },
    )

    os.makedirs("/RAID01/data/PCA_data/", exist_ok=True)
    ds.to_netcdf(
        "/RAID01/data/2010_2020_5_parameters_300hPa_PC1.nc"
    )
