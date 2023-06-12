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
    Date: 2023-05-17
    
    In this code we will use monthly or 2-monthly data to test the contrail signal
        
"""

# import modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)

# --------- import done ------------
# --------- Plot style -------------
mpl.rc("font", family="Times New Roman")
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")


#########################################
##### start seperate time test ##########
#########################################

# Implementation for MERRA2 dust AOD
# extract the data from 2010 to 2014 like above
data_merra2_2010_2020_new_lon = xr.open_dataset(
    "/RAID01/data/merra2/merra_2_daily_2010_2020_new_lon.nc"
)


# Extract the PC1 and Cld data for each year
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
    extracted_PC_data = {}
    extracted_Cld_data = {}
    extracted_IWP_data = {}

    for year in range(2017, 2021):
        extracted_PC_data[year] = PC_years[year].reshape(-1, 40, 360)

    for year in range(2017, 2021):
        extracted_Cld_data[year] = Cld_years[year].reshape(-1, 40, 360)

    for year in range(2017, 2021):
        extracted_IWP_data[year] = IWP_years[year].reshape(-1, 40, 360)

    return extracted_PC_data, extracted_Cld_data, extracted_IWP_data


def generate_filtered_data_for_all_years(
    AOD_data: np.ndarray,
    IWP_data: np.ndarray,
    PC_all: np.ndarray,
    Cld_data_all: np.ndarray,
    PC_gap: np.ndarray,
    IWP_gap: np.ndarray,
    AOD_gap: np.ndarray,
):
    filter_cld_under_AOD_IWP_PC_constrain = (
        Filter_data_fit_PC1_gap_IWP_AOD_constrain(
            lat=[i for i in range(40)],
            lon=[i for i in range(360)],
        )
    )

    # Now we can filter the CLd and PC1 data into pieces
    # Based on AOD, IWP, PC1 gap we just created
    # Shape is (AOD_bin, IWP_bin, PC_bin, lat, lon)
    (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        PC_match_PC_gap_IWP_AOD_constrain_mean,
    ) = filter_cld_under_AOD_IWP_PC_constrain.Filter_data_fit_gap(
        Cld_data=Cld_data_all.reshape(-1, 40, 360),
        PC_data=PC_all.reshape(-1, 40, 360),
        IWP_data=IWP_data.reshape(-1, 40, 360),
        AOD_data=AOD_data.reshape(-1, 40, 360),
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
        AOD_gap=AOD_gap,
    )

    return (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        PC_match_PC_gap_IWP_AOD_constrain_mean,
    )


def save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean: np.ndarray,
    AOD_gap: np.ndarray,
    IWP_gap: np.ndarray,
    PC_gap: np.ndarray,
    save_path: str = "/RAID01/data/Filtered_data/",
    save_str: str = "Filtered_data",
):
    # Save the fitted data as netcdf file
    Cld_match_PC_gap_IWP_AOD_constrain_mean = xr.DataArray(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        dims=["AOD_bin", "IWP_bin", "PC_bin", "lat", "lon"],
        coords={
            "AOD_bin": AOD_gap[1:],
            "IWP_bin": IWP_gap[1:],
            "PC_bin": PC_gap[1:],
            "lat": np.arange(40),
            "lon": np.arange(360),
        },
    )

    Cld_match_PC_gap_IWP_AOD_constrain_mean.to_netcdf(
        save_path
        + "Cld_match_PC_gap_IWP_AOD_constrain_mean_"
        + save_str
        + ".nc"
    )


def read_filtered_data_out(
    file_name: str = "Cld_match_PC_gap_IWP_AOD_constrain_mean_2010_2020.nc",
):
    Cld_match_PC_gap_IWP_AOD_constrain_mean = xr.open_dataarray(
        "/RAID01/data/Filtered_data/" + file_name
    )

    return Cld_match_PC_gap_IWP_AOD_constrain_mean


def filter_extreme_2_5_percent(IWP_data, AOD_data, PC_data, Cld_data):
    # Calculate the threshold values for the largest and smallest 2.5% of AOD and IWP data
    lower_threshold_IWP = np.nanpercentile(IWP_data, 2.5)
    upper_threshold_IWP = np.nanpercentile(IWP_data, 97.5)
    lower_threshold_AOD = np.nanpercentile(AOD_data, 2.5)
    upper_threshold_AOD = np.nanpercentile(AOD_data, 97.5)

    # Create a mask for extreme values in AOD and IWP data
    extreme_mask = (
        (IWP_data < lower_threshold_IWP)
        | (IWP_data > upper_threshold_IWP)
        | (AOD_data < lower_threshold_AOD)
        | (AOD_data > upper_threshold_AOD)
    )

    # Apply the mask to IWP, AOD, PC, and CLD data
    IWP_filtered = np.where(extreme_mask, np.nan, IWP_data)
    AOD_filtered = np.where(extreme_mask, np.nan, AOD_data)
    PC_filtered = np.where(extreme_mask, np.nan, PC_data)
    Cld_filtered = np.where(extreme_mask, np.nan, Cld_data)

    return IWP_filtered, AOD_filtered, PC_filtered, Cld_filtered


def filter_extreme_5_percent(IWP_data, AOD_data, PC_data, Cld_data):
    # Calculate the threshold values for the largest and smallest 2.5% of AOD and IWP data
    lower_threshold_IWP = np.nanpercentile(IWP_data, 5)
    upper_threshold_IWP = np.nanpercentile(IWP_data, 95)
    lower_threshold_AOD = np.nanpercentile(AOD_data, 5)
    upper_threshold_AOD = np.nanpercentile(AOD_data, 95)

    # Create a mask for extreme values in AOD and IWP data
    extreme_mask = (
        (IWP_data < lower_threshold_IWP)
        | (IWP_data > upper_threshold_IWP)
        | (AOD_data < lower_threshold_AOD)
        | (AOD_data > upper_threshold_AOD)
    )

    # Apply the mask to IWP, AOD, PC, and CLD data
    IWP_filtered = np.where(extreme_mask, np.nan, IWP_data)
    AOD_filtered = np.where(extreme_mask, np.nan, AOD_data)
    PC_filtered = np.where(extreme_mask, np.nan, PC_data)
    Cld_filtered = np.where(extreme_mask, np.nan, Cld_data)

    return IWP_filtered, AOD_filtered, PC_filtered, Cld_filtered


def filter_extreme_5_percent_IPR(Cld_data):
    # Calculate the threshold values for the largest and smallest 2.5% of AOD and IWP data
    lower_threshold_IPR = np.nanpercentile(Cld_data, 5)
    upper_threshold_IPR = np.nanpercentile(Cld_data, 95)

    # Create a mask for extreme values in AOD and IWP data
    extreme_mask = (Cld_data < lower_threshold_IPR) | (
        Cld_data > upper_threshold_IPR
    )

    # Apply the mask to IWP, AOD, PC, and CLD data
    Cld_filtered = np.where(extreme_mask, np.nan, Cld_data)

    return Cld_filtered


def process_2020_and_2017_2019(
    data_merra2, cld_var_name: str = "Cldarea"
):
    import gc

    # ---------- Read PCA&CLD data from netcdf file --------
    # now we read IWP and other cld data (not IWP) from netcdf file
    # triout 1 : Use lattitudinal band from 20N to 50N
    # Read the data
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
    ) = read_PC1_CERES_20_60_lat_band_from_netcdf(
        CERES_Cld_dataset_name=cld_var_name
    )

    # Use the extracted data for further processing and reshape the arrays
    PC_data = {year: PC_years[year] for year in range(2017, 2021)}
    Cld_data = {year: Cld_years[year] for year in range(2017, 2021)}
    IWP_data = {year: IWP_years[year] for year in range(2017, 2021)}

    # Delete the original data to save memory
    del PC_years, Cld_years, IWP_years
    gc.collect()

    # For each year of data in the dictionary
    for year in PC_data:
        # Extract the data for March and April
        PC_data[year] = PC_data[year][2:6, :, :, :]
        Cld_data[year] = Cld_data[year][2:6, :, :, :]
        IWP_data[year] = IWP_data[year][2:6, :, :, :]

    # Access the reshaped data using the year as a key
    PC_2017, PC_2018, PC_2019, PC_2020 = (
        PC_data[2017],
        PC_data[2018],
        PC_data[2019],
        PC_data[2020],
    )
    Cld_2017, Cld_2018, Cld_2019, Cld_2020 = (
        Cld_data[2017],
        Cld_data[2018],
        Cld_data[2019],
        Cld_data[2020],
    )
    IWP_2017, IWP_2018, IWP_2019, IWP_2020 = (
        IWP_data[2017],
        IWP_data[2018],
        IWP_data[2019],
        IWP_data[2020],
    )

    # Concatenate the data for PC as 2017-2019
    PC_2017_2019 = np.concatenate([PC_2017, PC_2018, PC_2019], axis=0)
    # Do so for Cldarea data
    Cld_2017_2019 = np.concatenate(
        [Cld_2017, Cld_2018, Cld_2019], axis=0
    )
    # Do so for IWP data
    IWP_2017_2019 = np.concatenate(
        [IWP_2017, IWP_2018, IWP_2019], axis=0
    )

    # Extract Dust aerosol data for 2020 and 2017-2019
    Dust_AOD_2020 = (
        data_merra2["DUEXTTAU"]
        .sel(lat=slice(21, 60))
        .sel(time=slice("2020", "2020"))
        .values
    )
    Dust_AOD_2017_2019 = (
        data_merra2["DUEXTTAU"]
        .sel(lat=slice(21, 60))
        .sel(time=slice("2017", "2019"))
        .values
    )

    # For the Dust AOD data
    Dust_AOD_2020 = Dust_AOD_2020.reshape(12, 28, 40, 360)[2:6, :, :]
    Dust_AOD_2017_2019 = Dust_AOD_2017_2019.reshape(
        3, 12, 28, 40, 360
    )[:, 2:6, :, :].reshape(12, 28, 40, 360)

    # Data read finished
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # filter the extreme 2.5% data for IWP and AOD only

    Cld_temp = np.concatenate([Cld_2017_2019, Cld_2020], axis=0)
    IWP_temp = np.concatenate([IWP_2017_2019, IWP_2020], axis=0)
    PC_temp = np.concatenate([PC_2017_2019, PC_2020], axis=0)
    AOD_temp = np.concatenate(
        [Dust_AOD_2017_2019, Dust_AOD_2020], axis=0
    )

    # Assuming IWP_data, Dust_AOD, PC_all, and Cld_all are your input arrays with shape (3686, 180, 360)
    (
        IWP_temp_filtered,
        AOD_temp_filtered,
        PC_temp_filtered,
        Cld_temp_filtered,
    ) = filter_extreme_5_percent(IWP_temp, AOD_temp, PC_temp, Cld_temp)

    if cld_var_name == "Cldicerad":
        Cld_temp_filtered = filter_extreme_5_percent_IPR(
            Cld_temp_filtered
        )

    # now the data are reshaped into (year, month(march, april), lat, lon)
    IWP_temp_filtered = IWP_temp_filtered.reshape(4, 4, 28, 40, 360)
    AOD_temp_filtered = AOD_temp_filtered.reshape(4, 4, 28, 40, 360)
    PC_temp_filtered = PC_temp_filtered.reshape(4, 4, 28, 40, 360)
    Cld_temp_filtered = Cld_temp_filtered.reshape(4, 4, 28, 40, 360)

    #### triout for IWP constrain the same time with PC1 gap constrain ####

    # We try to set an universal PC, AOD, IWP gap for all years
    # This is trying to hit all data with the same constrain

    # first we need to divide IWP data and PC1 data into n intervals
    # this step is aimed to create pcolormesh plot for PC1 and IWP data
    # Divide 1, IWP data
    # filter the extreme 5% data
    Cld_2017_2019 = Cld_temp_filtered[:3].reshape(-1, 40, 360)
    Cld_2020 = Cld_temp_filtered[-1].reshape(-1, 40, 360)

    # Divide 1, IWP data
    IWP_2017_2019 = IWP_temp_filtered[:3].reshape(-1, 40, 360)
    IWP_2020 = IWP_temp_filtered[-1].reshape(-1, 40, 360)

    divide_IWP = DividePCByDataVolume(
        dataarray_main=IWP_temp_filtered,
        n=30,
    )
    IWP_gap = divide_IWP.main_gap()

    # Divide 2, PC1 data
    PC_2017_2019 = PC_temp_filtered[:3].reshape(-1, 40, 360)
    PC_2020 = PC_temp_filtered[-1].reshape(-1, 40, 360)

    divide_PC = DividePCByDataVolume(
        dataarray_main=PC_temp_filtered,
        n=30,
    )
    PC_gap = divide_PC.main_gap()

    # Divide 3, Dust AOD data
    # Divide AOD data as well
    AOD_2017_2019 = AOD_temp_filtered[:3].reshape(-1, 40, 360)
    AOD_2020 = AOD_temp_filtered[-1].reshape(-1, 40, 360)

    divide_AOD = DividePCByDataVolume(
        dataarray_main=AOD_temp_filtered,
        n=6,
    )
    AOD_gap = divide_AOD.main_gap()

    # Load the data
    (
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
        PC_match_PC_gap_IWP_AOD_constrain_mean_2020,
    ) = generate_filtered_data_for_all_years(
        AOD_data=AOD_2020,
        IWP_data=IWP_2020,
        PC_all=PC_2020,
        Cld_data_all=Cld_2020,
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
        AOD_gap=AOD_gap,
    )

    # Save the filtered data
    save_filtered_data_as_nc(
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
        AOD_gap,
        IWP_gap,
        PC_gap,
        save_str="2020_filter_extreme_AOD_IWP_IPR_" + cld_var_name,
    )

    (
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
        PC_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    ) = generate_filtered_data_for_all_years(
        AOD_data=AOD_2017_2019,
        IWP_data=IWP_2017_2019,
        PC_all=PC_2017_2019,
        Cld_data_all=Cld_2017_2019,
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
        AOD_gap=AOD_gap,
    )

    save_filtered_data_as_nc(
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
        AOD_gap,
        IWP_gap,
        PC_gap,
        save_str="2017_2019_filter_extreme_AOD_IWP_IPR_"
        + cld_var_name,
    )

    # Read the filtered data
    # Read the Dust_AOD constrain data for 2020 and 2017-2019 data
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020 = read_filtered_data_out(
        file_name="Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_filter_extreme_AOD_IWP_IPR_"
        + cld_var_name
        + ".nc"
    )

    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019 = read_filtered_data_out(
        file_name="Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_filter_extreme_AOD_IWP_IPR_"
        + cld_var_name
        + ".nc"
    )

    return (
        AOD_gap,
        IWP_gap,
        PC_gap,
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    )


(
    AOD_gap,
    IWP_gap,
    PC_gap,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_HCF,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_HCF,
) = process_2020_and_2017_2019(
    data_merra2_2010_2020_new_lon, cld_var_name="Cldarea"
)

(
    AOD_gap,
    IWP_gap,
    PC_gap,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_IPR,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_IPR,
) = process_2020_and_2017_2019(
    data_merra2_2010_2020_new_lon, cld_var_name="Cldicerad"
)

(
    AOD_gap,
    IWP_gap,
    PC_gap,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_CEH,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_CEH,
) = process_2020_and_2017_2019(
    data_merra2_2010_2020_new_lon, cld_var_name="Cldeff_hgth"
)

# CERES_Cld_dataset = [
#     "Cldarea",
#     "Cldicerad",
#     "Cldeff_hgth",
#     "Cldpress_base",
#     "Cldhgth_top",
#     "Cldtau",
#     "Cldtau_lin",
#     "IWP",
#     "Cldemissir",
# ]

########################################################################
# ------------------------------------------------------------------
# Plot 4D plot to demenstrate the constrain of PC1, IWP, Dust AOD
# ------------------------------------------------------------------
########################################################################


# version 4
# Seperate the 3D plot into 2 subplots by AOD interval
# color nan values with self-defined color
def create_colormap_with_nan(cmap_name, nan_color="silver"):
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))
    cmap_with_nan = ListedColormap(colors)
    cmap_with_nan.set_bad(nan_color)
    return cmap_with_nan


def plot_3d_colored_IWP_PC1_AOD_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    high_cloud_amount_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    savefig_str,
    aod_range,  # Add this parameter to define the AOD range
    vmin,  # Add this parameter to define the minimum value for the color scale
    vmax,  # Add this parameter to define the maximum value for the color scale
    cmap,  # Add this parameter to define the custom colormap
):
    """
    Create a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval.

    Args:
        Cld_match_PC_gap_IWP_AOD_constrain_mean (numpy.array): Cloud data under each restriction, shape (AOD_bin, IWP_bin, PC_bin, lat, lon)
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        savefig_str (str): String for saving the figure
        aod_range (tuple): Tuple defining the start and end indices of the AOD cases to plot
        vmin (float): Minimum value for the color scale
        vmax (float): Maximum value for the color scale
    """

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    # Add this line after the function's docstring to create a colormap that handles NaN values
    cmap_with_nan = create_colormap_with_nan(cmap)

    # Iterate over the specified AOD range
    for aod_num in range(aod_range[0], aod_range[1]):
        Z = aod_num * np.ones_like(X)

        # Plot the 2D pcolormesh color fill map for the current AOD interval
        ax.plot_surface(
            Z,
            X,
            Y,
            rstride=1,
            cstride=1,
            facecolors=cmap_with_nan(
                (high_cloud_amount_mean[aod_num] - vmin)
                / (vmax - vmin)
            ),
            shade=False,
            edgecolor="none",
            alpha=0.95,
            antialiased=False,
            linewidth=0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # Turn off the grid lines
    ax.grid(False)

    # Add color bar
    m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r)
    m.set_cmap(cmap_with_nan)
    m.set_array(high_cloud_amount_mean)
    m.set_clim(vmin, vmax)
    fig.colorbar(
        m, shrink=0.3, aspect=9, pad=0.01, label=colobar_label
    )

    ax.view_init(elev=12, azim=-65)
    ax.dist = 12

    # Save the figure
    plt.savefig(savefig_str)
    plt.show()


def plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    vmin,  # Add this parameter to define the minimum value for the color scale
    vmax,  # Add this parameter to define the maximum value for the color scale
    cmap="Spectral_r",
):
    """
    This function is basiclly used to call the above main plot function
    This function plots a 3D fill plot with the following parameters:
    Cld_match_PC_gap_IWP_AOD_constrain_mean: 3D array of IWP, AOD, and PC1 values
    high_cloud_amount_mean: 2D array of cloud amount values
    xlabel: label for x-axis
    ylabel: label for y-axis
    zlabel: label for z-axis
    savefig_str: string to name the saved figure with
    aod_range: range of AOD values to plot
    """
    # Calculate the mean high cloud amount for each AOD interval, IWP, and PC1 bin
    high_cloud_amount_mean = np.nanmean(
        Cld_match_PC_gap_IWP_AOD_constrain_mean, axis=(3, 4)
    )
    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        "subplot_1.png",
        (0, 3),
        vmin,  # Add this parameter to define the minimum value for the color scale
        vmax,  # Add this parameter to define the maximum value for the color scale
        cmap=cmap,
    )
    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        "subplot_2.png",
        (3, 6),
        vmin,  # Add this parameter to define the minimum value for the color scale
        vmax,  # Add this parameter to define the maximum value for the color scale
        cmap=cmap,
    )


# Call the function with different AOD ranges and save each figure separately
# Plot the dust-AOD constrained hcf data
# 2020 filtered data
def plot_data(data, title, vmin, vmax):
    plot_both_3d_fill_plot_min_max_version(
        data,
        "Dust-AOD",
        "PC1",
        "IWP",
        title,
        vmin=vmin,
        vmax=vmax,
    )


plot_config = [
    {
        "data": Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_HCF,
        "title": "HCF (%)",
        "vmin": 0,
        "vmax": 35,
    },
    {
        "data": Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_HCF,
        "title": "HCF (%)",
        "vmin": 0,
        "vmax": 35,
    },
    {
        "data": Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_IPR,
        "title": "IPR (micron)",
        "vmin": 23,
        "vmax": 31.5,
    },
    {
        "data": Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_IPR,
        "title": "IPR (micron)",
        "vmin": 23,
        "vmax": 31.5,
    },
    {
        "data": Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_CEH,
        "title": "CEH (km)",
        "vmin": 10,
        "vmax": 14.2,
    },
    {
        "data": Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_CEH,
        "title": "CEH (km)",
        "vmin": 10,
        "vmax": 14.2,
    },
]


for config in plot_config:
    plot_data(
        config["data"], config["title"], config["vmin"], config["vmax"]
    )


# ------------------------------------------------------------------------------------------
def calculate_abnormal(data_2020, data_2017_2019):
    return data_2020 - data_2017_2019


def plot_abnormal_data(data, title, vmin, vmax, cmap="RdBu_r"):
    plot_both_3d_fill_plot_min_max_version(
        data,
        "Dust-AOD",
        "PC1",
        "IWP",
        title,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )


abnormal_data = [
    {
        "data": calculate_abnormal(
            Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_HCF,
            Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_HCF,
        ),
        "title": "HCF (%)",
        "vmin": -12,
        "vmax": 12,
    },
    {
        "data": calculate_abnormal(
            Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_IPR,
            Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_IPR,
        ),
        "title": "IPR (micron)",
        "vmin": -8,
        "vmax": 8,
    },
    {
        "data": calculate_abnormal(
            Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_CEH,
            Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_CEH,
        ),
        "title": "CEH (km)",
        "vmin": -2.3,
        "vmax": 2.3,
    },
]

# Output abnormal arrays
Cld_match_PC_gap_IWP_AOD_constrain_mean_HCF_abnormal = abnormal_data[
    0
]["data"]
Cld_match_PC_gap_IWP_AOD_constrain_mean_IPR_abnormal = abnormal_data[
    1
]["data"]
Cld_match_PC_gap_IWP_AOD_constrain_mean_CEH_abnormal = abnormal_data[
    2
]["data"]

for data in abnormal_data:
    plot_abnormal_data(
        data["data"], data["title"], data["vmin"], data["vmax"]
    )

# #######################################################################
# ###### We only analyze the april to july cld and pc1 data #############
# ###### In order to extract the contrail maximum signal ################
# #######################################################################

# Array shape in (AOD gap, IWP gap, PC1 gap, lat, lon)
# Test for each AOD gap!
# lets divide this shit into 3 IWP big gaps


def plot_global_spatial_distribution(
    data, var_name, title, vmin=None, vmax=None
):
    from matplotlib.colors import BoundaryNorm

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(20, 59, 40)

    from matplotlib.colors import LinearSegmentedColormap

    # Create custom colormap
    cmap = plt.cm.get_cmap("RdBu_r")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 5),
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
        vmin=vmin,
        vmax=vmax,
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
        shrink=0.3,
        extend="both",
    )
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


# create a mask for a specific region defined by latitude and longitude ranges
def create_region_mask(latitudes, longitudes, lat_range, lon_range):
    """
    Create a mask for a specific region defined by latitude and longitude ranges.

    Args:
        latitudes (numpy.array): Array of latitudes.
        longitudes (numpy.array): Array of longitudes.
        lat_range (tuple): The latitude range.
        lon_range (tuple): The longitude range.

    Returns:
        numpy.array: The mask array.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)

    return lat_mask[:, None] & lon_mask


def compute_mean_HCF_abnormal_masked(AOD_range, IWP_range, data, mask):
    """
    Compute the mean of the masked data.

    Args:
        AOD_range (slice): The slice for AOD.
        IWP_range (slice): The slice for IWP.
        data (numpy.array): The data to be masked and computed.
        mask (numpy.array): The mask array.

    Returns:
        numpy.array: The computed mean of the masked data.
    """
    # Compute the mean
    data = np.nanmean(
        data[AOD_range, IWP_range, :, :, :], axis=(0, 1, 2)
    )

    # Apply the mask
    masked_data = np.where(mask, data, np.nan)

    return masked_data


def plot_abnormal_HCF_scenario(low, mid, high, var_name):
    plot_global_spatial_distribution(
        low,
        var_name,
        "2020 - (2017-2019) anormally (low IWP)",
        vmin=-5,
        vmax=5,
    )
    plot_global_spatial_distribution(
        mid,
        var_name,
        "2020 - (2017-2019) anormally (mid IWP)",
        vmin=-5,
        vmax=5,
    )
    plot_global_spatial_distribution(
        high,
        var_name,
        "2020 - (2017-2019) anormally (high IWP)",
        vmin=-10,
        vmax=10,
    )


# Define the slices for IWP and AOD
IWP_slices = [slice(0, 10), slice(10, 20), slice(20, 30)]
AOD_slices = [slice(0, 2), slice(2, 4), slice(4, 6)]


# Define the data name and variable name3
datas = [
    Cld_match_PC_gap_IWP_AOD_constrain_mean_HCF_abnormal,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_IPR_abnormal,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_CEH_abnormal,
]
var_names = ["HCF (%)", "IPR (micron)", "CEH (km)"]


# Define latitudes and longitudes
latitudes = np.linspace(20, 59, 40)  # adjust these to match your data
longitudes = np.linspace(
    0, 359, 360
)  # adjust these to match your data

# Create a meshgrid for latitudes and longitudes
lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)

# Create mask for the US region
US_mask = ~(
    (30 <= lat_grid)
    & (lat_grid <= 50)
    & (260 <= lon_grid)
    & (lon_grid <= 310)
)


# Plot all situation for each variable and each AOD gap, IWP gap
for idx, data in enumerate(datas):
    for AOD_slice in AOD_slices:
        low = compute_mean_HCF_abnormal_masked(AOD_slice, IWP_slices[0], data)
        mid = compute_mean_HCF_abnormal_masked(AOD_slice, IWP_slices[1], data)
        high = compute_mean_HCF_abnormal_masked(
            AOD_slice, IWP_slices[2], data
        )
        plot_abnormal_HCF_scenario(low, mid, high, var_names[idx])


# test for the IPR data
data = Cld_match_PC_gap_IWP_AOD_constrain_mean_IPR_abnormal


# ------------------------------------------------------------------------------------------
# define the region
US_region = ((30, 50), (260, 310))
# Define the two parts of the Euro region
Euro_region_1 = ((35, 60), (345, 360))
Euro_region_2 = ((35, 60), (0, 35))
China_region = ((15, 40), (105, 140))

# ------------------------------------------------------------------------------------------
# create a mask for each region
US_mask = create_region_mask(latitudes, longitudes, *US_region)
# Create the masks for each part of Euro
Euro_mask_1 = create_region_mask(latitudes, longitudes, *Euro_region_1)
Euro_mask_2 = create_region_mask(latitudes, longitudes, *Euro_region_2)

# Combine the two masks into a single Euro mask
Euro_mask = np.logical_or(Euro_mask_1, Euro_mask_2)
China_mask = create_region_mask(latitudes, longitudes, *China_region)
# ------------------------------------------------------------------------------------------

# Define the list of masks and corresponding region names
masks = [US_mask, Euro_mask, China_mask]
region_names = ["US", "Euro", "China"]

# The AOD cases
AOD_cases = ["Low", "Mid", "High"]
IWP_cases = ["Low", "Mid", "High"]

# The dictionary for storing the results
results = {}

# For each mask, compute the masked data for each data, AOD case, and IWP case
for mask, region_name in zip(masks, region_names):
    for idx, data in enumerate(datas):
        for AOD_idx, AOD_slice in enumerate(AOD_slices):
            for IWP_idx, IWP_slice in enumerate(IWP_slices):
                # Compute the masked data
                masked_data = compute_mean_HCF_abnormal_masked(
                    AOD_slice, IWP_slice, data, mask
                )

                # Store the result in the dictionary
                results[
                    (
                        region_name,
                        var_names[idx],
                        AOD_cases[AOD_idx],
                        IWP_cases[IWP_idx],
                    )
                ] = masked_data

# ------------------------------------------------------------------------------------------
# The results are stored in a dictionary with a tuple of 
# the region name, variable name, AOD case, and IWP case as the key

# You can access the data for a specific region, variable, AOD case, and IWP case
# specific_masked_data = results[("US", "HCF (%)", "Low", "Mid")]
# ------------------------------------------------------------------------------------------
