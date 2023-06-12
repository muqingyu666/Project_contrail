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
import matplotlib.ticker as ticker
import numpy as np
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)

# --------- import done ------------
# --------- Plot style -------------
# Set parameter to avoid warning
mpl.style.use("seaborn-v0_8-ticks")
mpl.rc("font", family="Times New Roman")

#########################################
##### start seperate time test ##########
#########################################

# Implementation for MERRA2 dust AOD
# extract the data from 2010 to 2014 like above
data_merra2_2010_2020_new_lon = xr.open_dataset(
    "/RAID01/data/merra2/merra_2_daily_2010_2020_new_lon.nc"
)


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

    return (Cld_match_PC_gap_IWP_AOD_constrain_mean,)


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


def process_each_year(data_merra2, cld_var_name: str = "Cldarea"):
    import gc

    import numpy as np

    (
        PC_all,
        PC_years,
        Cld_all,
        Cld_years,
        IWP_data,
        IWP_years,
    ) = read_PC1_CERES_20_60_lat_band_from_netcdf_2015_2020(
        CERES_Cld_dataset_name=cld_var_name
    )

    # Use the extracted data for further processing and reshape the arrays
    PC_data = {year: PC_years[year] for year in range(2015, 2021)}
    Cld_data = {year: Cld_years[year] for year in range(2015, 2021)}
    IWP_data = {year: IWP_years[year] for year in range(2015, 2021)}

    # Delete the original data to save memory
    del PC_years, Cld_years, IWP_years
    gc.collect()

    # For each year of data in the dictionary
    for year in PC_data:
        PC_data[year] = PC_data[year][2:7, :, :, :]
        Cld_data[year] = Cld_data[year][2:7, :, :, :]
        IWP_data[year] = IWP_data[year][2:7, :, :, :]

    # Create a dictionary to store the results
    results = {}

    # For each year of data in the dictionary
    for year in range(2015, 2021):
        # Get the year specific data
        PC_year = PC_data[year]
        Cld_year = Cld_data[year]
        IWP_year = IWP_data[year]

        # Extract Dust aerosol data for the year
        Dust_AOD_year = (
            data_merra2["DUEXTTAU"]
            .sel(lat=slice(21, 60))
            .sel(time=slice(str(year), str(year)))
            .values
        )

        # Reshape Dust AOD data
        Dust_AOD_year = Dust_AOD_year.reshape(12, 28, 40, 360)[
            2:7, :, :, :
        ]

        # Collect all data
        Cld_temp = Cld_year
        IWP_temp = IWP_year
        PC_temp = PC_year
        AOD_temp = Dust_AOD_year

        (
            IWP_temp_filtered,
            AOD_temp_filtered,
            PC_temp_filtered,
            Cld_temp_filtered,
        ) = filter_extreme_5_percent(
            IWP_temp, AOD_temp, PC_temp, Cld_temp
        )

        if cld_var_name == "Cldicerad":
            Cld_temp_filtered = filter_extreme_5_percent_IPR(
                Cld_temp_filtered
            )

        # Reshape filtered data
        IWP_temp_filtered = IWP_temp_filtered.reshape(5, 28, 40, 360)
        AOD_temp_filtered = AOD_temp_filtered.reshape(5, 28, 40, 360)
        PC_temp_filtered = PC_temp_filtered.reshape(5, 28, 40, 360)
        Cld_temp_filtered = Cld_temp_filtered.reshape(5, 28, 40, 360)

        # Divide all data by the data volume
        divide_IWP = DividePCByDataVolume(
            dataarray_main=IWP_temp_filtered,
            n=30,
        )
        IWP_gap = divide_IWP.main_gap()

        divide_PC = DividePCByDataVolume(
            dataarray_main=PC_temp_filtered, n=30
        )
        PC_gap = divide_PC.main_gap()

        divide_AOD = DividePCByDataVolume(
            dataarray_main=AOD_temp_filtered, n=6
        )
        AOD_gap = divide_AOD.main_gap()

        # Load the data
        (
            Cld_match_PC_gap_IWP_AOD_constrain_mean,
        ) = generate_filtered_data_for_all_years(
            AOD_data=AOD_temp_filtered,
            IWP_data=IWP_temp_filtered,
            PC_all=PC_temp_filtered,
            Cld_data_all=Cld_temp_filtered,
            PC_gap=PC_gap,
            IWP_gap=IWP_gap,
            AOD_gap=AOD_gap,
        )

        # Save the filtered data
        save_filtered_data_as_nc(
            Cld_match_PC_gap_IWP_AOD_constrain_mean,
            AOD_gap,
            IWP_gap,
            PC_gap,
            save_str=f"{year}_filter_extreme_AOD_IWP_IPR_"
            + cld_var_name,
        )

        results[year] = {
            "AOD_gap": AOD_gap,
            "IWP_gap": IWP_gap,
            "PC_gap": PC_gap,
            "Cld_data": Cld_match_PC_gap_IWP_AOD_constrain_mean,
        }

    return results


def process_each_total_years(
    data_merra2, cld_var_name: str = "Cldarea"
):
    import gc

    import numpy as np

    (
        PC_all,
        PC_years,
        Cld_all,
        Cld_years,
        IWP_data,
        IWP_years,
    ) = read_PC1_CERES_20_60_lat_band_from_netcdf_2015_2020(
        CERES_Cld_dataset_name=cld_var_name
    )

    # define the years
    years = [2015, 2016, 2017, 2018, 2019, 2020]

    # Create a dictionary to store the results
    results = {}

    # For each year in the list
    # Prepare the years to process
    process_years = years

    # Collect all the data for the years to process
    PC_data = {year: PC_years[year] for year in process_years}
    Cld_data = {year: Cld_years[year] for year in process_years}
    IWP_data = {year: IWP_years[year] for year in process_years}

    # For each year of data in the dictionary
    for year in PC_data:
        # Extract the data for March and April
        PC_data[year] = PC_data[year][2:7, :, :, :]
        Cld_data[year] = Cld_data[year][2:7, :, :, :]
        IWP_data[year] = IWP_data[year][2:7, :, :, :]

    # Process all years together
    PC_data_all_years = np.concatenate(
        [PC_data[year] for year in process_years], axis=0
    ).reshape(6, 5, 28, 40, 360)
    Cld_data_all_years = np.concatenate(
        [Cld_data[year] for year in process_years], axis=0
    ).reshape(6, 5, 28, 40, 360)
    IWP_data_all_years = np.concatenate(
        [IWP_data[year] for year in process_years], axis=0
    ).reshape(6, 5, 28, 40, 360)

    # Extract Dust aerosol data for the year
    # For each year to be processed
    Dust_AOD_all_years = []
    for year in process_years:
        # Extract Dust aerosol data for the year
        Dust_AOD_year = (
            data_merra2["DUEXTTAU"]
            .sel(lat=slice(21, 60))
            .sel(
                time=slice(str(year), str(year))
            )  # Slice for each year and convert to str for matching
            .values
        )

        Dust_AOD_all_years.append(Dust_AOD_year)

    # Concatenate all Dust_AOD_year data along time axis (assumed to be the first axis)
    Dust_AOD_all_years = np.concatenate(Dust_AOD_all_years, axis=0)

    # Reshape Dust AOD data
    Dust_AOD_all_years = Dust_AOD_all_years.reshape(
        6, 12, 28, 40, 360
    )[:, 2:7, :, :, :]

    # Collect all data
    Cld_temp = Cld_data_all_years
    IWP_temp = IWP_data_all_years
    PC_temp = PC_data_all_years
    AOD_temp = Dust_AOD_all_years

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

    # Reshape filtered data
    IWP_temp_filtered = IWP_temp_filtered.reshape(6, 5, 28, 40, 360)
    AOD_temp_filtered = AOD_temp_filtered.reshape(6, 5, 28, 40, 360)
    PC_temp_filtered = PC_temp_filtered.reshape(6, 5, 28, 40, 360)
    Cld_temp_filtered = Cld_temp_filtered.reshape(6, 5, 28, 40, 360)

    # Divide all data by the data volume
    divide_IWP = DividePCByDataVolume(
        dataarray_main=IWP_temp_filtered,
        n=30,
    )
    IWP_gap = divide_IWP.main_gap()

    divide_PC = DividePCByDataVolume(
        dataarray_main=PC_temp_filtered, n=30
    )
    PC_gap = divide_PC.main_gap()

    divide_AOD = DividePCByDataVolume(
        dataarray_main=AOD_temp_filtered, n=6
    )
    AOD_gap = divide_AOD.main_gap()

    # Load the data
    (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
    ) = generate_filtered_data_for_all_years(
        AOD_data=AOD_temp_filtered,
        IWP_data=IWP_temp_filtered,
        PC_all=PC_temp_filtered,
        Cld_data_all=Cld_temp_filtered,
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
        AOD_gap=AOD_gap,
    )

    # Save the filtered data
    save_filtered_data_as_nc(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        AOD_gap,
        IWP_gap,
        PC_gap,
        save_str="2015_2020_filter_extreme_AOD_IWP_IPR_"
        + cld_var_name,
    )

    results = {
        "AOD_gap": AOD_gap,
        "IWP_gap": IWP_gap,
        "PC_gap": PC_gap,
        "Cld_data": Cld_match_PC_gap_IWP_AOD_constrain_mean,
    }

    return results


# run the function
results_HCF = process_each_year(
    data_merra2_2010_2020_new_lon, cld_var_name="Cldarea"
)

results_IPR = process_each_year(
    data_merra2_2010_2020_new_lon, cld_var_name="Cldicerad"
)

results_CEH = process_each_year(
    data_merra2_2010_2020_new_lon, cld_var_name="Cldeff_hgth"
)

# run the function for leave one out
results_HCF_leave_out = process_each_total_years(
    data_merra2_2010_2020_new_lon, cld_var_name="Cldarea"
)

results_IPR_leave_out = process_each_total_years(
    data_merra2_2010_2020_new_lon, cld_var_name="Cldicerad"
)

results_CEH_leave_out = process_each_total_years(
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

# the output of the function:
# results[year] = {
#     'AOD_gap': AOD_gap,
#     'IWP_gap': IWP_gap,
#     'PC_gap': PC_gap,
#     'Cld_data': Cld_match_PC_gap_IWP_AOD_constrain_mean,
# }

########################################################################
# Calculate the anomaly of each year ###################################
########################################################################

results_HCF_anomaly = {}
results_IPR_anomaly = {}
results_CEH_anomaly = {}

# Loop over each year
for year in range(2015, 2021):
    results_HCF_anomaly[year] = {
        "Cld_data": results_HCF[year]["Cld_data"]
        - results_HCF_leave_out["Cld_data"]
    }

    results_IPR_anomaly[year] = {
        "Cld_data": results_IPR[year]["Cld_data"]
        - results_IPR_leave_out["Cld_data"]
    }

    results_CEH_anomaly[year] = {
        "Cld_data": results_CEH[year]["Cld_data"]
        - results_CEH_leave_out["Cld_data"]
    }


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


# plot settings
plot_vars = [
    {
        "results": results_HCF,
        "title": "HCF (%)",
        "vmin": 0,
        "vmax": 35,
    },
    {
        "results": results_IPR,
        "title": "IPR (micron)",
        "vmin": 23,
        "vmax": 31.5,
    },
    {
        "results": results_CEH,
        "title": "CEH (km)",
        "vmin": 10,
        "vmax": 14.2,
    },
]


plot_vars_leave_out = [
    {
        "results": results_HCF_leave_out,
        "title": "HCF (%)",
        "vmin": 0,
        "vmax": 35,
    },
    {
        "results": results_IPR_leave_out,
        "title": "IPR (micron)",
        "vmin": 23,
        "vmax": 31.5,
    },
    {
        "results": results_CEH_leave_out,
        "title": "CEH (km)",
        "vmin": 10,
        "vmax": 14.2,
    },
]


# Plot the data for each year
# Verification of the results mainly
# Make sure that 2016-2020 data are consistent
for plot_var in plot_vars:
    for year in range(2015, 2021):
        data = plot_var["results"][year]["Cld_data"]
        title = f"{plot_var['title']} {year}"
        plot_data(data, title, plot_var["vmin"], plot_var["vmax"])

# Plot the leave out data for each year
for plot_var in plot_vars_leave_out:
    data = plot_var["results"]["Cld_data"]
    title = f"{plot_var['title']}"
    plot_data(data, title, plot_var["vmin"], plot_var["vmax"])


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

# Define latitudes and longitudes
latitudes = np.linspace(20, 59, 40)  # adjust these to match your data
longitudes = np.linspace(
    0, 359, 360
)  # adjust these to match your data

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

# The dictionary for storing each results
results_dict = {
    "HCF (%)": results_HCF_anomaly,
    "IPR (micron)": results_IPR_anomaly,
    "CEH (km)": results_CEH_anomaly,
}

# The dictionary for storing the results
results_anomaly = {}

# For each mask, compute the masked data for each data, AOD case, and IWP case
for var_name, result in results_dict.items():
    # Loop over the masks and region names
    for mask, region_name in zip(masks, region_names):
        # Loop over the years
        for year in range(2015, 2021):
            # Get the data for the year
            data_year = result[year]["Cld_data"]

            # Loop over the AOD and IWP cases
            for AOD_idx, AOD_slice in enumerate(AOD_slices):
                for IWP_idx, IWP_slice in enumerate(IWP_slices):
                    # Compute the masked data
                    masked_data = compute_mean_HCF_abnormal_masked(
                        AOD_slice, IWP_slice, data_year, mask
                    )

                    # Optionally, you can plot the masked data to check
                    # plot_global_spatial_distribution(
                    #     masked_data,
                    #     var_name,
                    #     f"{year} - {region_name} - {AOD_cases[AOD_idx]} - {IWP_cases[IWP_idx]}",
                    #     vmin=-7,
                    #     vmax=7,
                    # )
                    # Store the result in the dictionary
                    results_anomaly[
                        (
                            year,
                            region_name,
                            var_name,
                            AOD_cases[AOD_idx],
                            IWP_cases[IWP_idx],
                        )
                    ] = masked_data
  
# ------------------------------------------------------------------------------------------
# The results are stored in a dictionary with a tuple of
# the region name, variable name, AOD case, and IWP case as the key

# You can access the data for a specific region, variable, AOD case, and IWP case
# specific_masked_data = results_anomaly[(2016, "US", "HCF (%)", "Low", "Mid")]
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# Var name : HCF (%), IPR (micron), CEH (km)
# AOD case : Low, Mid, High
# IWP case : Low, Mid, High
# ------------------------------------------------------------------------------------------


# Plot each IWP and AOD case for each region
# This function will plot 9 subplots, 3 cols are Low, Mid, High AOD cases
# 3 rows are Low, Mid, High IWP cases
# Each subplot contains 3 lines for US, Euro, China
def plot_annual_data(
    results_anomaly,
    var_name="HCF (%)",
    region_names=["US", "Euro", "China"],
    AOD_cases=["Low", "Mid", "High"],
    IWP_cases=["Low", "Mid", "High"],
):
    # Define the years
    years = range(2015, 2021)

    # Create a figure with 3x3 subplots
    fig, axs = plt.subplots(
        nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=True
    )

    # Loop through AOD cases
    for col_idx, AOD_case in enumerate(AOD_cases):
        # Loop through IWP cases
        for row_idx, IWP_case in enumerate(IWP_cases):
            # For each region
            for region in region_names:
                # Prepare the data for plotting
                data_to_plot = [
                    results_anomaly[
                        (year, region, var_name, AOD_case, IWP_case)
                    ]
                    for year in years
                ]
                # Convert list to numpy array
                data_to_plot = np.array(data_to_plot)

                # mean by all spatial points
                data_to_plot_array = np.nanmean(
                    data_to_plot, axis=(1, 2)
                )

                # Add the line to the plot with marker
                axs[row_idx, col_idx].plot(
                    years, data_to_plot_array, label=region, marker="o"
                )

            # Enable legend
            axs[row_idx, col_idx].legend()

            # Set the x-ticks to display as integers (not floats)
            axs[row_idx, col_idx].xaxis.set_major_locator(
                ticker.MaxNLocator(integer=True)
            )

    # Group labels for AOD cases
    for i, label in enumerate(AOD_cases):
        fig.text(
            (i + 0.5) / 3,
            1.007,
            label + " AOD",
            ha="center",
            va="center",
            fontsize=16,
        )

    # Group labels for IWP cases
    for i, label in enumerate(IWP_cases):
        fig.text(
            -0.01,
            (2.5 - i) / 3,
            label + " IWP",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=16,
        )

    # Set the x label as year
    fig.text(0.5, -0.01, "Year", ha="center", fontsize=16)

    # Set the y label as var_name anomaly
    fig.text(
        -0.038,
        0.5,
        var_name + " " + "anomaly",
        va="center",
        rotation="vertical",
        fontsize=16,
    )

    # Display the plot
    plt.tight_layout()
    plt.show()


# plot the data
# Plot HCF anomaly
plot_annual_data(results_anomaly=results_anomaly, var_name="HCF (%)")

# Plot IPR anomaly
plot_annual_data(
    results_anomaly=results_anomaly, var_name="IPR (micron)"
)

# Plot CEH anomaly
plot_annual_data(results_anomaly=results_anomaly, var_name="CEH (km)")
