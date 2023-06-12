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
import gc

import matplotlib as mpl
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
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
        extracted_PC_data[year] = PC_years[year].reshape(-1, 30, 360)

    for year in range(2017, 2021):
        extracted_Cld_data[year] = Cld_years[year].reshape(-1, 30, 360)

    for year in range(2017, 2021):
        extracted_IWP_data[year] = IWP_years[year].reshape(-1, 30, 360)

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
            lat=[i for i in range(30)],
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
        Cld_data=Cld_data_all.reshape(-1, 30, 360),
        PC_data=PC_all.reshape(-1, 30, 360),
        IWP_data=IWP_data.reshape(-1, 30, 360),
        AOD_data=AOD_data.reshape(-1, 30, 360),
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
        AOD_gap=AOD_gap,
    )

    del PC_match_PC_gap_IWP_AOD_constrain_mean

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
            "lat": np.arange(30),
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
    """
    This function filters the extreme values in IWP, AOD, PC, and Cld data.

    Args:
        IWP_data (numpy.ndarray): The IWP data.
        AOD_data (numpy.ndarray): The AOD data.
        PC_data (numpy.ndarray): The PC data.
        Cld_data (numpy.ndarray): The Cld data.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
        The filtered IWP, AOD, PC, and Cld data.

    """
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


def process_AOD_IWP_Cld_data(
    IWP_data, AOD_data, PC_data, Cld_data, IWP_gap, AOD_gap, PC_gap
):

    Cld_match = generate_filtered_data_for_all_years(
        AOD_data=AOD_data,
        IWP_data=IWP_data,
        PC_all=PC_data,
        Cld_data_all=Cld_data,
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
        AOD_gap=AOD_gap,
    )

    return Cld_match, AOD_gap, IWP_gap, PC_gap


def save_data(
    Cld_data,
    AOD_gap,
    IWP_gap,
    PC_gap,
    region,
    year_str,
    cld_var_name,
):
    save_str = f"{year_str}_filter_extreme_AOD_IWP_IPR_{cld_var_name}_{region}"
    save_filtered_data_as_nc(
        Cld_data, AOD_gap, IWP_gap, PC_gap, save_str=save_str
    )


def read_data(
    region,
    year_str,
    cld_var_name,
):
    save_str = f"{year_str}_filter_extreme_AOD_IWP_IPR_{cld_var_name}_{region}"
    save_path = "/RAID01/data/Filtered_data/"
    total_str = (
        save_path
        + "Cld_match_PC_gap_IWP_AOD_constrain_mean_"
        + save_str
        + ".nc"
    )

    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020 = (
        read_filtered_data_out(
            file_name=total_str + cld_var_name + ".nc"
        )
    )

    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019 = (
        read_filtered_data_out(
            file_name=total_str + cld_var_name + ".nc"
        )
    )

    return (
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    )


def read_all_processed_data(regions, years, cld_var_name):
    all_data = {}

    for region in regions:
        for year in years:
            all_data[(region, year)] = read_data(
                region, year, cld_var_name
            )

    return all_data


def process_2020_and_2017_2019(
    data_merra2, cld_var_name: str = "Cldarea"
):
    import gc

    # Read PCA&CLD data from netcdf file
    (
        PC_data_all,
        PC_years,
        Cld_data_all,
        Cld_years,
        IWP_data_all,
        IWP_years,
    ) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name=cld_var_name)

    # Reshape data arrays
    PC_data = {
        year: PC_years[year].reshape(-1, 180, 360)
        for year in range(2017, 2021)
    }
    Cld_data = {
        year: Cld_years[year].reshape(-1, 180, 360)
        for year in range(2017, 2021)
    }
    IWP_data = {
        year: IWP_years[year].reshape(-1, 180, 360)
        for year in range(2017, 2021)
    }

    # delete the original data to save memory
    del (
        PC_years,
        Cld_years,
        IWP_years,
        PC_data_all,
        Cld_data_all,
        IWP_data_all,
    )
    gc.collect()

    PC_2017_2019 = np.concatenate(
        [PC_data[year] for year in range(2017, 2020)], axis=0
    )
    Cld_2017_2019 = np.concatenate(
        [Cld_data[year] for year in range(2017, 2020)], axis=0
    )
    IWP_2017_2019 = np.concatenate(
        [IWP_data[year] for year in range(2017, 2020)], axis=0
    )

    # Extract Dust aerosol data for 2020 and 2017-2019
    Dust_AOD_2017_2019 = (
        data_merra2["DUEXTTAU"].sel(time=slice("2017", "2019")).values
    )

    Cld_temp = np.concatenate([Cld_2017_2019, Cld_data[2020]], axis=0)
    IWP_temp = np.concatenate([IWP_2017_2019, IWP_data[2020]], axis=0)
    PC_temp = np.concatenate([PC_2017_2019, PC_data[2020]], axis=0)

    # Preparing the data for 2020
    PC_2020 = PC_data[2020]
    Cld_2020 = Cld_data[2020]
    IWP_2020 = IWP_data[2020]
    Dust_AOD_2020 = (
        data_merra2["DUEXTTAU"].sel(time=slice("2020", "2020")).values
    )

    AOD_temp = np.concatenate(
        [Dust_AOD_2017_2019, Dust_AOD_2020], axis=0
    )
    # Data read finished
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # filter the extreme 2.5% data for IWP and AOD only
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

    # define the longitude and latitude
    latitudes = np.linspace(-90, 90, 180)
    longitudes = np.linspace(0, 360, 360)

    # define the region
    US_region = ((30, 50), (260, 310))
    # Define the two parts of the Euro region
    Euro_region_1 = ((35, 60), (345, 360))
    Euro_region_2 = ((35, 60), (0, 35))
    China_region = ((15, 40), (105, 140))

    # create a mask for each region
    US_mask = create_region_mask(latitudes, longitudes, *US_region)
    # Create the masks for each part of Euro
    Euro_mask_1 = create_region_mask(
        latitudes, longitudes, *Euro_region_1
    )
    Euro_mask_2 = create_region_mask(
        latitudes, longitudes, *Euro_region_2
    )

    # Combine the two masks into a single Euro mask
    Euro_mask = np.logical_or(Euro_mask_1, Euro_mask_2)
    China_mask = create_region_mask(
        latitudes, longitudes, *China_region
    )

    # Apply the masks to the data arrays
    IWP_temp_filtered_US = np.where(US_mask, IWP_temp_filtered, np.nan)
    IWP_temp_filtered_EURO = np.where(
        Euro_mask, IWP_temp_filtered, np.nan
    )
    IWP_temp_filtered_CHINA = np.where(
        China_mask, IWP_temp_filtered, np.nan
    )

    AOD_temp_filtered_US = np.where(US_mask, AOD_temp_filtered, np.nan)
    AOD_temp_filtered_EURO = np.where(
        Euro_mask, AOD_temp_filtered, np.nan
    )
    AOD_temp_filtered_CHINA = np.where(
        China_mask, AOD_temp_filtered, np.nan
    )

    PC_temp_filtered_US = np.where(US_mask, PC_temp_filtered, np.nan)
    PC_temp_filtered_EURO = np.where(
        Euro_mask, PC_temp_filtered, np.nan
    )
    PC_temp_filtered_CHINA = np.where(
        China_mask, PC_temp_filtered, np.nan
    )

    Cld_temp_filtered_US = np.where(US_mask, Cld_temp_filtered, np.nan)
    Cld_temp_filtered_EURO = np.where(
        Euro_mask, Cld_temp_filtered, np.nan
    )
    Cld_temp_filtered_CHINA = np.where(
        China_mask, Cld_temp_filtered, np.nan
    )

    return (
        IWP_temp_filtered_US,
        IWP_temp_filtered_EURO,
        IWP_temp_filtered_CHINA,
        AOD_temp_filtered_US,
        AOD_temp_filtered_EURO,
        AOD_temp_filtered_CHINA,
        PC_temp_filtered_US,
        PC_temp_filtered_EURO,
        PC_temp_filtered_CHINA,
        Cld_temp_filtered_US,
        Cld_temp_filtered_EURO,
        Cld_temp_filtered_CHINA,
    )


def final_process(cld_var_name):
    years = ["2017_2019", "2020"]
    regions = ["US", "EURO", "CHINA"]

    # Create dictionaries for your data
    IWP_temp_filtered = {
        region: globals()[f"IWP_temp_filtered_{region}"]
        for region in regions
    }
    AOD_temp_filtered = {
        region: globals()[f"AOD_temp_filtered_{region}"]
        for region in regions
    }
    PC_temp_filtered = {
        region: globals()[f"PC_temp_filtered_{region}"]
        for region in regions
    }
    Cld_temp_filtered = {
        region: globals()[f"Cld_temp_filtered_{region}"]
        for region in regions
    }

    for region in regions:
        # Divide the data
        Cld = {
            year: Cld_temp_filtered[region][:1008, :, :]
            if year == "2017_2019"
            else Cld_temp_filtered[region][1008:, :, :]
            for year in years
        }
        IWP = {
            year: IWP_temp_filtered[region][:1008, :, :]
            if year == "2017_2019"
            else IWP_temp_filtered[region][1008:, :, :]
            for year in years
        }
        PC = {
            year: PC_temp_filtered[region][:1008, :, :]
            if year == "2017_2019"
            else PC_temp_filtered[region][1008:, :, :]
            for year in years
        }
        AOD = {
            year: AOD_temp_filtered[region][:1008, :, :]
            if year == "2017_2019"
            else AOD_temp_filtered[region][1008:, :, :]
            for year in years
        }

        divide_IWP = DividePCByDataVolume(
            dataarray_main=IWP_temp_filtered[region],
            n=30,
        )
        IWP_gap = divide_IWP.main_gap()

        divide_PC = DividePCByDataVolume(
            dataarray_main=PC_temp_filtered[region],
            n=30,
        )
        PC_gap = divide_PC.main_gap()

        divide_AOD = DividePCByDataVolume(
            dataarray_main=AOD_temp_filtered[region],
            n=6,
        )
        AOD_gap = divide_AOD.main_gap()

        for year in years:
            # Process the data
            (
                Cld_match,
                AOD_gap,
                IWP_gap,
                PC_gap,
            ) = process_AOD_IWP_Cld_data(
                IWP[year], AOD[year], PC[year], Cld[year], IWP_gap, AOD_gap, PC_gap
            )

            # Save the data
            save_data(
                Cld_match,
                AOD_gap,
                IWP_gap,
                PC_gap,
                region,
                year,
                cld_var_name,
            )

    # Read the filtered data
    # Read the Dust_AOD constrain data for 2020 and 2017-2019 data
    all_data = read_all_processed_data(regions, years, cld_var_name)

    return all_data


my_cld_var_name = "Cldicerad"

# First process the 2020 and 2017-2019 data into different regions
(
    IWP_temp_filtered_US,
    IWP_temp_filtered_EURO,
    IWP_temp_filtered_CHINA,
    AOD_temp_filtered_US,
    AOD_temp_filtered_EURO,
    AOD_temp_filtered_CHINA,
    PC_temp_filtered_US,
    PC_temp_filtered_EURO,
    PC_temp_filtered_CHINA,
    Cld_temp_filtered_US,
    Cld_temp_filtered_EURO,
    Cld_temp_filtered_CHINA,
) = process_2020_and_2017_2019(
    data_merra2_2010_2020_new_lon, cld_var_name=my_cld_var_name
)

# final process of the data to filtered out the data
all_data = final_process(cld_var_name=my_cld_var_name)

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

regions = ["US", "EURO", "CHINA"]
years = ["2017_2019", "2020"]

# Iterate over all_data dictionary
for (region, year), data in all_data.items():
    # Unpack the data
    (
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    ) = data

    # Create specific variable names for each region and year
    var_name_2020 = (
        f"Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_{region}"
    )
    var_name_2017_2019 = (
        f"Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_{region}"
    )

    # Assign data to the newly created variables
    globals()[
        var_name_2020
    ] = Cld_match_PC_gap_IWP_AOD_constrain_mean_2020
    globals()[
        var_name_2017_2019
    ] = Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019


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
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name="Cldarea")


def plot_global_spatial_distribution(
    data,
    var_name,
    title,
    vmin=None,
    vmax=None,
):
    # Create lon and lat
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    # Create custom colormap
    cmap = plt.cm.get_cmap("RdYlBu_r")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 8),
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
        shrink=0.65,
        extend="both",
    )
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


Dust_AOD_all = (
    data_merra2_2010_2020_new_lon["DUEXTTAU"]
    .sel(time=slice("2010", "2021"))
    .values
)

# define the latitudes and longitudes
latitudes = np.linspace(-90, 90, 180)  # latitudes from -90 to 90
longitudes = np.linspace(0, 360, 360)  # longitudes from 0 to 360

# define the region
US_region = ((30, 50), (260, 310))
# Define the two parts of the Euro region
Euro_region_1 = ((35, 60), (345, 360))
Euro_region_2 = ((35, 60), (0, 35))
China_region = ((15, 40), (105, 140))

# Create the masks for the regions
US_mask = create_region_mask(latitudes, longitudes, *US_region)

# Create the masks for each part
Euro_mask_1 = create_region_mask(latitudes, longitudes, *Euro_region_1)
Euro_mask_2 = create_region_mask(latitudes, longitudes, *Euro_region_2)

# Combine the two masks into a single Euro mask
Euro_mask = np.logical_or(Euro_mask_1, Euro_mask_2)

China_mask = create_region_mask(latitudes, longitudes, *China_region)

# Apply the masks to the data arrays
Cld_all_US = np.where(US_mask, Cld_all, np.nan)
Cld_all_EURO = np.where(Euro_mask, Cld_all, np.nan)
Cld_all_CHINA = np.where(China_mask, Cld_all, np.nan)

PC_all_US = np.where(US_mask, PC_all, np.nan)
PC_all_EURO = np.where(Euro_mask, PC_all, np.nan)
PC_all_CHINA = np.where(China_mask, PC_all, np.nan)

IWP_data_US = np.where(US_mask, IWP_data, np.nan)
IWP_data_EURO = np.where(Euro_mask, IWP_data, np.nan)
IWP_data_CHINA = np.where(China_mask, IWP_data, np.nan)

Dust_AOD_all_US = np.where(US_mask, Dust_AOD_all, np.nan)
Dust_AOD_all_EURO = np.where(Euro_mask, Dust_AOD_all, np.nan)
Dust_AOD_all_CHINA = np.where(China_mask, Dust_AOD_all, np.nan)

plot_global_spatial_distribution(
    data=np.nanmean(Cld_all, axis=(0, 1, 2)),
    var_name="Cloud Area Fraction",
    title="Cloud Area Fraction",
    vmin=1,
    vmax=45,
)

# plot cld data for each region
plot_global_spatial_distribution(
    data=np.nanmean(Cld_all_US, axis=(0, 1, 2)),
    var_name="Cloud Area Fraction",
    title="Cloud Area Fraction in US",
    vmin=1,
    vmax=45,
)
plot_global_spatial_distribution(
    data=np.nanmean(Cld_all_EURO, axis=(0, 1, 2)),
    var_name="Cloud Area Fraction",
    title="Cloud Area Fraction in EURO",
    vmin=1,
    vmax=45,
)
plot_global_spatial_distribution(
    data=np.nanmean(Cld_all_CHINA, axis=(0, 1, 2)),
    var_name="Cloud Area Fraction",
    title="Cloud Area Fraction in CHINA",
    vmin=1,
    vmax=45,
)

# plot IWP data for each region
plot_global_spatial_distribution(
    data=np.nanmean(IWP_data, axis=(0, 1, 2)),
    var_name="IWP",
    title="IWP",
    vmin=10,
    vmax=510,
)

plot_global_spatial_distribution(
    data=np.nanmean(IWP_data_US, axis=(0, 1, 2)),
    var_name="IWP",
    title="IWP in US",
    vmin=10,
    vmax=510,
)
plot_global_spatial_distribution(
    data=np.nanmean(IWP_data_EURO, axis=(0, 1, 2)),
    var_name="IWP",
    title="IWP in EURO",
    vmin=10,
    vmax=510,
)
plot_global_spatial_distribution(
    data=np.nanmean(IWP_data_CHINA, axis=(0, 1, 2)),
    var_name="IWP",
    title="IWP in CHINA",
    vmin=10,
    vmax=510,
)

# plot Dust AOD data for each region
plot_global_spatial_distribution(
    data=np.nanmean(Dust_AOD_all, axis=(0)),
    var_name="Dust AOD",
    title="Dust AOD",
    vmin=0.01,
    vmax=0.58,
)

plot_global_spatial_distribution(
    data=np.nanmean(Dust_AOD_all_US, axis=(0)),
    var_name="Dust AOD",
    title="Dust AOD in US",
    vmin=0.01,
    vmax=0.58,
)
plot_global_spatial_distribution(
    data=np.nanmean(Dust_AOD_all_EURO, axis=(0)),
    var_name="Dust AOD",
    title="Dust AOD in EURO",
    vmin=0.01,
    vmax=0.58,
)
plot_global_spatial_distribution(
    data=np.nanmean(Dust_AOD_all_CHINA, axis=(0)),
    var_name="Dust AOD",
    title="Dust AOD in CHINA",
    vmin=0.01,
    vmax=0.58,
)

# plot PC data for each region
plot_global_spatial_distribution(
    data=np.nanmean(PC_all, axis=(0, 1, 2)),
    var_name="PC1",
    title="PC1",
    vmin=-2.7,
    vmax=3.1,
)

plot_global_spatial_distribution(
    data=np.nanmean(PC_all_US, axis=(0, 1, 2)),
    var_name="PC1",
    title="PC1 in US",
    vmin=-2.7,
    vmax=3.1,
)
plot_global_spatial_distribution(
    data=np.nanmean(PC_all_EURO, axis=(0, 1, 2)),
    var_name="PC1",
    title="PC1 in EURO",
    vmin=-2.7,
    vmax=3.1,
)
plot_global_spatial_distribution(
    data=np.nanmean(PC_all_CHINA, axis=(0, 1, 2)),
    var_name="PC1",
    title="PC1 in CHINA",
    vmin=-2.7,
    vmax=3.1,
)


########################################################################
# ------------------------------------------------------------------
# Plot 4D plot to demenstrate the constrain of PC1, IWP, Dust AOD
# ------------------------------------------------------------------
########################################################################


# version 3
# Seperate the 3D plot into 2 subplots by AOD interval
# color nan values with self-defined color
# color nan values with self-defined color
def create_colormap_with_nan(cmap_name, nan_color="silver"):
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))
    cmap_with_nan = ListedColormap(colors)
    cmap_with_nan.set_bad(nan_color)
    return cmap_with_nan


def plot_3d_colored_IWP_PC1_AOD(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    high_cloud_amount_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    savefig_str,
    aod_range,  # Add this parameter to define the AOD range
    cmap="Spectral_r",  # Add this parameter to define the custom colormap
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
        cmap (str or matplotlib.colors.Colormap): Custom colormap to use for the plot
    """

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    # Compute the 90th percentile value for normalization
    norm_value = np.nanpercentile(high_cloud_amount_mean, 99.999)

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
                high_cloud_amount_mean[aod_num] / norm_value
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
    m = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap))
    m.set_cmap(cmap_with_nan)
    m.set_array(high_cloud_amount_mean)
    m.set_clim(np.nanmin(high_cloud_amount_mean), norm_value)
    fig.colorbar(
        m, shrink=0.3, aspect=9, pad=0.01, label=colobar_label
    )

    ax.view_init(elev=20, azim=-66)
    ax.dist = 12

    # Save the figure
    plt.savefig(savefig_str)
    plt.show()


def plot_both_3d_fill_plot(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
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
    plot_3d_colored_IWP_PC1_AOD(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        savefig_str="subplot_1.png",
        aod_range=(0, 3),
    )
    plot_3d_colored_IWP_PC1_AOD(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        savefig_str="subplot_2.png",
        aod_range=(3, 6),
    )


# version 4
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
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    "Dust-AOD",
    "PC1",
    "IWP",
    "HCF (%)",
    vmin=0,
    vmax=40,
)

# 2017-2019 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "HCF (%)",
    vmin=0,
    vmax=42,
)

# Plot the dust-AOD constrained ice effective radius data
# 2020 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    "Dust-AOD",
    "PC1",
    "IWP",
    "IPR (micron)",
    vmin=23,
    vmax=30.5,
)

# 2017-2019 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "IPR (micron)",
    vmin=23,
    vmax=30.5,
)

# Plot the dust-AOD constrained cloud effective height data
# 2020 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CEH (km)",
    vmin=10.4,
    vmax=14.6,
)

# 2017-2019 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CEH (km)",
    vmin=10.4,
    vmax=14.7,
)

# Plot the dust-AOD constrained cloud top height data
# 2020 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CTH (km)",
    vmin=12.1,
    vmax=14.9,
)

# 2017-2019 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CTH (km)",
    vmin=12.1,
    vmax=14.9,
)

# Plot the dust-AOD constrained cloud emissivity data
# 2020 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CEMIS",
    vmin=0,
    vmax=0.27,
)

# 2017-2019 filtered data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CEMIS",
    vmin=0,
    vmax=0.27,
)

# ------------------------------------------------------------------------------------------
# 2020 - (2017-2019) filtered data
# plot the HCF data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020
    - Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "HCF (%)",
    vmin=-3.5,
    vmax=3.5,
    cmap="RdBu_r",
)

# plot the IER data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020
    - Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "IPR (micron)",
    vmin=-3.2,
    vmax=3.2,
    cmap="RdBu_r",
)

# plot the CEH data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020
    - Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CEH (km)",
    vmin=-0.8,
    vmax=0.8,
    cmap="RdBu_r",
)

# plot the cemis data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020
    - Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CEMIS",
    vmin=-0.05,
    vmax=0.05,
    cmap="RdBu_r",
)

# plot the CTH data
plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020
    - Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CTH (km)",
    vmin=-1.3,
    vmax=1.3,
    cmap="RdBu_r",
)

# -----------------------------------------------------------------------
# Plot spatial distribution of different AOD gap values
# -----------------------------------------------------------------------
# Divide 3, Dust AOD data
# Divide AOD data as well
AOD_temp = np.concatenate([Dust_AOD_2017_2019, Dust_AOD_2020], axis=0)
divide_AOD = DividePCByDataVolume(
    dataarray_main=AOD_temp,
    n=6,
)
AOD_gap = divide_AOD.main_gap()

Dust_2017_2020_mean = np.nanmean(
    np.concatenate([Dust_AOD_2017_2019, Dust_AOD_2020], axis=0), axis=0
)


def plot_spatial_distribution(data, var_name, title, AOD_intervals):
    from matplotlib.colors import BoundaryNorm

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(20, 49, 30)

    lons, lats = np.meshgrid(lon, lat)

    from matplotlib.colors import LinearSegmentedColormap

    # Define colors for each interval (use as many colors as the number of intervals)
    colors = ["green", "blue", "yellow", "orange", "red", "purple"]

    # Create custom colormap
    cmap = plt.cm.get_cmap("viridis", len(AOD_intervals) - 1)

    # Set up the norm and boundaries for the colormap
    norm = BoundaryNorm(AOD_intervals, cmap.N)

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12.5, 3),
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
        norm=norm,
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
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


plot_spatial_distribution(
    data=Dust_2017_2020_mean,
    var_name="Dust AOD",
    title="Spatial Distribution of Dust AOD Section",
    AOD_intervals=AOD_gap,
)


def plot_global_spatial_distribution(
    data,
    var_name,
    title,
):
    from matplotlib.colors import BoundaryNorm

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(20, 49, 30)

    lons, lats = np.meshgrid(lon, lat)

    # Create custom colormap
    cmap = plt.cm.get_cmap("RdBu_r")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12.5, 3),
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
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


Dust_AOD_2020_mean = np.nanmean(Dust_AOD_2020, axis=0)
PC_2020_mean = np.nanmean(PC_2020, axis=0)
Cld_2020_mean = np.nanmean(Cld_2020, axis=0)
IWP_2020_mean = np.nanmean(IWP_2020, axis=0)

Dust_AOD_2017_2019_mean = np.nanmean(Dust_AOD_2017_2019, axis=0)
PC_2017_2019_mean = np.nanmean(PC_2017_2019, axis=0)
Cld_2017_2019_mean = np.nanmean(Cld_2017_2019, axis=0)
IWP_2017_2019_mean = np.nanmean(IWP_2017_2019, axis=0)

# verify the input data
plot_global_spatial_distribution(
    Dust_AOD_2020_mean,
    "AOD",
    "AOD Spatial Distribution",
)
plot_global_spatial_distribution(
    PC_2020_mean,
    "PC1",
    "PC1 Spatial Distribution",
)
plot_global_spatial_distribution(
    Cld_2020_mean,
    "CEH",
    "CEH Spatial Distribution",
)
plot_global_spatial_distribution(
    IWP_2020_mean, "IWP", "IWP Spatial Distribution"
)


plot_global_spatial_distribution(
    Dust_AOD_2017_2019_mean,
    "AOD",
    "AOD Spatial Distribution",
)
plot_global_spatial_distribution(
    PC_2017_2019_mean,
    "PC1",
    "PC1 Spatial Distribution",
)
plot_global_spatial_distribution(
    Cld_2017_2019_mean,
    "CEH",
    "CEH Spatial Distribution",
)
plot_global_spatial_distribution(
    IWP_2017_2019_mean, "IWP", "IWP Spatial Distribution"
)


def plot_statistical_distribution(data, var_name):
    # Flatten the data
    flat_data = data.flatten()

    # Plot histogram and KDE
    plt.figure(figsize=(12, 6))

    # Plot the histogram
    sns.histplot(
        flat_data,
        kde=False,
        bins=100,
        color="blue",
        alpha=0.6,
        stat="density",
        label="Histogram",
    )

    # Plot the KDE
    sns.kdeplot(flat_data, color="red", lw=2, label="KDE")

    plt.xlabel(var_name)
    plt.ylabel("Probability Density of " + var_name)
    plt.title("Probability Density Function (PDF)")
    plt.legend()
    plt.show()


# verify the input data
# 2020
plot_statistical_distribution(PC_2020, "PC1")
plot_statistical_distribution(Cld_2020, "CEH")
plot_statistical_distribution(Dust_AOD_2020, "AOD")
plot_statistical_distribution(IWP_2020, "IWP")

# 2017-2019
plot_statistical_distribution(PC_2017_2019, "PC1")
plot_statistical_distribution(Cld_2017_2019, "CEH")
plot_statistical_distribution(Dust_AOD_2017_2019, "AOD")
plot_statistical_distribution(IWP_2017_2019, "IWP")

# #######################################################################
# ###### We only analyze the april to july cld and pc1 data #############
# ###### In order to extract the contrail maximum signal ################
# #######################################################################

# # region
# (
#     Cld_lowermost_error_2017_2019,
#     Cld_highermost_error_2017_2019,
#     Cld_2017_2019_match_PC_gap_filtered_1_2_month_median,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2017_2019_match_PC_gap_1_2_month_median
# )
# (
#     Cld_lowermost_error_2020,
#     Cld_highermost_error_2020,
#     Cld_2020_match_PC_gap_filtered_1_2_month_median,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2020_match_PC_gap_1_2_month_median
# )

# (
#     Cld_lowermost_error_2017_2019,
#     Cld_highermost_error_2017_2019,
#     Cld_2017_2019_match_PC_gap_filtered_1_2_month_mean,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2017_2019_match_PC_gap_1_2_month_mean
# )
# (
#     Cld_lowermost_error_2020,
#     Cld_highermost_error_2020,
#     Cld_2020_match_PC_gap_filtered_1_2_month_mean,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2020_match_PC_gap_1_2_month_mean
# )

# (
#     Cld_lowermost_error_2017_2019,
#     Cld_highermost_error_2017_2019,
#     Cld_2017_2019_match_PC_gap_filtered_3_4_month_median,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2017_2019_match_PC_gap_3_4_month_median
# )
# (
#     Cld_lowermost_error_2020,
#     Cld_highermost_error_2020,
#     Cld_2020_match_PC_gap_filtered_3_4_month_median,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2020_match_PC_gap_3_4_month_median
# )

# (
#     Cld_lowermost_error_2017_2019,
#     Cld_highermost_error_2017_2019,
#     Cld_2017_2019_match_PC_gap_filtered_3_4_month_mean,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2017_2019_match_PC_gap_3_4_month_mean
# )
# (
#     Cld_lowermost_error_2020,
#     Cld_highermost_error_2020,
#     Cld_2020_match_PC_gap_filtered_3_4_month_mean,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2020_match_PC_gap_3_4_month_mean
# )

# (
#     Cld_lowermost_error_2017_2019,
#     Cld_highermost_error_2017_2019,
#     Cld_2017_2019_match_PC_gap_filtered_5_6_month_median,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2017_2019_match_PC_gap_5_6_month_median
# )
# (
#     Cld_lowermost_error_2020,
#     Cld_highermost_error_2020,
#     Cld_2020_match_PC_gap_filtered_5_6_month_median,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2020_match_PC_gap_5_6_month_median
# )

# (
#     Cld_lowermost_error_2017_2019,
#     Cld_highermost_error_2017_2019,
#     Cld_2017_2019_match_PC_gap_filtered_5_6_month_mean,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2017_2019_match_PC_gap_5_6_month_mean
# )
# (
#     Cld_lowermost_error_2020,
#     Cld_highermost_error_2020,
#     Cld_2020_match_PC_gap_filtered_5_6_month_mean,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2020_match_PC_gap_5_6_month_mean
# )

# (
#     Cld_lowermost_error_2017_2019,
#     Cld_highermost_error_2017_2019,
#     Cld_2017_2019_match_PC_gap_filtered_7_8_month_median,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2017_2019_match_PC_gap_7_8_month_median
# )
# (
#     Cld_lowermost_error_2020,
#     Cld_highermost_error_2020,
#     Cld_2020_match_PC_gap_filtered_7_8_month_median,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2020_match_PC_gap_7_8_month_median
# )

# (
#     Cld_lowermost_error_2017_2019,
#     Cld_highermost_error_2017_2019,
#     Cld_2017_2019_match_PC_gap_filtered_7_8_month_mean,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2017_2019_match_PC_gap_7_8_month_mean
# )
# (
#     Cld_lowermost_error_2020,
#     Cld_highermost_error_2020,
#     Cld_2020_match_PC_gap_filtered_7_8_month_mean,
# ) = filter_data_PC1_gap_lowermost_highermost_error(
#     Cld_2020_match_PC_gap_7_8_month_mean
# )
# # endregion

# #######################################################################
# #######################################################################
# #######################################################################
# ###### We only analyze the april to july cld and pc1 data #############
# ###### In order to extract the contrail maximum signal ################
# #######################################################################

# # region
# # -----------------------------------------------------------------------------
# # divide the PC1 data into 6 parts

# concatenated_data = np.concatenate(
#     (
#         PC_2020_1_2_month[:, 110:140, :],
#         PC_2020_3_4_month[:, 110:140, :],
#         PC_2020_5_6_month[:, 110:140, :],
#         PC_2020_7_8_month[:, 110:140, :],
#     ),
#     axis=0,
# )

# dividePC = DividePCByDataVolume(
#     dataarray_main=concatenated_data,
#     n=5,
#     start=-2.5,
#     end=5.5,
#     gap=0.05,
# )
# gap_num_min, gap_num_max = dividePC.gap_number_for_giving_main_gap()

# # retrieve the sparse PC_constrain data
# # dimension 1 means atmospheric condiations
# # 0 means bad, 1 means moderate, 2 means good
# Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean = np.empty(
#     (3, 360)
# )
# Cld_delta_2020_match_PC_gap_filtered_3_4_month_mean = np.empty(
#     (3, 360)
# )
# Cld_delta_2020_match_PC_gap_filtered_5_6_month_mean = np.empty(
#     (3, 360)
# )
# Cld_delta_2020_match_PC_gap_filtered_7_8_month_mean = np.empty(
#     (3, 360)
# )
# # 1-2 month
# for i in range(0, 3):
#     Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean[
#         i
#     ] = np.nanmean(
#         Cld_2020_match_PC_gap_filtered_1_2_month_mean[
#             int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
#         ],
#         axis=(0, 1),
#     ) - np.nanmean(
#         Cld_2017_2019_match_PC_gap_filtered_1_2_month_mean[
#             int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
#         ],
#         axis=(0, 1),
#     )
#     # 3-4 month
#     Cld_delta_2020_match_PC_gap_filtered_3_4_month_mean[
#         i
#     ] = np.nanmean(
#         Cld_2020_match_PC_gap_filtered_3_4_month_mean[
#             int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
#         ],
#         axis=(0, 1),
#     ) - np.nanmean(
#         Cld_2017_2019_match_PC_gap_filtered_3_4_month_mean[
#             int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
#         ],
#         axis=(0, 1),
#     )
#     # 5-6 month
#     Cld_delta_2020_match_PC_gap_filtered_5_6_month_mean[
#         i
#     ] = np.nanmean(
#         Cld_2020_match_PC_gap_filtered_5_6_month_mean[
#             int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
#         ],
#         axis=(0, 1),
#     ) - np.nanmean(
#         Cld_2017_2019_match_PC_gap_filtered_5_6_month_mean[
#             int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
#         ],
#         axis=(0, 1),
#     )
#     # 7-8 month
#     Cld_delta_2020_match_PC_gap_filtered_7_8_month_mean[
#         i
#     ] = np.nanmean(
#         Cld_2020_match_PC_gap_filtered_7_8_month_mean[
#             int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
#         ],
#         axis=(0, 1),
#     ) - np.nanmean(
#         Cld_2017_2019_match_PC_gap_filtered_7_8_month_mean[
#             int(gap_num_min[i]) : int(gap_num_max[i]), 110:140, :
#         ],
#         axis=(0, 1),
#     )


# # Improved version
# # define lists for atmospheric conditions and calculation type
# atm_conditions = ["bad", "moderate", "good"]
# calc_types = ["median", "mean"]

# # create empty dictionaries to store the results
# results_median = {}
# results_mean = {}

# # -----------------------------------------------------------------------------
# # 1 - 2 month

# # iterate over atmospheric conditions and calculation types
# for i, atm in enumerate(atm_conditions):
#     for calc_type in calc_types:
#         # create the variable names dynamically
#         var_name = f"{atm}_filtered_{calc_type}"
#         cld_2020_var = (
#             f"Cld_2020_match_PC_gap_filtered_1_2_month_{calc_type}"
#         )
#         cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_1_2_month_{calc_type}"
#         gap_min_var = gap_num_min[i]
#         gap_max_var = gap_num_max[i]

#         # call the compare_cld_between_2020_others function
#         results = compare_cld_between_2020_others(
#             Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
#             Cld_all_match_PC_gap_others=locals()[cld_others_var],
#             start=int(gap_min_var),
#             end=int(gap_max_var),
#         )

#         # store the results in the appropriate dictionary
#         if calc_type == "median":
#             results_median[var_name] = results
#         elif calc_type == "mean":
#             results_mean[var_name] = results

# # assign the results to variables
# Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median = (
#     results_median["bad_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median = (
#     results_median["moderate_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median = (
#     results_median["good_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean = (
#     results_mean["bad_filtered_mean"]
# )
# Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean = (
#     results_mean["moderate_filtered_mean"]
# )
# Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean = (
#     results_mean["good_filtered_mean"]
# )

# # ---------------------------------------------------------------------------------
# # 3 - 4 month
# # iterate over atmospheric conditions and calculation types
# for i, atm in enumerate(atm_conditions):
#     for calc_type in calc_types:
#         # create the variable names dynamically
#         var_name = f"{atm}_filtered_{calc_type}"
#         cld_2020_var = (
#             f"Cld_2020_match_PC_gap_filtered_3_4_month_{calc_type}"
#         )
#         cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_3_4_month_{calc_type}"
#         gap_min_var = gap_num_min[i]
#         gap_max_var = gap_num_max[i]

#         # call the compare_cld_between_2020_others function
#         results = compare_cld_between_2020_others(
#             Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
#             Cld_all_match_PC_gap_others=locals()[cld_others_var],
#             start=int(gap_min_var),
#             end=int(gap_max_var),
#         )

#         # store the results in the appropriate dictionary
#         if calc_type == "median":
#             results_median[var_name] = results
#         elif calc_type == "mean":
#             results_mean[var_name] = results

# # assign the results to variables
# Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median = (
#     results_median["bad_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median = (
#     results_median["moderate_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median = (
#     results_median["good_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean = (
#     results_mean["bad_filtered_mean"]
# )
# Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean = (
#     results_mean["moderate_filtered_mean"]
# )
# Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean = (
#     results_mean["good_filtered_mean"]
# )

# # --------------------------------------------------------------------------------------------
# # 5 - 6 month
# # iterate over atmospheric conditions and calculation types
# for i, atm in enumerate(atm_conditions):
#     for calc_type in calc_types:
#         # create the variable names dynamically
#         var_name = f"{atm}_filtered_{calc_type}"
#         cld_2020_var = (
#             f"Cld_2020_match_PC_gap_filtered_5_6_month_{calc_type}"
#         )
#         cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_5_6_month_{calc_type}"
#         gap_min_var = gap_num_min[i]
#         gap_max_var = gap_num_max[i]

#         # call the compare_cld_between_2020_others function
#         results = compare_cld_between_2020_others(
#             Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
#             Cld_all_match_PC_gap_others=locals()[cld_others_var],
#             start=int(gap_min_var),
#             end=int(gap_max_var),
#         )

#         # store the results in the appropriate dictionary
#         if calc_type == "median":
#             results_median[var_name] = results
#         elif calc_type == "mean":
#             results_mean[var_name] = results

# Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median = (
#     results_median["bad_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median = (
#     results_median["moderate_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median = (
#     results_median["good_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean = (
#     results_mean["bad_filtered_mean"]
# )
# Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean = (
#     results_mean["moderate_filtered_mean"]
# )
# Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean = (
#     results_mean["good_filtered_mean"]
# )

# # --------------------------------------------------------------------------------------------
# # 7 - 8 month
# # iterate over atmospheric conditions and calculation types
# for i, atm in enumerate(atm_conditions):
#     for calc_type in calc_types:
#         # create the variable names dynamically
#         var_name = f"{atm}_filtered_{calc_type}"
#         cld_2020_var = (
#             f"Cld_2020_match_PC_gap_filtered_7_8_month_{calc_type}"
#         )
#         cld_others_var = f"Cld_2017_2019_match_PC_gap_filtered_7_8_month_{calc_type}"
#         gap_min_var = gap_num_min[i]
#         gap_max_var = gap_num_max[i]

#         # call the compare_cld_between_2020_others function
#         results = compare_cld_between_2020_others(
#             Cld_all_match_PC_gap_2020=locals()[cld_2020_var],
#             Cld_all_match_PC_gap_others=locals()[cld_others_var],
#             start=int(gap_min_var),
#             end=int(gap_max_var),
#         )

#         # store the results in the appropriate dictionary
#         if calc_type == "median":
#             results_median[var_name] = results
#         elif calc_type == "mean":
#             results_mean[var_name] = results

# Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median = (
#     results_median["bad_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median = (
#     results_median["moderate_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median = (
#     results_median["good_filtered_median"]
# )
# Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean = (
#     results_mean["bad_filtered_mean"]
# )
# Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean = (
#     results_mean["moderate_filtered_mean"]
# )
# Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean = (
#     results_mean["good_filtered_mean"]
# )
# # --------------------------------------------------------------------------------------------
# # endregion

# ########################################################################
# ###### Draw the Cld difference between 2020 and 2010-2019 ###############
# ###### But by latitude this time, 20N-60N, Contrail region ##############
# ########################################################################

# #########################################################
# ############ smoothed data ##############################
# #########################################################
# # ------------------------------------------------------------
# # plot the aviation fill between version
# # ------------------------------------------------------------
# y_lim_lst = [[-4, 6.5], [-4.5, 6.5], [-5.2, 8]]

# # 1-2 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
#     - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="January-February",
#     step=5,
# )

# # 3-4 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
#     - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="March-April",
#     step=5,
# )

# # 5-6 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
#     - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="May-June",
#     step=5,
# )

# # 7-8 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
#     - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="July-August",
#     step=5,
# )

# # no compare between PC1 and no PC1 constrain
# # ------------------------------------------------------------
# # plot the aviation fill between version
# # ------------------------------------------------------------
# y_lim_lst = [[-1.15, 1.15], [-1.15, 1.15], [-1.15, 1.15]]

# # 1-2 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
#     - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="January-February",
#     step=10,
# )

# # 3-4 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
#     - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="March-April",
#     step=10,
# )

# # 5-6 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
#     - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="May-June",
#     step=5,
# )

# # 7-8 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_no_compare(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
#     - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="July-August",
#     step=10,
# )


# # zscores
# # ------------------------------------------------------------
# # plot the aviation fill between version
# # ------------------------------------------------------------
# y_lim_lst = [[-1.15, 1.15], [-1.15, 1.15], [-1.15, 1.15]]

# # 1-2 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_1_2_month, axis=0)
#     - np.nanmean(Cld_2017_2019_1_2_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="January-February",
#     step=10,
# )

# # 3-4 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_3_4_month, axis=0)
#     - np.nanmean(Cld_2017_2019_3_4_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="March-April",
#     step=10,
# )

# # 5-6 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_5_6_month, axis=0)
#     - np.nanmean(Cld_2017_2019_5_6_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="May-June",
#     step=5,
# )

# # 7-8 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_zscore(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
#     Cld_data_aux=np.nanmean(Cld_2020_7_8_month, axis=0)
#     - np.nanmean(Cld_2017_2019_7_8_month, axis=0),
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="July-August",
#     step=10,
# )

# # --------------------------------------------
# # plot the improve version
# # --------------------------------------------
# y_lim_lst = [[-4, 6.5], [-4.5, 6.5], [-5.2, 8]]

# # 1-2 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_1_2_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_1_2_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_1_2_month_good_filtered_median,
#     Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="January-February",
#     step=5,
# )

# # 3-4 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_3_4_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_3_4_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_3_4_month_good_filtered_median,
#     Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="March-April",
#     step=5,
# )

# # 5-6 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_5_6_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_5_6_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_5_6_month_good_filtered_median,
#     Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="May-June",
#     step=5,
# )

# # 7-8 month
# compare_cld_between_PC_condition_by_each_Lat_smoothed_aviation_fill_median_mean_with_actual_aviation_improve(
#     Cld_data_PC_condition_0_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_mean,
#     Cld_data_PC_condition_1_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_mean,
#     Cld_data_PC_condition_2_mean=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_mean,
#     Cld_data_PC_condition_0_median=Cld_all_match_PC_gap_sub_2020_7_8_month_bad_filtered_median,
#     Cld_data_PC_condition_1_median=Cld_all_match_PC_gap_sub_2020_7_8_month_moderate_filtered_median,
#     Cld_data_PC_condition_2_median=Cld_all_match_PC_gap_sub_2020_7_8_month_good_filtered_median,
#     Cld_data_aux_proses=Cld_delta_2020_match_PC_gap_filtered_1_2_month_mean,
#     Cld_data_name=r"$\Delta$" + "HCF(%)",
#     y_lim_lst=y_lim_lst,
#     title="July-August",
#     step=5,
# )

# # ############################################################################################################
# # ####### Expired plot function #######

# # # # version 1
# # # def plot_3d_colored_IWP_PC1_AOD(
# # #     Cld_match_PC_gap_IWP_AOD_constrain_mean,
# # #     high_cloud_amount_mean,
# # #     xlabel,
# # #     ylabel,
# # #     zlabel,
# # #     savefig_str,
# # # ):
# # #     """
# # #     Create a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval.

# # #     Args:
# # #         Cld_match_PC_gap_IWP_AOD_constrain_mean (numpy.array): Cloud data under each restriction, shape (AOD_bin, IWP_bin, PC_bin, lat, lon)
# # #         xlabel (str): Label for the x-axis
# # #         ylabel (str): Label for the y-axis
# # #         zlabel (str): Label for the z-axis
# # #         savefig_str (str): String for saving the figure
# # #     """

# # #     fig = plt.figure(figsize=(15, 15))
# # #     ax = fig.add_subplot(111, projection="3d")

# # #     AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
# # #     IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
# # #     PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

# # #     X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

# # #     for aod_num in range(AOD_bin):
# # #         Z = aod_num * np.ones_like(X)

# # #         # Plot the 2D pcolormesh color fill map for the current AOD interval
# # #         ax.plot_surface(
# # #             Z,
# # #             X,
# # #             Y,
# # #             rstride=1,
# # #             cstride=1,
# # #             facecolors=plt.cm.Spectral_r(
# # #                 high_cloud_amount_mean[aod_num]
# # #                 / np.nanmax(high_cloud_amount_mean)
# # #             ),
# # #             shade=False,
# # #             edgecolor="none",
# # #             alpha=0.95,
# # #             antialiased=False,
# # #             linewidth=0,  # Add this line to remove grid lines
# # #         )

# # #     ax.set_xlabel(xlabel)
# # #     ax.set_ylabel(ylabel)
# # #     ax.set_zlabel(zlabel)

# # #     # Turn off the grid lines
# # #     ax.grid(False)

# # #     # Add color bar
# # #     m = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r)
# # #     m.set_array(high_cloud_amount_mean)
# # #     m.set_clim(
# # #         np.nanmin(high_cloud_amount_mean),
# # #         np.nanmax(high_cloud_amount_mean),
# # #     )
# # #     fig.colorbar(m, shrink=0.3, aspect=9, pad=0.01, label="HCF (%)")

# # #     ax.view_init(elev=20, azim=-60)
# # #     ax.dist = 12

# # #     # Save the figure
# # #     plt.savefig(savefig_str)
# # #     plt.show()


# # # # version 2
# # # def plot_3d_colored_IWP_PC1_AOD(
# # #     Cld_match_PC_gap_IWP_AOD_constrain_mean,
# # #     high_cloud_amount_mean,
# # #     xlabel,
# # #     ylabel,
# # #     zlabel,
# # #     savefig_str,
# # # ):
# # #     """
# # #     Create a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval.

# # #     Args:
# # #         Cld_match_PC_gap_IWP_AOD_constrain_mean (numpy.array): Cloud data under each restriction, shape (AOD_bin, IWP_bin, PC_bin, lat, lon)
# # #         xlabel (str): Label for the x-axis
# # #         ylabel (str): Label for the y-axis
# # #         zlabel (str): Label for the z-axis
# # #         savefig_str (str): String for saving the figure
# # #     """

# # #     fig = plt.figure(figsize=(15, 15))
# # #     ax = fig.add_subplot(111, projection="3d")

# # #     AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
# # #     IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
# # #     PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

# # #     X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

# # #     # Compute the 90th percentile value for normalization
# # #     norm_value = np.nanpercentile(high_cloud_amount_mean, 99.8)

# # #     for aod_num in range(AOD_bin):
# # #         Z = aod_num * np.ones_like(X)

# # #         # Plot the 2D pcolormesh color fill map for the current AOD interval
# # #         ax.plot_surface(
# # #             Z,
# # #             X,
# # #             Y,
# # #             rstride=1,
# # #             cstride=1,
# # #             facecolors=plt.cm.Spectral_r(
# # #                 high_cloud_amount_mean[aod_num] / norm_value
# # #             ),
# # #             shade=False,
# # #             edgecolor="none",
# # #             alpha=0.95,
# # #             antialiased=False,
# # #             linewidth=0,
# # #         )

# # #     ax.set_xlabel(xlabel)
# # #     ax.set_ylabel(ylabel)
# # #     ax.set_zlabel(zlabel)

# # #     # Turn off the grid lines
# # #     ax.grid(False)

# # #     # Add color bar
# # #     m = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r)
# # #     m.set_array(high_cloud_amount_mean)
# # #     m.set_clim(np.nanmin(high_cloud_amount_mean), norm_value)
# # #     fig.colorbar(m, shrink=0.3, aspect=9, pad=0.01, label="HCF (%)")

# # #     ax.view_init(elev=20, azim=-60)
# # #     ax.dist = 12

# # #     # Save the figure
# # #     plt.savefig(savefig_str)
# # #     plt.show()
