#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\b
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

    Code to perform 3d spatial distribution of cld data constrained by IWP,AOD,PC1
        Or maybe just plot the 2d spatial distribution of 2020-(2017~2019) and let PS do whats left
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2023-04-20
    
    Including the following parts:
        
        1) Read the PC1 and atmospheric variables data
        
        2) Plot the correlation between PC1 and atmospheric variables
        
"""

# import modules
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

# ---------- Read PCA&CLD data from netcdf file --------

# ---------- Read in Cloud area data ----------
# now we read IWP and other cld data (not IWP) from netcdf file
# triout 1 : Use lattitudinal band from 20N to 50N

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
) = read_PC1_CERES_20_50_lat_band_from_netcdf(
    CERES_Cld_dataset_name="Cldarea"
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

# use the 2010-2020 PC1 only
PC_all = PC_all[-11:]

# Extract the PC1 and Cldarea data for each year
def extract_PC1_CERES_each_year(
    PC_years: dict[int, np.ndarray],
    Cld_years: dict[int, np.ndarray],
    IWP_years: dict[int, np.ndarray],
) -> None:
    """ Here is the explanation for the code above:
    1. We loop over the years from 2017 to 2020, and extract the PC1 data for each year. We assign this data to
    the global variable 'PC_{year}', where {year} is the year we are currently looping over. Note that we
    reshape the data to have the shape (n, 30, 360), where n is the number of days in that year.
    2. We loop over the years from 2017 to 2020, and extract the CERES data for each year. We assign this data to
    the global variable 'Cld_{year}', where {year} is the year we are currently looping over. Note that we
    reshape the data to have the shape (n, 30, 360), where n is the number of days in that year.
    3. We loop over the years from 2017 to 2020, and extract the IWP data for each year. We assign this data to
    the global variable 'IWP_{year}', where {year} is the year we are currently looping over. Note that we
    reshape the data to have the shape (n, 30, 360), where n is the number of days in that year. """
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
        globals()[f"PC_{year}"] = PC_years[year].reshape(-1, 30, 360)

    for year in range(2017, 2021):
        globals()[f"Cld_{year}"] = Cld_years[year].reshape(-1, 30, 360)

    for year in range(2017, 2021):
        globals()[f"IWP_{year}"] = IWP_years[year].reshape(-1, 30, 360)

extract_PC1_CERES_each_year(PC_years, Cld_years, IWP_years)

#########################################
##### start seperate time test ##########
#########################################

# Implementation for MERRA2 dust AOD
# extract the data from 2010 to 2014 like above
data_merra2_2010_2020_new_lon = xr.open_dataset(
    "/RAID01/data/merra2/merra_2_daily_2010_2020_new_lon.nc"
)

# Extract Dust aerosol data for 2020 and 2017-2019
Dust_AOD_2020 = (
    data_merra2_2010_2020_new_lon["DUEXTTAU"]
    .sel(lat=slice(21, 50))
    .sel(time=slice("2020", "2020"))
    .values
)
Dust_AOD_2017_2019 = (
    data_merra2_2010_2020_new_lon["DUEXTTAU"]
    .sel(lat=slice(21, 50))
    .sel(time=slice("2017", "2019"))
    .values
)

# Concatenate the data for PC as 2017-2019
PC_2017_2019 = np.concatenate([PC_2017, PC_2018, PC_2019], axis=0)
# Do so for Cldarea data
Cld_2017_2019 = np.concatenate([Cld_2017, Cld_2018, Cld_2019], axis=0)
# Do so for IWP data
IWP_2017_2019 = np.concatenate([IWP_2017, IWP_2018, IWP_2019], axis=0)

# ------------------------------------------------------
# ---------- Read the filtered data out ----------------
# ------------------------------------------------------

def read_filtered_data_out(
    file_name: str = "Cld_match_PC_gap_IWP_AOD_constrain_mean_2010_2020.nc",
):
    """_summary_

    Parameters
    ----------
    file_name : str, optional
        _description_, by default "Cld_match_PC_gap_IWP_AOD_constrain_mean_2010_2020.nc"

    Returns
    -------
    _type_
        _description_
    """    
    Cld_match_PC_gap_IWP_AOD_constrain_mean = xr.open_dataarray(
        "/RAID01/data/Filtered_data/" + file_name
    )

    return Cld_match_PC_gap_IWP_AOD_constrain_mean


# Read the filtered data
# Read the Dust_AOD constrain data for 2020 and 2017-2019 data
# The data shape is (AOD gap, IWP gap, PC gap, lat, lon)
# And the lat is only for 20N~50N,, the lon is for 0~360
Cld_match_PC_gap_IWP_AOD_constrain_mean_2020 = read_filtered_data_out(
    file_name="Cld_match_PC_gap_IWP_AOD_constrain_mean_2020_filter_extreme_AOD_IWP_IPR_Cldicerad.nc"
)

Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019 = read_filtered_data_out(
    file_name="Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019_filter_extreme_AOD_IWP_IPR_Cldicerad.nc"
)

# --------------------------------- Extract the data ----------------------------------------
# The data shape is (AOD gap, IWP gap, PC gap, lat, lon)
# Now is 4, 30, 30, 30, 360
# We want to extract data for specific AOD gap, IWP gap, PC gap
# For example, AOD gap = 1-2, IWP gap = 1, PC gap = 1-5
# The data shape will be (lat, lon)
# --------------------------------- Extract the data ----------------------------------------
# Select the specific data
# This code is used to extract data from the data frame Cld_match_PC_gap_IWP_AOD_constrain_mean_2020 and Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
# which is used to create the data frame Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2020.
# This code is used in the function filter_data, which is used in the function get_data.
def extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start: int = 1,
    AOD_gap_end: int = 2,
    IWP_gap_start: int = 1,
    IWP_gap_end: int = 2,
    PC_gap_start: int = 1,
    PC_gap_end: int = 5,
):
    """Extract the data from the Cld_match_PC_gap_IWP_AOD_constrain_mean_2020 and Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019
    arrays that correspond to the AOD gap between 1 and 2, IWP gap of 1, and PC gap between 1 and 5. Calculate the mean of this data
    for both the 2017-2019 and 2020 time periods, ignoring NaN values.

    Parameters
    ----------
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020 : numpy.ndarray
        Array containing the mean of the PC gap between 1 and 5, IWP gap of 1, and AOD gap between 1 and 2, for the year 2020.
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019 : numpy.ndarray
        Array containing the mean of the PC gap between 1 and 5, IWP gap of 1, and AOD gap between 1 and 2, for the years 2017
        through 2019.

    Returns
    -------
    mean_2017_2019 : float
        Mean of the Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019 array, ignoring NaN values.
    mean_2020 : float
        Mean of the Cld_match_PC_gap_IWP_AOD_constrain_mean_2020 array, ignoring NaN values.
    """

    # Extract the data
    data_anormaly = (
        Cld_match_PC_gap_IWP_AOD_constrain_mean_2020
        - Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019
    )
    data_anormaly_specific = data_anormaly[
        AOD_gap_start:AOD_gap_end,
        IWP_gap_start:IWP_gap_end,
        PC_gap_start:PC_gap_end,
    ]

    # Calculate the mean while ignoring NaN values
    mean_anormaly_specific = np.nanmean(
        data_anormaly_specific, axis=(0, 1, 2)
    )

    return mean_anormaly_specific

# Extract the HCF data
# No AOD constrain
mean_anormaly_specific_IWP_0 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=0,
    IWP_gap_end=9,
    PC_gap_start=0,
    PC_gap_end=15,
)
mean_anormaly_specific_IWP_1 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=2,
    IWP_gap_end=4,
    PC_gap_start=0,
    PC_gap_end=30,
)
mean_anormaly_specific_IWP_2 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2, 
    IWP_gap_start=4,
    IWP_gap_end=6,
    PC_gap_start=0,
    PC_gap_end=30,
)

# Low AOD scenario
mean_anormaly_specific_IWP_0 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=0,
    IWP_gap_end=2,
    PC_gap_start=0,
    PC_gap_end=15,
)
mean_anormaly_specific_IWP_1 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=2,
    IWP_gap_end=4,
    PC_gap_start=0,
    PC_gap_end=15,
)
mean_anormaly_specific_IWP_2 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2, 
    IWP_gap_start=4,
    IWP_gap_end=6,
    PC_gap_start=0,
    PC_gap_end=15,
)

# Hiigh AOD scenario
mean_anormaly_specific_IWP_0 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=4,
    AOD_gap_end=6,
    IWP_gap_start=14,
    IWP_gap_end=17,
    PC_gap_start=24,
    PC_gap_end=30,
)
mean_anormaly_specific_IWP_1 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=4,
    AOD_gap_end=6,
    IWP_gap_start=17,
    IWP_gap_end=20,
    PC_gap_start=24,
    PC_gap_end=30,
)
mean_anormaly_specific_IWP_2 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=4,
    AOD_gap_end=6, 
    IWP_gap_start=20,
    IWP_gap_end=23,
    PC_gap_start=24,
    PC_gap_end=30,
)


# ----------------------------------------------------------------------
# Extract the IPR data
# All IWP scenario
mean_anormaly_specific_IWP_0 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=0,
    IWP_gap_end=6,
    PC_gap_start=0,
    PC_gap_end=20,
)


# Low AOD scenario only
mean_anormaly_specific_IWP_0 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=0,
    IWP_gap_end=2,
    PC_gap_start=0,
    PC_gap_end=20,
)
mean_anormaly_specific_IWP_1 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=2,
    IWP_gap_end=4,
    PC_gap_start=0,
    PC_gap_end=20,
)
mean_anormaly_specific_IWP_2 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=4,
    IWP_gap_end=6,
    PC_gap_start=0,
    PC_gap_end=20,
)

# ---------------------------------------------------------------------
# Extract the specified CEH data
# Low AOD scenario
mean_anormaly_specific_IWP_0 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=4,
    IWP_gap_start=0,
    IWP_gap_end=4,
    PC_gap_start=0,
    PC_gap_end=21,
)

# no constraint scenario
mean_anormaly_specific_IWP_0 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=6,
    IWP_gap_start=0,
    IWP_gap_end=30,
    PC_gap_start=0,
    PC_gap_end=31,
)

# High AOD scenario
mean_anormaly_specific_IWP_1 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=5,
    AOD_gap_end=6,
    IWP_gap_start=16,
    IWP_gap_end=22,
    PC_gap_start=25,
    PC_gap_end=30,
)

mean_anormaly_specific_IWP_1 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=5,
    IWP_gap_end=10,
    PC_gap_start=6,
    PC_gap_end=20,
)
mean_anormaly_specific_IWP_2 = extract_specific_filtered_data(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2020,
    Cld_match_PC_gap_IWP_AOD_constrain_mean_2017_2019,
    AOD_gap_start=0,
    AOD_gap_end=2,
    IWP_gap_start=10,
    IWP_gap_end=15,
    PC_gap_start=6,
    PC_gap_end=20,
)

# Now the data shape is (lat, lon)
# We want to plot a 20N~50N, 0~360 map showing cld data 2020-(2017~2019)

########################################################################################
########################################################################################
### Start spatial distribution plot ####################################################
########################################################################################
########################################################################################

# We want to plot a 20N~50N, 0~360 map showing cld data 2020-(2017~2019)
# The AOD and IWP and PC1 gap range are selected visially

# Plot global spatial distribution of a variable
def plot_global_spatial_distribution(
    data: np.ndarray,
    vmin: float,
    vmax: float,
    var_name: str,
    title: str,
    cmap: str = "RdBu_r",
):
    """Plot global spatial distribution of a variable.

    Parameters
    ----------
    data : np.ndarray
        2D numpy array of shape (lat, lon) containing the variable to plot.
    var_name : str
        Name of the variable to plot.
    title : str
        Title of the plot.
    cmap : str, optional
        Colormap to use, by default "RdBu_r"

    """
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(20, 49, 30)

    # Create custom colormap
    cmap = mpl.colormaps[cmap]

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12.5, 3),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
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
        shrink=0.5,
        extend="both",
    )
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

# ---------------------------------------------------------------------
# Plot the HCF anormaly data
# Higher IWP constraint
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_0,
    vmin=-1,
    vmax=1,
    var_name="HCF",
    title="2020 HCF Anormaly IWP Low",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_1,
    vmin=-1,
    vmax=1,
    var_name="HCF",
    title="2020 HCF Anormaly IWP Mid",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_2,
    vmin=-1,
    vmax=1,
    var_name="HCF",
    title="2020 HCF Anormaly IWP High",
)

# Low AOD scenario
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_0,
    vmin=-0.6,
    vmax=0.6,
    var_name="HCF",
    title="2020 HCF Anormaly IWP Low",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_1,
    vmin=-0.6,
    vmax=0.6,
    var_name="HCF",
    title="2020 HCF Anormaly IWP Mid",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_2,
    vmin=-0.6,
    vmax=0.6,
    var_name="HCF",
    title="2020 HCF Anormaly IWP High",
)

# High AOD scenario
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_0,
    vmin=-9,
    vmax=9,
    var_name="HCF",
    title="2020 HCF Anormaly IWP Low",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_1,
    vmin=-9,
    vmax=9,
    var_name="HCF",
    title="2020 HCF Anormaly IWP Mid",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_2,
    vmin=-9,
    vmax=9,
    var_name="HCF",
    title="2020 HCF Anormaly IWP High",
)

# ----------------------------------------------
# Plot the Icerad anormaly data
# No constrain scenario
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_0,
    vmin=-3,
    vmax=3,
    var_name="IPR",
    title="2020 IPR Anormaly IWP Low",
)

plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_0,
    vmin=-5,
    vmax=5,
    var_name="IPR",
    title="2020 IPR Anormaly IWP Low",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_1,
    vmin=-5,
    vmax=5,
    var_name="IPR",
    title="2020 IPR Anormaly IWP Mid",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_2,
    vmin=-5,
    vmax=5,
    var_name="IPR",
    title="2020 IPR Anormaly IWP High",
)

# ----------------------------------------------
# Plot the CEH anormaly data
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_0,
    vmin=-2.2,
    vmax=2.2,
    var_name="CEH",
    title="2020 CEH Anormaly AOD Low",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_1,
    vmin=-1,
    vmax=1,
    var_name="CEH",
    title="2020 CEH Anormaly AOD High",
)
plot_global_spatial_distribution(
    data=mean_anormaly_specific_IWP_2,
    vmin=-2.5,
    vmax=2.5,
    var_name="CEH",
    title="2020 CEH Anormaly",
)
