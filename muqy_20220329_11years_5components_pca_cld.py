import calendar
import glob
import os
import time
import tkinter
from itertools import product

import cartopy.crs as ccrs
import matplotlib
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from metpy.units import units
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import norm, zscore
from sklearn.decomposition import PCA


def dcmap(file_path):
    """
    Color control

    Parameters
    ----------
    file_path : str
        Path of the color file to be read.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
    """
    fid = open(file_path)
    data = fid.readlines()
    n = len(data)
    rgb = np.zeros((n, 3))
    for i in np.arange(n):
        rgb[i][0] = data[i].split(",")[0]
        rgb[i][1] = data[i].split(",")[1]
        rgb[i][2] = data[i].split(",")[2]
        rgb[i] = rgb[i] / 255.0
        icmap = mpl.colors.ListedColormap(rgb, name="my_color")
    return icmap


def execute(year, month):
    """
    perform the main function to process the ERS5 data

    Parameters
    ----------
    year : int
        year of the data
    month : int
        month of the data
    """
    year_str = str(year).zfill(4)
    month_str = str(month).zfill(2)
    # day_str=str(day).zfill(2)

    time_str0 = year_str + month_str
    # time_str1 = year_str+month_str+day_str

    ##########################################################################################
    t_hour = np.zeros((28, 24))
    lat1 = np.zeros((28, 181))
    lon1 = np.zeros((28, 360))
    pressure1 = np.zeros((28, 8))

    RH1 = np.zeros((28, 8, 181, 360))
    T1 = np.zeros((28, 8, 181, 360))
    W1 = np.zeros((28, 8, 181, 360))
    Z1 = np.zeros((28, 8, 181, 360))
    U1 = np.zeros((28, 8, 181, 360))
    ##########################################################################################

    ERA_file = glob.glob(
        "/RAID01/data/muqy/EAR5/"
        + year_str
        + "/"
        + month_str
        + "/"
        + "ERA5_T_W_Z_RH__"
        + "*.nc"
    )
    ERA_file_U = glob.glob(
        "/RAID01/data/muqy/ERA5_u/"
        + year_str
        + "/"
        + month_str
        + "/"
        + "ERA5_Uwind__"
        + "*.nc"
    )

    for i in range(0, 28):
        FILE_NAME_ERA = ERA_file[i]
        FILE_NAME_ERA_U = ERA_file_U[i]

        file_obj = xr.open_dataset(FILE_NAME_ERA)
        uwind_file_obj = xr.open_dataset(FILE_NAME_ERA_U)

        lat = np.array(file_obj.latitude)
        lon = np.array(file_obj.longitude)

        U = np.array(uwind_file_obj.u)
        P = np.array(file_obj.level)
        Geo = np.array(file_obj.z)
        RH = np.array(file_obj.r)
        T = np.array(file_obj.t)
        W = np.array(file_obj.w)
        t = np.array(file_obj.time)

        t_hour[i] = t[0]
        pressure1[i, :] = P[:]
        lat1[i, :] = lat[:]
        lon1[i, :] = lon[:]

        RH = np.nanmean(RH[:, :, :, :], axis=0)
        T = np.nanmean(T[:, :, :, :], axis=0)
        W = np.nanmean(W[:, :, :, :], axis=0)
        Z = np.nanmean(Geo[:, :, :, :], axis=0)
        U = np.nanmean(U[:, :, :, :], axis=0)

        RH1[i, :, :, :] = RH[:, :, :]
        Z1[i, :, :, :] = Z[:, :, :]
        T1[i, :, :, :] = T[:, :, :]
        W1[i, :, :, :] = W[:, :, :]
        U1[i, :, :, :] = U[:, :, :]

    os.makedirs(
        "/RAID01/data/muqy_python/EAR5_dealt/", exist_ok="true"
    )

    ds = xr.Dataset(
        {
            "RH": (
                ("Time", "Level", "Latitude", "Longitude"),
                RH1[:, :, :, :],
            ),
            "Geo": (
                ("Time", "Level", "Latitude", "Longitude"),
                Z1[:, :, :, :],
            ),
            "T": (
                ("Time", "Level", "Latitude", "Longitude"),
                T1[:, :, :, :],
            ),
            "W": (
                ("Time", "Level", "Latitude", "Longitude"),
                W1[:, :, :, :],
            ),
            "U": (
                ("Time", "Level", "Latitude", "Longitude"),
                U1[:, :, :, :],
            ),
        },
        coords={
            "lat": ("Latitude", lat[:]),
            "lon": ("Longitude", lon[:]),
            "time": pd.date_range(
                year_str + "-" + month_str + "-01",
                periods=28,
                normalize=True,
            ),
            "level": ("Level", pressure1[0, :]),
        },
    )
    ds.to_netcdf(
        "/RAID01/data/muqy_python/EAR5_dealt/ERA5_daily_monthly_"
        + time_str0
        + ".nc"
    )


for i in range(2010, 2021):
    for j in range(1, 13):
        execute(i, j)


class ERA5_parameters_preproccess(object):
    def __init__(self, FILE_NAME_ERA, day):
        """
        initialize the class

        Parameters
        ----------
        FILE_NAME_ERA : str
            ERA5 file name
        day : int
            day of the month
        """
        self.FILE_NAME_ERA = FILE_NAME_ERA
        self.day = day

    def __call__(self):
        """
        Call all moudle

        Returns
        -------
        ERA5_parameters_preproccess object
        every parameters! ready to go!
        """
        (
            lat,
            lon,
            P,
            T,
            W,
            U,
            T250,
            T300,
            RH300,
            RH250,
            W300,
            U300,
            Z300,
            Z250,
        ) = self.read_parameters_from_netcdf()

        (
            RH300_1,
            RH250_1,
            T250_1,
            T300_1,
            W300_1,
            U300_1,
            Z300_1,
            Z250_1,
        ) = self.preproccess_parameters(
            T250, T300, RH300, RH250, W300, U300, Z300, Z250
        )

        stab = self.unstability_calculator(
            RH300_1, RH250_1, T250_1, T300_1, Z300_1, Z250_1,
        )
        return (
            stab,
            RH300_1,
            RH250_1,
            T250_1,
            T300_1,
            W300_1,
            U300_1,
            Z300_1,
            Z250_1,
        )

    def read_parameters_from_netcdf(self):
        """
        Read ERA5 parameters from netcdf file

        Returns
        -------
        lat, lon, P, T, W, U, T250, T300, RH300, RH250, W300, U300, Z300, Z250
        """
        # open the file containing the ERA5 data
        file_obj = xr.open_dataset(self.FILE_NAME_ERA)

        # extract the atmopheric variables
        lat = file_obj.lat
        lon = file_obj.lon
        P = file_obj.level
        z = file_obj.Geo

        RH = file_obj.RH
        T = file_obj.T
        W = file_obj.W
        U = file_obj.U

        # extract the atmospheric variables at 300 hPa & 250 hPa
        T250 = T[:, 6, :, :]
        T300 = T[:, 7, :, :]
        RH300 = RH[:, 7, :, :]
        RH250 = RH[:, 6, :, :]
        W300 = W[:, 7, :, :]
        U300 = U[:, 7, :, :]
        Z300 = z[:, 7, :, :]
        Z250 = z[:, 6, :, :]

        return (
            lat,
            lon,
            P,
            T,
            W,
            U,
            T250,
            T300,
            RH300,
            RH250,
            W300,
            U300,
            Z300,
            Z250,
        )

    def preproccess_parameters(
        self, T250, T300, RH300, RH250, W300, U300, Z300, Z250
    ):
        """
        This function preproccess the ERA5 data to be used in the

        Parameters
        ----------
        T250 : numpy.ndarray
            The temperature at 250 hPa
        T300 : numpy.ndarray
            The temperature at 300 hPa
        RH300 : numpy.ndarray
            The relative humidity at 300 hPa
        RH250 : numpy.ndarray
            The relative humidity at 250 hPa
        W300 : numpy.ndarray
            The vertical velocity at 300 hPa
        U300 : numpy.ndarray
            The U wind speed at 300 hPa
        Z300 : numpy.ndarray
            The geopotential height at 300 hPa
        Z250 : numpy.ndarray
            The geopotential height at 250 hPa

        Returns
        -------
        1D numpy.ndarray of all parameters
        """
        # delete the variables that are not needed
        # (era5 resolution is 181 x 360 --> 180 x 360)
        T250 = np.delete(T250, 0, axis=1)
        T300 = np.delete(T300, 0, axis=1)
        RH300 = np.delete(RH300, 0, axis=1)
        RH250 = np.delete(RH250, 0, axis=1)
        W300 = np.delete(W300, 0, axis=1)
        U300 = np.delete(U300, 0, axis=1)
        Z300 = np.delete(Z300, 0, axis=1)
        Z250 = np.delete(Z250, 0, axis=1)

        # flip the variables to have the same orientation as the CERES data
        # and reshape the variables to 1D arrays
        T250_1 = np.array(
            np.flipud(T250[self.day, :, :])
        ).reshape(64800)
        T300_1 = np.array(
            np.flipud(T300[self.day, :, :])
        ).reshape(64800)
        RH300_1 = np.array(
            np.flipud(RH300[self.day, :, :])
        ).reshape(64800)
        RH250_1 = np.array(
            np.flipud(RH250[self.day, :, :])
        ).reshape(64800)
        W300_1 = np.array(
            np.flipud(W300[self.day, :, :])
        ).reshape(64800)
        U300_1 = np.array(
            np.flipud(U300[self.day, :, :])
        ).reshape(64800)
        Z300_1 = np.array(
            np.flipud(Z300[self.day, :, :])
        ).reshape(64800)
        Z250_1 = np.array(
            np.flipud(Z250[self.day, :, :])
        ).reshape(64800)

        # Reshape the variables to 1D arrays
        RH300_1[RH300_1 <= 0] = 0.01
        RH250_1[RH250_1 <= 0] = 0.01

        return (
            RH300_1,
            RH250_1,
            T250_1,
            T300_1,
            W300_1,
            U300_1,
            Z300_1,
            Z250_1,
        )

    def unstability_calculator(
        self, RH300_1, RH250_1, T250_1, T300_1, Z300_1, Z250_1,
    ):
        """
        Calculate the convective unstability of the atmosphere

        Parameters
        ----------
        RH300_1 : numpy.ndarray
            Relative humidity at 300 hPa
        RH250_1 : numpy.ndarray
            Relative humidity at 250 hPa
        T250_1 : numpy.ndarray
            Temperature at 250 hPa
        T300_1 : numpy.ndarray
            Temperature at 300 hPa
        Z300_1 : numpy.ndarray
            Height of the 300 hPa layer
        Z250_1 : numpy.ndarray
            Height of the 250 hPa layer

        Returns
        -------
        stab : numpy.ndarray
            Convective unstability of the atmosphere 250-300 hPa
        """
        # r300 = np.array(
        #     mpcalc.mixing_ratio_from_relative_humidity(
        #         300.0 * units.mbar,
        #         T300_1 * units.kelvin,
        #         RH300_1 * units.dimensionless,
        #     )
        # )
        # r250 = np.array(
        #     mpcalc.mixing_ratio_from_relative_humidity(
        #         250.0 * units.mbar,
        #         T250_1 * units.kelvin,
        #         RH250_1 * units.dimensionless,
        #     )
        # )
        # Tl300 = 1 / (
        #     1 / (T300_1 - 55) - (np.log(RH300_1 / 100)) / 2840
        # )
        # Tl250 = 1 / (
        #     1 / (T250_1 - 55) - (np.log(RH250_1 / 100)) / 2840
        # )
        # e300 = np.array(
        #     mpcalc.vapor_pressure(
        #         300.0 * units.mbar, r300 * units.dimensionless
        #     )
        # )
        # e250 = np.array(
        #     mpcalc.vapor_pressure(
        #         250.0 * units.mbar, r250 * units.dimensionless
        #     )
        # )
        # sitaDL300 = (
        #     T300_1
        #     * (1000 / (300 - e300)) ** 0.2854
        #     * (T300_1 / Tl300) ** (0.28 * 10 ** (-3) * r300)
        # )
        # sitaDL250 = (
        #     T250_1
        #     * (1000 / (250 - e300)) ** 0.2854
        #     * (T250_1 / Tl250) ** (0.28 * 10 ** (-3) * r250)
        # )
        # sitaE300 = sitaDL300 * np.exp(
        #     (3.036 / Tl300 - 0.00178)
        #     * r300
        #     * (1 + 0.448 * 10 ** (-3) * r300)
        # )
        # sitaE250 = sitaDL250 * np.exp(
        #     (3.036 / Tl250 - 0.00178)
        #     * r250
        #     * (1 + 0.448 * 10 ** (-3) * r250)
        # )
        # stab = (sitaE300 - sitaE250) / (Z300_1 - Z250_1)
        dewpoint300 = np.array(
            mpcalc.dewpoint_from_relative_humidity(
                T300_1 * units.kelvin,
                RH300_1 * units.dimensionless,
            )
        )
        dewpoint250 = np.array(
            mpcalc.dewpoint_from_relative_humidity(
                T250_1 * units.kelvin,
                RH250_1 * units.dimensionless,
            )
        )
        sitaE300 = np.array(
            mpcalc.equivalent_potential_temperature(
                300.0 * units.mbar,
                T300_1 * units.kelvin,
                dewpoint300 * units.degree_Celsius,
            )
        )
        sitaE250 = np.array(
            mpcalc.equivalent_potential_temperature(
                250.0 * units.mbar,
                T250_1 * units.kelvin,
                dewpoint250 * units.degree_Celsius,
            )
        )
        stab = (sitaE300 - sitaE250) / (Z300_1 - Z250_1)

        return stab


# Make auxiliary arrays for atmospheric variables
RelativeH_1d = np.zeros((1))
Temperature_1d = np.zeros((1))
Wvelocity_1d = np.zeros((1))
Stability_1d = np.zeros((1))
Uwind_1d = np.zeros((1))

# Start main loop (file loop)
for file_num in range(0, 132):

    # year_str=str(2010).zfill(4)
    # month_str=str(1).zfill(2)
    # time_str0=year_str+month_str

    # auxiliar loop for the number of days in the month (28 days a month)
    ERA_file = glob.glob(
        "/RAID01/data/muqy_python/EAR5_dealt/ERA5_daily_monthly_"
        + "*.nc"
    )
    FILE_NAME_ERA = ERA_file[file_num]

    for day in range(0, 28):

        data_preproccess = ERA5_parameters_preproccess(
            FILE_NAME_ERA, day
        )
        (
            stab,
            RH300_1,
            RH250_1,
            T250_1,
            T300_1,
            W300_1,
            U300_1,
            Z300_1,
            Z250_1,
        ) = data_preproccess()

        # Concatenate all 11 years data into 1D array
        RelativeH_1d = np.concatenate(
            (RelativeH_1d, RH300_1), axis=0
        )
        Temperature_1d = np.concatenate(
            (Temperature_1d, T300_1), axis=0
        )
        Wvelocity_1d = np.concatenate(
            (Wvelocity_1d, W300_1), axis=0
        )
        Stability_1d = np.concatenate(
            (Stability_1d, stab), axis=0
        )
        Uwind_1d = np.concatenate((Uwind_1d, U300_1), axis=0)


# Delete the first auxiliary data and perform Zscore normalization
RelativeH_1d_N = stats.zscore(
    np.delete(RelativeH_1d, 0, axis=0)
)  # RH
Temperature_1d_N = stats.zscore(
    np.delete(Temperature_1d, 0, axis=0)
)  # T
Wvelocity_1d_N = stats.zscore(
    np.delete(Wvelocity_1d, 0, axis=0)
)  # W
Stability_1d_N = stats.zscore(
    np.delete(Stability_1d, 0, axis=0)
)  # stability sita/z
Uwind_1d_N = stats.zscore(
    np.delete(Uwind_1d, 0, axis=0)
)  # U wind

