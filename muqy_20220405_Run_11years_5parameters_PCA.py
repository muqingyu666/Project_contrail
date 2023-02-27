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

    Code for PCA method using pre-combined ERA5 data
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2022-04-05
    
    Including the following parts:
        
        1) Read the Pre-combined ERA5 atmospheric data 
        
        2) Calculate convective instability from those data
        
        3) Form the 1 demensional array from U WIND, VERTICAL VELOCITY, TEMPERATURE, 
        RELATIVE HUMIDITY, CONVECTIVE INSTABILITY
        
        4) Run PCA procedure to get PC1
        
        5) Write PC1 in the nc file
        
"""

import glob
import os

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


class ERA5_parameters_preproccess_PCA(object):
    """
    Read ERA5 parameters from netcdf file and preproccess them in order to generate
    the PC1 array in shape of [month, day, lat, lon]

    No input arguments needed, all function runs in the __call__ method automatically
    after the initialization of the class
    """

    def __init__(
        self,
    ):
        self.ERA_FILE_PATH = "/RAID01/data/Pre_Data_PCA/EAR5_dealt/"
        self.ERA_FILE_NAME = "ERA5_daily_monthly_"

    def __call__(self):
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
                self.ERA_FILE_PATH + self.ERA_FILE_NAME + "*.nc"
            )
            FILE_NAME_ERA = ERA_file[file_num]

            for day in range(0, 28):
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
                ) = self.prepariton_for_PCA(FILE_NAME_ERA, day)

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
                Uwind_1d = np.concatenate(
                    (Uwind_1d, U300_1), axis=0
                )

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

        PC_all = self.Principle_Component_Analysis(
            RelativeH_1d_N,
            Temperature_1d_N,
            Wvelocity_1d_N,
            Stability_1d_N,
            Uwind_1d_N,
        )
        return PC_all

    def prepariton_for_PCA(self, FILE_NAME_ERA, day):
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
        ) = self.read_parameters_from_netcdf(FILE_NAME_ERA)

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
            day, T250, T300, RH300, RH250, W300, U300, Z300, Z250
        )

        stab = self.unstability_calculator(
            RH300_1,
            RH250_1,
            T250_1,
            T300_1,
            Z300_1,
            Z250_1,
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

    def read_parameters_from_netcdf(self, FILE_NAME_ERA):
        # open the file containing the ERA5 data
        file_obj = xr.open_dataset(FILE_NAME_ERA)

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
        self,
        day,
        T250,
        T300,
        RH300,
        RH250,
        W300,
        U300,
        Z300,
        Z250,
    ):
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
        T250_1 = np.array(np.flipud(T250[day, :, :])).reshape(64800)
        T300_1 = np.array(np.flipud(T300[day, :, :])).reshape(64800)
        RH300_1 = np.array(np.flipud(RH300[day, :, :])).reshape(
            64800
        )
        RH250_1 = np.array(np.flipud(RH250[day, :, :])).reshape(
            64800
        )
        W300_1 = np.array(np.flipud(W300[day, :, :])).reshape(64800)
        U300_1 = np.array(np.flipud(U300[day, :, :])).reshape(64800)
        Z300_1 = np.array(np.flipud(Z300[day, :, :])).reshape(64800)
        Z250_1 = np.array(np.flipud(Z250[day, :, :])).reshape(64800)

        # Reshape the variables to 1D arrays
        RH300_1[RH300_1 <= 0] = 0.01
        RH250_1[RH250_1 <= 0] = 0.01

        stab = self.unstability_calculator(
            RH300_1,
            RH250_1,
            T250_1,
            T300_1,
            Z300_1,
            Z250_1,
        )

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
        self,
        RH300_1,
        RH250_1,
        T250_1,
        T300_1,
        Z300_1,
        Z250_1,
    ):
        r300 = np.array(
            mpcalc.mixing_ratio_from_relative_humidity(
                300.0 * units.mbar,
                T300_1 * units.kelvin,
                RH300_1 * units.dimensionless,
            )
        )
        r250 = np.array(
            mpcalc.mixing_ratio_from_relative_humidity(
                250.0 * units.mbar,
                T250_1 * units.kelvin,
                RH250_1 * units.dimensionless,
            )
        )
        Tl300 = 1 / (
            1 / (T300_1 - 55) - (np.log(RH300_1 / 100)) / 2840
        )
        Tl250 = 1 / (
            1 / (T250_1 - 55) - (np.log(RH250_1 / 100)) / 2840
        )
        e300 = np.array(
            mpcalc.vapor_pressure(
                300.0 * units.mbar, r300 * units.dimensionless
            )
        )
        e250 = np.array(
            mpcalc.vapor_pressure(
                250.0 * units.mbar, r250 * units.dimensionless
            )
        )
        sitaDL300 = (
            T300_1
            * (1000 / (300 - e300)) ** 0.2854
            * (T300_1 / Tl300) ** (0.28 * 10 ** (-3) * r300)
        )
        sitaDL250 = (
            T250_1
            * (1000 / (250 - e300)) ** 0.2854
            * (T250_1 / Tl250) ** (0.28 * 10 ** (-3) * r250)
        )
        sitaE300 = sitaDL300 * np.exp(
            (3.036 / Tl300 - 0.00178)
            * r300
            * (1 + 0.448 * 10 ** (-3) * r300)
        )
        sitaE250 = sitaDL250 * np.exp(
            (3.036 / Tl250 - 0.00178)
            * r250
            * (1 + 0.448 * 10 ** (-3) * r250)
        )
        stab = (sitaE300 - sitaE250) / (Z300_1 - Z250_1)
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

    def Principle_Component_Analysis(
        self,
        RelativeH_1d_N,
        Temperature_1d_N,
        Wvelocity_1d_N,
        Stability_1d_N,
        Uwind_1d_N,
    ):
        """
        _summary_

        Parameters
        ----------
        RelativeH_1d_N : np.array
            array of relative humidity values with shape of 1D
        Temperature_1d_N : np.array
            array of temperature values with shape of 1D
        Wvelocity_1d_N : np.array
            array of vertical velocity values with shape of 1D
        Stability_1d_N : np.array
            array of stability values with shape of 1D
        Uwind_1d_N : np.array
            array of Uwind velocity values with shape of 1D

        Returns
        -------
        PC_all : np.array
            array of principle components with shape of 4D: [month, day, lat, lon]
        """
        PC_all = np.zeros((239500800, 5))
        PC_all[:, 0] = RelativeH_1d_N
        PC_all[:, 1] = Temperature_1d_N
        PC_all[:, 2] = Wvelocity_1d_N
        PC_all[:, 3] = Stability_1d_N
        PC_all[:, 4] = Uwind_1d_N

        pca = PCA(n_components=1, whiten=False, copy=False)

        # Print the amount of variance that each PCA component accounts for
        # print(pca.explained_variance_ratio_)
        # pca.fit(PC_all)
        PC_all = pca.fit_transform(PC_all)

        PC_all = stats.zscore(PC_all)

        PC_all = PC_all.reshape(239500800)
        PC_all = PC_all.reshape(132, 28, 180, 360)

        return PC_all


def save_PCA_data_as_netcdf(PC_all):
    """
    save principle components as netcdf file

    Parameters
    ----------
    PC_all : array
        array of principle components with shape of 4D: [month, day, lat, lon]
    """
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    PC_all = PC_all.reshape(132, 28, 180, 360)  # PCA

    ds = xr.Dataset(
        {
            "PC1": (
                ("Month", "Day", "Latitude", "Longitude"),
                PC_all[:, :, :, :],
            ),
        },
        coords={
            "lat": ("Latitude", np.linspace(-90, 89, 180)),
            "lon": ("Longitude", np.linspace(0, 359, 360)),
            "month": ("months", np.linspace(0, 131, 132)),
            "day": ("days", np.linspace(0, 27, 28)),
        },
    )

    os.makedirs("/RAID01/data/PCA_data/", exist_ok=True)
    ds.to_netcdf(
        "/RAID01/data/PCA_data/2010_2020_5_parameters_300hPa_PC1.nc"
    )


# Run this script directly, perform PCA and save data
if __name__ == "__main__":
    # Initialize the class and run all functions
    ERA_PCA = ERA5_parameters_preproccess_PCA()
    PC_all = ERA_PCA()

    # Save data as netcdf
    save_PCA_data_as_netcdf(PC_all)
