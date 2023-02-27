# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:24:59 2021

@author: Mu o(*￣▽￣*)ブ
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

os.makedirs(
    "F:\\ERA5_daily_stored_per_month_global_allyear",
    exist_ok="true",
)


class CombineERA:
    def __init__(self, year, month):
        self.year = year
        self.month = month

    def execute(self):
        year_str = str(self.year).zfill(4)
        month_str = str(self.month).zfill(2)
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
        ##########################################################################################

        ERA_file = glob.glob(
            "H:\\ERA_5_GLOBAL\\"
            + year_str
            + "\\"
            + month_str
            + "\\"
            + "ERA5_T_W_Z_RH__"
            + "*.nc"
        )

        for i in range(0, 28):
            FILE_NAME_ERA = ERA_file[i]

            file_obj = xr.open_dataset(FILE_NAME_ERA)
            lat = file_obj.latitude
            lon = file_obj.longitude
            P = file_obj.level
            Geo = file_obj.z
            RH = file_obj.r
            T = file_obj.t
            W = file_obj.w
            t = file_obj.time

            t_hour[i] = t[0]
            pressure1[i, :] = P[:]
            lat1[i, :] = lat[:]
            lon1[i, :] = lon[:]

            RH = np.nanmean(RH[:, :, :, :], axis=0)
            T = np.nanmean(T[:, :, :, :], axis=0)
            W = np.nanmean(W[:, :, :, :], axis=0)
            Z = np.nanmean(Geo[:, :, :, :], axis=0)

            # for j in range(0,8):
            #     RH[j,:,:] = np.flipud(RH[j,:,:])
            #     T[j,:,:] = np.flipud(T[j,:,:])
            #     W[j,:,:] = np.flipud(W[j,:,:])
            #     Z[j,:,:] = np.flipud(Z[j,:,:])

            RH1[i, :, :, :] = RH[:, :, :]
            Z1[i, :, :, :] = Z[:, :, :]
            T1[i, :, :, :] = T[:, :, :]
            W1[i, :, :, :] = W[:, :, :]

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
        # os.makedirs('G:\\ERA5_daily_stored per month_China\\')
        ds.to_netcdf(
            "F:\\ERA5_daily_stored_per_month_global_allyear\\ERA5_daily_"
            + time_str0
            + ".nc"
        )

    def execute_u(self):
        year_str = str(self.year).zfill(4)
        month_str = str(self.month).zfill(2)
        # day_str=str(day).zfill(2)

        time_str0 = year_str + month_str
        # time_str1 = year_str+month_str+day_str

        ##########################################################################################
        t_hour = np.zeros((28, 24))
        lat1 = np.zeros((28, 181))
        lon1 = np.zeros((28, 360))
        pressure1 = np.zeros((28, 8))

        U1 = np.zeros((28, 8, 181, 360))
        ##########################################################################################

        ERA_file = glob.glob(
            "H:\\ERA5_u\\"
            + year_str
            + "\\"
            + month_str
            + "\\"
            + "ERA5_Uwind__"
            + "*.nc"
        )

        for i in range(0, 28):
            FILE_NAME_ERA = ERA_file[i]

            file_obj = xr.open_dataset(FILE_NAME_ERA)
            lat = file_obj.latitude
            lon = file_obj.longitude
            P = file_obj.level
            U = file_obj.u
            t = file_obj.time

            t_hour[i] = t[0]
            pressure1[i, :] = P[:]
            lat1[i, :] = lat[:]
            lon1[i, :] = lon[:]

            U = np.nanmean(U[:, :, :, :], axis=0)

            # for j in range(0,8):
            #     RH[j,:,:] = np.flipud(RH[j,:,:])
            #     T[j,:,:] = np.flipud(T[j,:,:])
            #     W[j,:,:] = np.flipud(W[j,:,:])
            #     Z[j,:,:] = np.flipud(Z[j,:,:])

            U1[i, :, :, :] = U[:, :, :]

        ds = xr.Dataset(
            {
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
        # os.makedirs('G:\\ERA5_daily_stored per month_China\\')
        ds.to_netcdf(
            "F:\\ERA5_daily_stored_per_month_global_allyear\\ERA5_daily_"
            + time_str0
            + ".nc"
        )


cb = CombineERA()
for i in range(2010, 2021):
    for j in range(0, 13):
        CombineERA.execute(i, j)
        CombineERA.execute_u(i, j)

A_N10 = np.zeros((1))
A_N11 = np.zeros((1))
A_N12 = np.zeros((1))
A_N13 = np.zeros((1))

for i in range(0, 66):

    # year_str=str(2010).zfill(4)
    # month_str=str(1).zfill(2)
    # time_str0=year_str+month_str

    ERA_file = glob.glob(
        "F:\\ERA5_daily_stored per month_global_3\\ERA5_daily_"
        + "*.nc"
    )
    FILE_NAME_ERA = ERA_file[i]

    for j in range(0, 28):

        file_obj = xr.open_dataset(FILE_NAME_ERA)

        lat = file_obj.lat
        lon = file_obj.lon
        P = file_obj.level
        z = file_obj.Geo
        RH = file_obj.RH
        T = file_obj.T
        W = file_obj.W

        T250_1 = np.flipud(np.delete(T[:, 6, :, :], 0, axis=1))[
            j, :, :
        ].reshape(64800)
        T300_1 = np.flipud(np.delete(T[:, 7, :, :], 0, axis=1))[
            j, :, :
        ].reshape(64800)
        RH300_1 = np.flipud(np.delete(RH[:, 7, :, :], 0, axis=1))[
            j, :, :
        ].reshape(64800)
        RH250_1 = np.flipud(np.delete(RH[:, 6, :, :], 0, axis=1))[
            j, :, :
        ].reshape(64800)
        W_1 = np.flipud(np.delete(W[:, 7, :, :], 0, axis=1))[
            j, :, :
        ].reshape(64800)
        Z300_1 = np.flipud(np.delete(z[:, 7, :, :], 0, axis=1))[
            j, :, :
        ].reshape(64800)
        Z250_1 = np.flipud(np.delete(z[:, 6, :, :], 0, axis=1))[
            j, :, :
        ].reshape(64800)

        RH300_1[RH300_1 <= 0] = 0.01
        RH250_1[RH250_1 <= 0] = 0.01

        # Calculate sitaE using lifting condensation temperature,the equation is on the paper Bolton (1980)
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

        A_N10 = np.concatenate((A_N10, RH300_1), axis=0)
        A_N11 = np.concatenate((A_N11, T300_1), axis=0)
        A_N12 = np.concatenate((A_N12, W_1), axis=0)
        A_N13 = np.concatenate((A_N13, stab), axis=0)

A_N10 = np.delete(A_N10, 0, axis=0)  # RH
A_N11 = np.delete(A_N11, 0, axis=0)  # T
A_N12 = np.delete(A_N12, 0, axis=0)  # W
A_N13 = np.delete(A_N13, 0, axis=0)  # stability sita/z
