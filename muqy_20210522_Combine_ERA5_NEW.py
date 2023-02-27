# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:25:12 2021

@author: Mu o(*￣▽￣*)ブ
"""

import xarray as xr
import numpy as np
import pandas as pd
import glob
import os

# os.makedirs('G:\\ERA5_daily_stored per month\\')


def execute(year, month):
    year_str = str(year).zfill(4)
    month_str = str(month).zfill(2)
    # day_str=str(day).zfill(2)

    time_str0 = year_str + month_str
    # time_str1 = year_str+month_str+day_str

    ##########################################################################################
    t1 = np.zeros((28, 24))
    lat1 = np.zeros((28, 81))
    lon1 = np.zeros((28, 141))
    p1 = np.zeros((28, 8))

    RH1 = np.zeros((28, 8, 81, 141))
    T1 = np.zeros((28, 8, 81, 141))
    W1 = np.zeros((28, 8, 81, 141))
    Z1 = np.zeros((28, 8, 81, 141))
    ##########################################################################################

    ERA_file = glob.glob(
        "G:\\ERA5_1\\"
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

        t1[i] = t[0]
        p1[i, :] = P[:]
        lat1[i, :] = lat[:]
        lon1[i, :] = lon[:]

        RH = np.mean(RH[:, :, :, :], axis=0)
        T = np.mean(T[:, :, :, :], axis=0)
        W = np.mean(W[:, :, :, :], axis=0)
        Z = np.mean(Geo[:, :, :, :], axis=0)

        for j in range(0, 8):
            RH[j, :, :] = np.flipud(RH[j, :, :])
            T[j, :, :] = np.flipud(T[j, :, :])
            W[j, :, :] = np.flipud(W[j, :, :])
            Z[j, :, :] = np.flipud(Z[j, :, :])

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
            "lat": ("Latitude", np.arange(0, 81, 1)),
            "lon": ("Longitude", np.arange(40, 181, 1)),
            "time": pd.date_range(
                year_str + "-" + month_str + "-01",
                periods=28,
                normalize=True,
            ),
            "level": ("Level", p1[0, :]),
        },
    )
    # os.makedirs('G:\\ERA5_daily_stored per month_China\\')
    ds.to_netcdf(
        "G:\\ERA5_daily_stored per month_China_3\\ERA5_daily_"
        + time_str0
        + ".nc"
    )


for i in range(2011, 2021):
    for j in range(2, 5):
        execute(i, j)

