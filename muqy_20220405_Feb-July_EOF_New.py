# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:57:19 2021

@author: Mu o(*￣▽￣*)ブ
"""

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats import zscore
import calendar
from sklearn.decomposition import PCA
import glob
import pandas as pd
import os
import metpy.calc as mpcalc
from metpy.units import units
import seaborn as sns
import matplotlib.colors as colors
from scipy.stats import norm
import time
from itertools import product
import math


def dcmap(file_path):
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


########################### ERA5 PCA ######################################

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
    # ERA_file = glob.glob(
    #     "/RAID01/data/muqy/EAR5_dealt/ERA5_daily_monthly_" + "*.nc"
    # )

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
        T250 = T[:, 6, :, :]
        T300 = T[:, 7, :, :]
        RH300 = RH[:, 7, :, :]
        RH250 = RH[:, 6, :, :]
        W = W[:, 7, :, :]
        Z300 = z[:, 7, :, :]
        Z250 = z[:, 6, :, :]

        T250 = np.delete(T250, 0, axis=1)
        T300 = np.delete(T300, 0, axis=1)
        RH300 = np.delete(RH300, 0, axis=1)
        RH250 = np.delete(RH250, 0, axis=1)
        W = np.delete(W, 0, axis=1)
        Z300 = np.delete(Z300, 0, axis=1)
        Z250 = np.delete(Z250, 0, axis=1)

        T250[j, :, :] = np.flipud(T250[j, :, :])
        T300[j, :, :] = np.flipud(T300[j, :, :])
        RH300[j, :, :] = np.flipud(RH300[j, :, :])
        RH250[j, :, :] = np.flipud(RH250[j, :, :])
        W[j, :, :] = np.flipud(W[j, :, :])
        Z300[j, :, :] = np.flipud(Z300[j, :, :])
        Z250[j, :, :] = np.flipud(Z250[j, :, :])

        RH300_1 = np.array(RH300[j, :, :]).reshape(64800)
        RH250_1 = np.array(RH250[j, :, :]).reshape(64800)
        RH300_1[RH300_1 <= 0] = 0.01
        RH250_1[RH250_1 <= 0] = 0.01
        T250_1 = np.array(T250[j, :, :]).reshape(64800)
        T300_1 = np.array(T300[j, :, :]).reshape(64800)
        W_1 = np.array(W[j, :, :]).reshape(64800)
        Z300_1 = np.array(Z300[j, :, :]).reshape(64800)
        Z250_1 = np.array(Z250[j, :, :]).reshape(64800)

        # Calculate sitaE using lifting condensation temperature,
        # the equation is on the paper Bolton (1980)
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

A_N10_N = stats.zscore(A_N10)
A_N11_N = stats.zscore(A_N11)
A_N12_N = stats.zscore(A_N12)
A_N13_N = stats.zscore(A_N13)

A_N1 = np.zeros((119750400, 4))
A_N1[:, 0] = A_N10_N
A_N1[:, 1] = A_N11_N
A_N1[:, 2] = A_N12_N
A_N1[:, 3] = A_N13_N

pca = PCA(n_components=1, whiten=True, copy=False)
# pca.fit(A_N1)
A_N1 = pca.fit_transform(A_N1)
# A_N1 = A_N1.reshape(11,28,180,360)
A_N1 = stats.zscore(A_N1)
# A = np.zeros((11,28,180,360))

# A_N1 = A_N1[1:11,:,:,:]
# A_N1 = np.mean(A_N1[:,:,:,:], axis=0)
A_N1 = A_N1.reshape(119750400)
A_N1 = A_N1.reshape(66, 28, 180, 360)
A_N1e = A_N1[0:60, :, :, :]  # 2010-2019
A_NKK = A_N1[48:66, :, :, :]  # 2017-2020
A_N13 = A_N1[0:6, :, :, :]  # 2010
A_N14 = A_N1[6:12, :, :, :]  # 2011
A_N15 = A_N1[12:18, :, :, :]  # 2012
A_N16 = A_N1[18:24, :, :, :]  # 2013
A_N17 = A_N1[24:30, :, :, :]  # 2014
A_N18 = A_N1[30:36, :, :, :]  # 2015
A_N19 = A_N1[36:42, :, :, :]  # 2016
A_N110 = A_N1[42:48, :, :, :]  # 2017
A_N111 = A_N1[48:54, :, :, :]  # 2018
A_N112 = A_N1[54:60, :, :, :]  # 2019
A_N113 = A_N1[60:66, :, :, :]  # 2020


A_N1 = A_N1.reshape(119750400)
A_N1e = A_N1e.reshape(108864000)
A_N13 = A_N13.reshape(10886400)
A_N14 = A_N14.reshape(10886400)
A_N15 = A_N15.reshape(10886400)
A_N16 = A_N16.reshape(10886400)
A_N17 = A_N17.reshape(10886400)
A_N18 = A_N18.reshape(10886400)
A_N19 = A_N19.reshape(10886400)
A_N110 = A_N110.reshape(10886400)
A_N111 = A_N111.reshape(10886400)
A_N112 = A_N112.reshape(10886400)
A_N113 = A_N113.reshape(10886400)


############################### Deal CERES data ###################################

A_N20 = np.zeros((1))
A_N21 = np.zeros((1))
A_N22 = np.zeros((1))
A_N23 = np.zeros((1))
A_N24 = np.zeros((1))
A_N25 = np.zeros((1))
A_N26 = np.zeros((1))
A_N27 = np.zeros((1))

CERES_file = glob.glob(
    "H:\\CERES_highcloud_1\\CERES_highcloud_" + "*.nc"
)

for i in range(0, 11):

    FILE_NAME = CERES_file[i]
    id_name = int(os.path.basename(CERES_file[i])[16:20])

    if calendar.isleap(id_name) == False:

        file_obj = xr.open_dataset(FILE_NAME)
        lat = file_obj.lat
        lon = file_obj.lon
        # t = file_obj.time
        cldarea = file_obj.cldarea_high_daily
        cldicerad = file_obj.cldicerad_high_daily
        cldtau = file_obj.cldtau_high_daily
        cldtauL = file_obj.cldtau_lin_high_daily
        iwp = file_obj.iwp_high_daily
        cldemissir = file_obj.cldemissir_high_daily

        cldarea = np.array(cldarea)
        cldicerad = np.array(cldicerad)
        cldtau = np.array(cldtau)
        cldtauL = np.array(cldtauL)
        iwp = np.array(iwp)
        cldemissir = np.array(cldemissir)

        for j in range(31, 59):  # FEB

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldareaL1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(59, 87):  # MAR

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(90, 118):  # APR

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(120, 148):  # MAY

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(151, 179):  # JUN

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(181, 209):  # JUL

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

    elif calendar.isleap(id_name) == True:

        file_obj = xr.open_dataset(FILE_NAME)
        lat = file_obj.lat
        lon = file_obj.lon
        # t = file_obj.time
        cldarea = file_obj.cldarea_high_daily
        cldicerad = file_obj.cldicerad_high_daily
        cldtau = file_obj.cldtau_high_daily
        cldtauL = file_obj.cldtau_lin_high_daily
        iwp = file_obj.iwp_high_daily
        cldemissir = file_obj.cldemissir_high_daily

        cldarea = np.array(cldarea)
        cldicerad = np.array(cldicerad)
        cldtau = np.array(cldtau)
        cldtauL = np.array(cldtauL)
        iwp = np.array(iwp)
        cldemissir = np.array(cldemissir)

        for j in range(31, 59):  # FEB

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(60, 88):  # MAR

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(91, 119):  # APR

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(121, 149):  # MAY

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(152, 180):  # JUN

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(182, 210):  # JUL

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldtauL1 = cldtauL[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldtauL1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)


A_N20 = np.delete(A_N20, 0, axis=0)  # cldarea
A_N21 = np.delete(A_N21, 0, axis=0)  # cldicerad
A_N22 = np.delete(A_N22, 0, axis=0)  # cldtau
A_N23 = np.delete(A_N23, 0, axis=0)  # iwp
A_N24 = np.delete(A_N24, 0, axis=0)  # cldtau_lin
A_N26 = np.delete(A_N26, 0, axis=0)  # cldemissir

A_N20t = A_N20.reshape(
    66, 28, 180, 360
)  # Choose the variable used in the plot
A_N20t[A_N20t == -999] = np.nan

A_NM = A_N20t[0:60, :, :, :]  # 2010-2019
A_NK = A_N20t[48:66, :, :, :]  # 2018-2020
A_N30 = A_N20t[0:6, :, :, :]  # 2010
A_N40 = A_N20t[6:12, :, :, :]  # 2011
A_N50 = A_N20t[12:18, :, :, :]  # 2012
A_N60 = A_N20t[18:24, :, :, :]  # 2013
A_N70 = A_N20t[24:30, :, :, :]  # 2014
A_N80 = A_N20t[30:36, :, :, :]  # 2015
A_N90 = A_N20t[36:42, :, :, :]  # 2016
A_N100 = A_N20t[42:48, :, :, :]  # 2017
A_N101 = A_N20t[48:54, :, :, :]  # 2018
A_N102 = A_N20t[54:60, :, :, :]  # 2019
A_N103 = A_N20t[60:66, :, :, :]  # 2020

A_N20t = A_N20t.reshape(119750400)
A_NM = A_NM.reshape(108864000)
A_N30 = A_N30.reshape(10886400)
A_N40 = A_N40.reshape(10886400)
A_N50 = A_N50.reshape(10886400)
A_N60 = A_N60.reshape(10886400)
A_N70 = A_N70.reshape(10886400)
A_N80 = A_N80.reshape(10886400)
A_N90 = A_N90.reshape(10886400)
A_N100 = A_N100.reshape(10886400)
A_N101 = A_N101.reshape(10886400)
A_N102 = A_N102.reshape(10886400)
A_N103 = A_N103.reshape(10886400)

########################## set the midddlepoint for cmap ###############################################################


class MidpointNormalize(colors.Normalize):
    def __init__(
        self, vmin=None, vmax=None, midpoint=None, clip=False
    ):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


########################## Filter the CERES array to fit PC1 gap ###################################


def A2020():
    for num1, num3, num4 in product(numlist1, numlist3, numlist4):
        if (
            any(
                ((np.array(num1) * 0.2) <= A_N113[:, num3, num4])
                & (
                    A_N113[:, num3, num4]
                    < ((np.array(num1) + 1) * 0.2)
                )
            )
            == True
        ):
            T_NNN2020[num1, num3, num4] = np.nanmean(
                A_N113[:, num3, num4][
                    np.where(
                        (
                            A_N113[:, num3, num4]
                            >= (np.array(num1) * 0.2)
                        )
                        & (
                            A_N113[:, num3, num4]
                            < ((np.array(num1) + 1) * 0.2)
                        )
                    )
                ]
            )
        else:
            T_NNN2020[num1, num3, num4] = np.nan
    return T_NNN2020


######## 分每一个空间格点，每一个EOF小段、每一天时间来计算每一年的空间场，这是最精细的方法 #########################################

# A_NM = A_NM.reshape(10, 168, 180, 360)
# A_N1e = A_N1e.reshape(10, 168, 180, 360)

# A_NK = A_NK.reshape(3, 21, 8, 180, 360)
# A_NKK = A_NKK.reshape(3, 21, 8, 180, 360)

# A_NN20 = A_N20t.reshape(11, 168, 180, 360)
# A_NN1 = A_N1.reshape(11, 168, 180, 360)

# numlist1 = [i for i in range(0, 20)]
# numlist2 = [i for i in range(0, 21)]
# numlist3 = [i for i in range(0, 8)]
# numlist4 = [i for i in range(0, 180)]
# numlist5 = [i for i in range(0, 360)]

# A_NNNall = np.zeros((20, 21, 180, 360))
# A_NNN2020 = np.zeros((20, 21, 8, 180, 360))
# A_NNN2019 = np.zeros((20, 21, 8, 180, 360))
# A_NNN2018 = np.zeros((20, 21, 8, 180, 360))
# A_NNN2017 = np.zeros((20, 21, 8, 180, 360))
# A_NNN2016 = np.zeros((20, 21, 8, 180, 360))
# A_NNN2015 = np.zeros((20, 21, 8, 180, 360))
# A_N60 = A_N60.reshape(21, 8, 180, 360)
# A_N16 = A_N16.reshape(21, 8, 180, 360)
# A_N70 = A_N70.reshape(21, 8, 180, 360)
# A_N17 = A_N17.reshape(21, 8, 180, 360)
# A_N80 = A_N80.reshape(21, 8, 180, 360)
# A_N18 = A_N18.reshape(21, 8, 180, 360)
# A_N90 = A_N90.reshape(21, 8, 180, 360)
# A_N19 = A_N19.reshape(21, 8, 180, 360)
# A_N100 = A_N100.reshape(21, 8, 180, 360)
# A_N110 = A_N110.reshape(21, 8, 180, 360)
# A_N101 = A_N101.reshape(21, 8, 180, 360)
# A_N111 = A_N111.reshape(21, 8, 180, 360)
# A_N103 = A_N103.reshape(21, 8, 180, 360)
# A_N113 = A_N113.reshape(21, 8, 180, 360)
# A_N102 = A_N102.reshape(21, 8, 180, 360)
# A_N112 = A_N112.reshape(21, 8, 180, 360)


# def Call():
#     for num1, num2, num3, num4, num5 in product(
#         numlist1, numlist2, numlist3, numlist4, numlist5
#     ):
#         if (
#             np.array(
#                 A_NK[:, num2, :, num4, num5][
#                     np.where(
#                         (A_NKK[:, num2, :, num4, num5] >= (np.array(num1) * 0.1))
#                         & (A_NKK[:, num2, :, num4, num5] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ].shape
#             )
#             > 0
#         ):
#             A_NNNall[num1, num2, num4, num5] = np.nanmean(A_NK[:, num2, :, num4, num5])
#         else:
#             A_NNNall[num1, num2, num4, num5] = np.nan
#     return A_NNNall


# def C2020():
#     for num1, num2, num3, num4, num5 in product(
#         numlist1, numlist2, numlist3, numlist4, numlist5
#     ):
#         if (
#             any(
#                 ((np.array(num1) * 0.1) <= A_N113[num2, num3, num4, num5])
#                 & (A_N113[num2, num3, num4, num5] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN2020[num1, num2, num3, num4] = (
#                 A_N103[num2, num3, num4, num5] - A_NNNall[num1, num2, num4, num5]
#             ) / A_NNNall[num1, num2, num4, num5]
#         else:
#             A_NNN2020[num1, num2, num3, num4] = np.nan
#     return A_NNN2020


# def C2019():
#     for num1, num2, num3, num4, num5 in product(
#         numlist1, numlist2, numlist3, numlist4, numlist5
#     ):
#         if (
#             any(
#                 ((np.array(num1) * 0.1) <= A_N112[num2, num3, num4, num5])
#                 & (A_N112[num2, num3, num4, num5] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN2020[num1, num2, num3, num4] = (
#                 A_N102[num2, num3, num4, num5] - A_NNNall[num1, num2, num4, num5]
#             ) / A_NNNall[num1, num2, num4, num5]
#         else:
#             A_NNN2020[num1, num2, num3, num4] = np.nan
#     return A_NNN2020


# def C2018():
#     for num1, num2, num3, num4, num5 in product(
#         numlist1, numlist2, numlist3, numlist4, numlist5
#     ):
#         if (
#             any(
#                 ((np.array(num1) * 0.1) <= A_N111[num2, num3, num4, num5])
#                 & (A_N111[num2, num3, num4, num5] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN2020[num1, num2, num3, num4] = (
#                 A_N101[num2, num3, num4, num5] - A_NNNall[num1, num2, num4, num5]
#             ) / A_NNNall[num1, num2, num4, num5]
#         else:
#             A_NNN2020[num1, num2, num3, num4] = np.nan
#     return A_NNN2020


# start = time.time()
# A_NNNall = Call()
# A_NNN2020 = C2020()
# A_NNN2019 = C2019()
# A_NNN2018 = C2018()
# end = time.time()
# print("Running time: %s Seconds" % (end - start))

############################################## 简化的四重循环思路 ###########################################################################

# A_NNNall = np.zeros((16, 21, 180, 360))
# A_NNN2020 = np.zeros((16, 21, 180, 360))
# A_NNN2019 = np.zeros((16, 21, 180, 360))
# A_NNN2018 = np.zeros((16, 21, 180, 360))
# A_N60 = A_N60.reshape(21, 8, 180, 360)
# A_N16 = A_N16.reshape(21, 8, 180, 360)
# A_N70 = A_N70.reshape(21, 8, 180, 360)
# A_N17 = A_N17.reshape(21, 8, 180, 360)
# A_N80 = A_N80.reshape(21, 8, 180, 360)
# A_N18 = A_N18.reshape(21, 8, 180, 360)
# A_N90 = A_N90.reshape(21, 8, 180, 360)
# A_N19 = A_N19.reshape(21, 8, 180, 360)
# A_N100 = A_N100.reshape(21, 8, 180, 360)
# A_N110 = A_N110.reshape(21, 8, 180, 360)
# A_N101 = A_N101.reshape(21, 8, 180, 360)
# A_N111 = A_N111.reshape(21, 8, 180, 360)
# A_N103 = A_N103.reshape(21, 8, 180, 360)
# A_N113 = A_N113.reshape(21, 8, 180, 360)
# A_N102 = A_N102.reshape(21, 8, 180, 360)
# A_N112 = A_N112.reshape(21, 8, 180, 360)
# A_NK = A_NK.reshape(3, 21, 8, 180, 360)
# A_NKK = A_NKK.reshape(3, 21, 8, 180, 360)

# # A_N103_temp = np.isnan(A_N103)
# # A_N102_temp = np.isnan(A_N102)
# # A_N101_temp = np.isnan(A_N101)
# # A_NK_temp = np.isnan(A_NK)

# # A_N103[A_N103_temp] = 0.001
# # A_N102[A_N102_temp] = 0.001
# # A_N101[A_N101_temp] = 0.001
# # A_NK[A_NK_temp] = 0.001

# numlist1 = [i for i in range(0, 16)]
# numlist2 = [i for i in range(0, 21)]
# numlist3 = [i for i in range(0, 180)]
# numlist4 = [i for i in range(0, 360)]


# def Call():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             np.array(
#                 A_NK[:, num2, :, num3, num4][
#                     np.where(
#                         (A_NKK[:, num2, :, num3, num4] >= (np.array(num1 - 4.5) * 0.5))
#                         & (A_NKK[:, num2, :, num3, num4] < (np.array(num1 - 3.5) * 0.5))
#                     )
#                 ].shape
#             )
#             > 0
#         ):
#             A_NNNall[num1, num2, num3, num4] = np.nanmean(
#                 A_NK[:, num2, :, num3, num4][
#                     np.where(
#                         (A_NKK[:, num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                         & (A_NKK[:, num2, :, num3, num4] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ]
#             )
#         else:
#             A_NNNall[num1, num2, num3, num4] = np.nan

#     return A_NNNall


# def C2020():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N113[num2, :, num3, num4])
#                 & (A_N113[num2, :, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN2020[num1, num2, num3, num4] = (
#                 np.nansum(
#                     A_N103[num2, :, num3, num4][
#                         np.where(
#                             (A_N113[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                             & (
#                                 A_N113[num2, :, num3, num4]
#                                 < ((np.array(num1) + 1) * 0.2)
#                             )
#                         )
#                     ]
#                 )
#                 - (
#                     np.array(
#                         A_N103[num2, :, num3, num4][
#                             np.where(
#                                 (A_N113[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                                 & (
#                                     A_N113[num2, :, num3, num4]
#                                     < ((np.array(num1) + 1) * 0.2)
#                                 )
#                             )
#                         ].shape
#                     )
#                 )
#                 * A_NNNall[num1, num2, num3, num4]
#             ) / A_NNNall[num1, num2, num3, num4]
#         else:
#             A_NNN2020[num1, num2, num3, num4] = np.nan
#     return A_NNN2020


# def C2019():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N112[num2, :, num3, num4])
#                 & (A_N112[num2, :, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN2019[num1, num2, num3, num4] = (
#                 np.nansum(
#                     A_N102[num2, :, num3, num4][
#                         np.where(
#                             (A_N112[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                             & (
#                                 A_N112[num2, :, num3, num4]
#                                 < ((np.array(num1) + 1) * 0.2)
#                             )
#                         )
#                     ]
#                 )
#                 - (
#                     np.array(
#                         A_N102[num2, :, num3, num4][
#                             np.where(
#                                 (A_N112[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                                 & (
#                                     A_N112[num2, :, num3, num4]
#                                     < ((np.array(num1) + 1) * 0.2)
#                                 )
#                             )
#                         ].shape
#                     )
#                 )
#                 * A_NNNall[num1, num2, num3, num4]
#             ) / A_NNNall[num1, num2, num3, num4]
#         else:
#             A_NNN2019[num1, num2, num3, num4] = np.nan
#     return A_NNN2019


# def C2018():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N111[num2, :, num3, num4])
#                 & (A_N111[num2, :, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN2018[num1, num2, num3, num4] = (
#                 np.nansum(
#                     A_N101[num2, :, num3, num4][
#                         np.where(
#                             (A_N111[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                             & (
#                                 A_N111[num2, :, num3, num4]
#                                 < ((np.array(num1) + 1) * 0.2)
#                             )
#                         )
#                     ]
#                 )
#                 - (
#                     np.array(
#                         A_N101[num2, :, num3, num4][
#                             np.where(
#                                 (A_N111[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                                 & (
#                                     A_N111[num2, :, num3, num4]
#                                     < ((np.array(num1) + 1) * 0.2)
#                                 )
#                             )
#                         ].shape
#                     )
#                 )
#                 * A_NNNall[num1, num2, num3, num4]
#             ) / A_NNNall[num1, num2, num3, num4]
#         else:
#             A_NNN2018[num1, num2, num3, num4] = np.nan
#     return A_NNN2018


# A_NNNall = Call()
# A_NNN2020 = C2020()
# A_NNN2019 = C2019()
# A_NNN2018 = C2018()

############################# Monthly mean errors version #############################

# A_NNN20202 = np.zeros((16, 7, 180, 360))
# A_NNN20192 = np.zeros((16, 7, 180, 360))
# A_NNN20182 = np.zeros((16, 7, 180, 360))
# A_NNNall2 = np.zeros((16, 7, 180, 360))
# A_N100 = A_N100.reshape(7, 24, 180, 360)
# A_N110 = A_N110.reshape(7, 24, 180, 360)
# A_N101 = A_N101.reshape(7, 24, 180, 360)
# A_N111 = A_N111.reshape(7, 24, 180, 360)
# A_N103 = A_N103.reshape(7, 24, 180, 360)
# A_N113 = A_N113.reshape(7, 24, 180, 360)
# A_N102 = A_N102.reshape(7, 24, 180, 360)
# A_N112 = A_N112.reshape(7, 24, 180, 360)
# A_NK = A_NK.reshape(3, 7, 24, 180, 360)
# A_NKK = A_NKK.reshape(3, 7, 24, 180, 360)
# numlist2 = [i for i in range(0, 7)]


# def Call2():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             np.array(
#                 A_NK[:, num2, :, num3, num4][
#                     np.where(
#                         (A_NKK[:, num2, :, num3, num4] >= (np.array(num1 - 4.5) * 0.5))
#                         & (A_NKK[:, num2, :, num3, num4] < (np.array(num1 - 3.5) * 0.5))
#                     )
#                 ].shape
#             )
#             > 0
#         ):
#             A_NNNall2[num1, num2, num3, num4] = np.nanmean(
#                 A_NK[:, num2, :, num3, num4][
#                     np.where(
#                         (A_NKK[:, num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                         & (A_NKK[:, num2, :, num3, num4] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ]
#             )
#         else:
#             A_NNNall2[num1, num2, num3, num4] = np.nan
#     return A_NNNall2


# def C20202():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N113[num2, :, num3, num4])
#                 & (A_N113[num2, :, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN20202[num1, num2, num3, num4] = (
#                 np.nansum(
#                     A_N103[num2, :, num3, num4][
#                         np.where(
#                             (A_N113[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                             & (
#                                 A_N113[num2, :, num3, num4]
#                                 < ((np.array(num1) + 1) * 0.2)
#                             )
#                         )
#                     ]
#                 )
#                 - (
#                     np.array(
#                         A_N103[num2, :, num3, num4][
#                             np.where(
#                                 (A_N113[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                                 & (
#                                     A_N113[num2, :, num3, num4]
#                                     < ((np.array(num1) + 1) * 0.2)
#                                 )
#                             )
#                         ].shape
#                     )
#                 )
#                 * A_NNNall2[num1, num2, num3, num4]
#             ) / A_NNNall2[num1, num2, num3, num4]
#         else:
#             A_NNN20202[num1, num2, num3, num4] = np.nan
#     return A_NNN20202


# def C20192():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N112[num2, :, num3, num4])
#                 & (A_N112[num2, :, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN20192[num1, num2, num3, num4] = (
#                 np.nansum(
#                     A_N102[num2, :, num3, num4][
#                         np.where(
#                             (A_N112[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                             & (
#                                 A_N112[num2, :, num3, num4]
#                                 < ((np.array(num1) + 1) * 0.2)
#                             )
#                         )
#                     ]
#                 )
#                 - (
#                     np.array(
#                         A_N102[num2, :, num3, num4][
#                             np.where(
#                                 (A_N112[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                                 & (
#                                     A_N112[num2, :, num3, num4]
#                                     < ((np.array(num1) + 1) * 0.2)
#                                 )
#                             )
#                         ].shape
#                     )
#                 )
#                 * A_NNNall2[num1, num2, num3, num4]
#             ) / A_NNNall2[num1, num2, num3, num4]
#         else:
#             A_NNN20192[num1, num2, num3, num4] = np.nan
#     return A_NNN20192


# def C20182():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N111[num2, :, num3, num4])
#                 & (A_N111[num2, :, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             A_NNN20182[num1, num2, num3, num4] = (
#                 np.nansum(
#                     A_N101[num2, :, num3, num4][
#                         np.where(
#                             (A_N111[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                             & (
#                                 A_N111[num2, :, num3, num4]
#                                 < ((np.array(num1) + 1) * 0.2)
#                             )
#                         )
#                     ]
#                 )
#                 - (
#                     np.array(
#                         A_N101[num2, :, num3, num4][
#                             np.where(
#                                 (A_N111[num2, :, num3, num4] >= (np.array(num1) * 0.2))
#                                 & (
#                                     A_N111[num2, :, num3, num4]
#                                     < ((np.array(num1) + 1) * 0.2)
#                                 )
#                             )
#                         ].shape
#                     )
#                 )
#                 * A_NNNall2[num1, num2, num3, num4]
#             ) / A_NNNall2[num1, num2, num3, num4]
#         else:
#             A_NNN20182[num1, num2, num3, num4] = np.nan
#     return A_NNN20182


# # def C2018():
# #     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
# #         if any(((np.array(num1)*0.2)<=A_N111[num2,:,num3,num4])&(A_N111[num2,:,num3,num4]<((np.array(num1)+1)*0.2)))==True:
# #             A_NNN2018[num1,num2,num3,num4] = (np.nansum(A_N101[num2,:,num3,num4]\
# #             [np.where((A_N111[num2,:,num3,num4]>=(np.array(num1)*0.2))&(A_N111[num2,:,num3,num4]<((np.array(num1)+1)*0.2)))])\
# #             -(np.array(A_N101[num2,:,num3,num4][np.where((A_N111[num2,:,num3,num4]>=(np.array(num1)*0.2))&(A_N111[num2,:,num3,num4]<\
# #             ((np.array(num1)+1)*0.2)))].shape))*A_NNNall[num1,num2,num3,num4])
# #         else :
# #             A_NNN2018[num1,num2,num3,num4] = np.nan
# #     return A_NNN2018

# start = time.time()

# A_NNNall2 = Call2()
# A_NNN20202 = C20202()
# A_NNN20192 = C20192()
# A_NNN20182 = C20182()
# end = time.time()
# print("Running time: %s Seconds" % (end - start))

# A_NK = A_NK.reshape(3, 7, 24, 180, 360)
# A_N103 = A_N103.reshape(7, 24, 180, 360)
# A_N102 = A_N102.reshape(7, 24, 180, 360)
# A_N101 = A_N101.reshape(7, 24, 180, 360)
# A_NNNall4 = np.zeros((7, 180, 360))
# A_NNN20205 = np.zeros((7, 180, 360))
# A_NNN20195 = np.zeros((7, 180, 360))
# A_NNN20185 = np.zeros((7, 180, 360))

# A_NNNall4 = np.nanmean(A_NK, axis=(0, 2))
# numlist2 = [i for i in range(0, 7)]


# def C20203():
#     for i in range(0, 7):
#         A_NNN20205[i, :, :] = (
#             (np.nansum(A_N103[i, :, :, :], axis=0)).reshape(180, 360)
#             - 28 * A_NNNall4[i, :, :]
#         ) / A_NNNall4[i, :, :]
#     return A_NNN20205


# def C20193():
#     for i in range(0, 7):
#         A_NNN20195[i, :, :] = (
#             (np.nansum(A_N102[i, :, :, :], axis=0)).reshape(180, 360)
#             - 28 * A_NNNall4[i, :, :]
#         ) / A_NNNall4[i, :, :]
#     return A_NNN20195


# def C20183():
#     for i in range(0, 7):
#         A_NNN20185[i, :, :] = (
#             (np.nansum(A_N101[i, :, :, :], axis=0)).reshape(180, 360)
#             - 28 * A_NNNall4[i, :, :]
#         ) / A_NNNall4[i, :, :]
#     return A_NNN20185


# A_NNN20205 = C20203()
# A_NNN20195 = C20193()
# A_NNN20185 = C20183()

# # def Aall():
# #     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
# #         if np.array(A_NK[:,num2,:,num3,num4][np.where((A_NKK[:,num2,:,num3,num4]>=(np.array(num1)*0.2))&(A_NKK[:,num2,:,num3,num4]<((np.array(num1)+1)*0.2)))].shape) > 0:
# #             A_NNNall[num1,num2,num3,num4] = np.nanmean(A_NK[:,num2,:,num3,num4][np.where((A_NKK[:,num2,:,num3,num4]>=(np.array(num1)*0.2))\
# #                                                                                          &(A_NKK[:,num2,:,num3,num4]<((np.array(num1)+1)*0.2)))])
# #         else :
# #             A_NNNall[num1,num2,num3,num4] = np.nan

# #     return A_NNNall

# numlist1 = [i for i in range(0, 20)]
# numlist3 = [i for i in range(0, 180)]
# numlist4 = [i for i in range(0, 360)]

# T_NNN2020 = np.zeros((40, 180, 360))
# T_NNN2019 = np.zeros((40, 180, 360))
# T_NNN2018 = np.zeros((40, 180, 360))
# A_N90 = A_N90.reshape(168, 180, 360)
# A_N19 = A_N19.reshape(168, 180, 360)
# A_N100 = A_N100.reshape(168, 180, 360)
# A_N110 = A_N110.reshape(168, 180, 360)
# A_N101 = A_N101.reshape(168, 180, 360)
# A_N111 = A_N111.reshape(168, 180, 360)
# A_N103 = A_N103.reshape(168, 180, 360)
# A_N113 = A_N113.reshape(168, 180, 360)
# A_N102 = A_N102.reshape(168, 180, 360)
# A_N112 = A_N112.reshape(168, 180, 360)


# def A2020():
#     for num1, num3, num4 in product(numlist1, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N113[:, num3, num4])
#                 & (A_N113[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             T_NNN2020[num1, num3, num4] = np.nanmean(
#                 A_N113[:, num3, num4][
#                     np.where(
#                         (A_N113[:, num3, num4] >= (np.array(num1) * 0.2))
#                         & (A_N113[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ]
#             )
#         else:
#             T_NNN2020[num1, num3, num4] = np.nan
#     return T_NNN2020


# def A2019():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N112[:, num3, num4])
#                 & (A_N112[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             T_NNN2019[num1, num3, num4] = np.nanmean(
#                 A_N112[:, num3, num4][
#                     np.where(
#                         (A_N112[:, num3, num4] >= (np.array(num1) * 0.2))
#                         & (A_N112[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ]
#             )
#         else:
#             T_NNN2019[num1, num3, num4] = np.nan
#     return T_NNN2019


# def A2018():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N111[:, num3, num4])
#                 & (A_N111[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             T_NNN2018[num1, num3, num4] = np.nanmean(
#                 A_N111[:, num3, num4][
#                     np.where(
#                         (A_N111[:, num3, num4] >= (np.array(num1) * 0.2))
#                         & (A_N111[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ]
#             )
#         else:
#             T_NNN2018[num1, num3, num4] = np.nan
#     return T_NNN2018


# start = time.time()
# T_NNN2020 = A2020()
# T_NNN2019 = A2019()
# T_NNN2018 = A2018()
# end = time.time()
# print("Running time: %s Seconds" % (end - start))

# Y_NNN2020 = np.zeros((40, 180, 360))
# Y_NNN2019 = np.zeros((40, 180, 360))
# Y_NNN2018 = np.zeros((40, 180, 360))


# def Y2020():
#     for num1, num3, num4 in product(numlist1, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N113[:, num3, num4])
#                 & (A_N113[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             Y_NNN2020[num1, num3, num4] = np.nanmean(
#                 A_N103[:, num3, num4][
#                     np.where(
#                         (A_N113[:, num3, num4] >= (np.array(num1) * 0.2))
#                         & (A_N113[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ]
#             )
#         else:
#             Y_NNN2020[num1, num3, num4] = np.nan
#     return Y_NNN2020


# def Y2019():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N112[:, num3, num4])
#                 & (A_N112[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             Y_NNN2019[num1, num3, num4] = np.nanmean(
#                 A_N102[:, num3, num4][
#                     np.where(
#                         (A_N112[:, num3, num4] >= (np.array(num1) * 0.2))
#                         & (A_N112[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ]
#             )
#         else:
#             Y_NNN2019[num1, num3, num4] = np.nan
#     return Y_NNN2019


# def Y2018():
#     for num1, num2, num3, num4 in product(numlist1, numlist2, numlist3, numlist4):
#         if (
#             any(
#                 ((np.array(num1) * 0.2) <= A_N111[:, num3, num4])
#                 & (A_N111[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#             )
#             == True
#         ):
#             Y_NNN2018[num1, num3, num4] = np.nanmean(
#                 A_N101[:, num3, num4][
#                     np.where(
#                         (A_N111[:, num3, num4] >= (np.array(num1) * 0.2))
#                         & (A_N111[:, num3, num4] < ((np.array(num1) + 1) * 0.2))
#                     )
#                 ]
#             )
#         else:
#             Y_NNN2018[num1, num3, num4] = np.nan
#     return Y_NNN2018


# start = time.time()
# Y_NNN2020 = Y2020()
# Y_NNN2019 = Y2019()
# Y_NNN2018 = Y2018()
# end = time.time()
# print("Running time: %s Seconds" % (end - start))

######## 将重要航线区空间总体平均，求出总变化趋势 #########

# A2015 = np.nanmean(A_NNN2015[:,:,90:160,:],axis=(2,3))
# A2016 = np.nanmean(A_NNN2016[:,:,90:160,:],axis=(2,3))
# A2017 = np.nanmean(A_NNN2017[:,:,90:160,:],axis=(2,3))
# A2018 = np.nanmean(A_NNN2018[:,:,90:160,:],axis=(2,3))
# A2019 = np.nanmean(A_NNN2019[:,:,90:160,:],axis=(2,3))
# A2020 = np.nanmean(A_NNN2020[:,:,90:160,:],axis=(2,3))
# # Aall = np.nanmean(A_NNNall[:,:,90:160,:],axis=(2,3))

# m2020 = np.nanmean(A_N103[:,:,:],axis=(1,2))
# m2019 = np.nanmean(A_N102[:,:,:],axis=(1,2))
# m2018 = np.nanmean(A_N101[:,:,:],axis=(1,2))
# m2017 = np.nanmean(A_N100[:,:,:],axis=(1,2))

# ######## 绘图，x轴为天数，y轴为平均后的Cldtau或其他什么变量 #########

import scipy
from scipy.signal import savgol_filter
import os

# Day = np.arange(0,168,1)

# plt.plot(Day, scipy.signal.savgol_filter(np.nanmean(A_N103,axis=(1,2)),21,3),label='A_N103',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(np.nanmean(A_NNNall,axis=(0,2,3)),21,3),label='A_NNNall',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(np.nanmean(A_N100,axis=(1,2)),21,3),label='A_N100',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(np.nanmean(A_N102,axis=(1,2)),21,3),label='A_N102',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(np.nanmean(A_N101,axis=(1,2)),21,3),label='A_N101',alpha = 0.8)
# plt.legend()

# plt.plot(Day, scipy.signal.savgol_filter(A2020[0,:],21,3),label='2020',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2019[0,:],21,3),label='2019',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2018[0,:],21,3),label='2018',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2017[0,:],21,3),label='2017',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2016[0,:],21,3),label='2016',alpha = 0.8)
# plt.legend()

# plt.plot(Day, scipy.signal.savgol_filter(m2020,21,3),label='2020',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(m2019,21,3),label='2019',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(m2018,21,3),label='2018',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(m2017,21,3),label='2017',alpha = 0.8)
# plt.legend()

# plt.plot(Day, scipy.signal.savgol_filter(A2020[1,:],21,3),label='0.2-0.4',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[2,:],21,3),label='0.4-0.6',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[3,:],21,3),label='0.6-0.8',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[4,:],21,3),label='0.8-1,0',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[5,:],21,3),label='1.0-1.2',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[6,:],21,3),label='1.2-1.4',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[7,:],21,3),label='1.4-1.6',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[8,:],21,3),label='1.6-1.8',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[9,:],21,3),label='1.8-2.0',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[10,:],21,3),label='2.0-2.2',alpha = 0.8)
# plt.plot(Day, scipy.signal.savgol_filter(A2020[11,:],21,3),label='2.2-2.4',alpha = 0.8)

# plt.legend()
# plt.title("Daily Varibility of Cldtau", fontsize=18)
# plt.xlabel("Day", fontsize=14)
# plt.ylabel("mean CERES_Cldtau()", fontsize=14)

# A_N2016 = np.zeros((42,180,360))
# A_N2017 = np.zeros((42,180,360))
# A_N2018 = np.zeros((42,180,360))
# A_N2019 = np.zeros((42,180,360))
# A_N2020 = np.zeros((42,180,360))
# A_NALL = np.zeros((42,180,360))

# start=time.time()
# for num1, num3, num4 in product(numlist1, numlist3, numlist4):
#     if A_N90[:,num3,num4][np.where((A_N19[:,num3,num4]>=(num1*0.1))&(A_N19[:,num3,num4]<((num1+1)*0.1)))].shape == 0:
#         A_N2016[num1,num3,num4] = np.nan
#     else :
#         A_N2016[num1,num3,num4] = np.nanmean(A_N90[:,num3,num4][np.where((A_N19[:,num3,num4]>=(num1*0.1))&(A_N19[:,num3,num4]<((num1+1)*0.1)))])

# for num1, num3, num4 in product(numlist1, numlist3, numlist4):
#     if A_NM[:,num3,num4][np.where((A_N1e[:,num3,num4]>=(num1*0.1))&(A_N1e[:,num3,num4]<((num1+1)*0.1)))].shape == 0:
#         A_NALL[num1,num3,num4] = np.nan
#     else :
#         A_NALL[num1,num3,num4] = np.nanmean(A_NM[:,num3,num4][np.where((A_N1e[:,num3,num4]>=(num1*0.1))&(A_N1e[:,num3,num4]<((num1+1)*0.1)))])

# for num1, num3, num4 in product(numlist1, numlist3, numlist4):
#     if A_N103[:,num3,num4][np.where((A_N113[:,num3,num4]>=(num1*0.1))&(A_N113[:,num3,num4]<((num1+1)*0.1)))].shape == 0:
#         A_N2020[num1,num3,num4] = np.nan
#     else :
#         A_N2020[num1,num3,num4] = np.nanmean(A_N103[:,num3,num4][np.where((A_N113[:,num3,num4]>=(num1*0.1))&(A_N113[:,num3,num4]<((num1+1)*0.1)))])

# for num1, num3, num4 in product(numlist1, numlist3, numlist4):
#     if A_N102[:,num3,num4][np.where((A_N112[:,num3,num4]>=(num1*0.1))&(A_N112[:,num3,num4]<((num1+1)*0.1)))].shape == 0:
#         A_N2019[num1,num3,num4] = np.nan
#     else :
#         A_N2019[num1,num3,num4] = np.nanmean(A_N102[:,num3,num4][np.where((A_N112[:,num3,num4]>=(num1*0.1))&(A_N112[:,num3,num4]<((num1+1)*0.1)))])

# for num1, num3, num4 in product(numlist1, numlist3, numlist4):
#     if A_N101[:,num3,num4][np.where((A_N111[:,num3,num4]>=(num1*0.1))&(A_N111[:,num3,num4]<((num1+1)*0.1)))].shape == 0:
#         A_N2018[num1,num3,num4] = np.nan
#     else :
#         A_N2018[num1,num3,num4] = np.nanmean(A_N101[:,num3,num4][np.where((A_N111[:,num3,num4]>=(num1*0.1))&(A_N111[:,num3,num4]<((num1+1)*0.1)))])

# for num1, num3, num4 in product(numlist1, numlist3, numlist4):
#     if A_N100[:,num3,num4][np.where((A_N110[:,num3,num4]>=(num1*0.1))&(A_N110[:,num3,num4]<((num1+1)*0.1)))].shape == 0:
#         A_N2017[num1,num3,num4] = np.nan
#     else :
#         A_N2017[num1,num3,num4] = np.nanmean(A_N100[:,num3,num4][np.where((A_N110[:,num3,num4]>=(num1*0.1))&(A_N110[:,num3,num4]<((num1+1)*0.1)))])

# end=time.time()
# print('Running time: %s Seconds'%(end-start))

# D2020 = A_NALL-A_N2020
# D2019 = A_NALL-A_N2019
# D2018 = A_NALL-A_N2018
# D2017 = A_NALL-A_N2017
# D2016 = A_NALL-A_N2016

# def SpatialComparation20(z):

#     lon = np.linspace(0,360,360)
#     lat = np.linspace(-90,90,180)

#     fig, ax = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=200)
#     plt.rc('font', size=10, weight='bold')

#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l')
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])

#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     # cmap = plt.get_cmap('seismic')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     # a = map.imshow(DDD,cmap=cmap,norm=MidpointNormalize(midpoint=0),vmax=30,vmin=-30)
#     a=map.pcolormesh(lon, lat,D2020[z],norm=colors.Normalize(vmin=-40, vmax=40),vmax=40,vmin=-40,cmap=cmap)
#     fig.colorbar(a,shrink=0.4,extend = 'both')
#     ax.set_title('2019 minus 2020 Cldarea('+str(round((z+1)*0.1,2))+'>EOF>='+str(round(z*0.1,2))+')',size=12)
#     plt.savefig('E://2019VS2020cldarea/2019minus2020_Cldarea_'+str(round((z+1)*0.1,2))+'_'+str(round(z*0.1,2))+'.png',dpi=200)
#     plt.show()

# for h in range(0,42):
#     SpatialComparation20(h)

############################################################################################

# def plotEOFgap(m,n):
#     lon = np.linspace(0,360,360)
#     lat = np.linspace(-90,90,180)

#     fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
#     plt.rc('font', size=10, weight='bold')

#     ax = axs[0]
#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l',ax = axs[0])
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])
#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     # cmap = plt.get_cmap('seismic')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     a=map.pcolormesh(lon, lat,np.nanmean(D2020[m:n,:,:],axis=0),norm=colors.Normalize(vmax=800,vmin=-800),vmax=800,vmin=-800,cmap=cmap)
#     ax.set_title('mean minus 2020 IWP('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)

#     ax = axs[1]
#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l',ax = axs[1])
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])
#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     # cmap = plt.get_cmap('seismic')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     a=map.pcolormesh(lon, lat,np.nanmean(D2019[m:n,:,:],axis=0),norm=colors.Normalize(vmax=800,vmin=-800),vmax=800,vmin=-800,cmap=cmap)
#     ax.set_title('mean minus 2019 IWP('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)

#     ax = axs[2]
#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l',ax = axs[2])
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])
#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     a=map.pcolormesh(lon, lat,np.nanmean(D2018[m:n,:,:],axis=0),norm=colors.Normalize(vmax=800,vmin=-800),vmax=800,vmin=-800,cmap=cmap)
#     fig.colorbar(a,ax=axs[:], location='right',shrink=0.9,extend = 'both')
#     ax.set_title('mean minus 2018 IWP('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)
#     plt.show()

#     ax = axs[3]
#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l',ax = axs[3])
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])
#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     a=map.pcolormesh(lon, lat,np.nanmean(D2017[m:n,:,:],axis=0),norm=colors.Normalize(vmax=800,vmin=-800),vmax=800,vmin=-800,cmap=cmap)
#     fig.colorbar(a,ax=axs[:], location='right',shrink=0.9,extend = 'both')
#     ax.set_title('mean minus 2017 IWP('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)
#     plt.show()

# plotEOFgap(14,20)

############################################################################################

# def plotbyEOFgap(m,n,t1,t2):
#     lon = np.linspace(0,360,360)
#     lat = np.linspace(-90,90,180)

#     fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
#     plt.rc('font', size=10, weight='bold')

#     ax = axs[0]
#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l',ax = axs[0])
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])
#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     # cmap = plt.get_cmap('seismic')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     a=map.pcolormesh(lon, lat,np.nanmean((A_NNNall[t1:t2,:,:]-A_NNN2020[t1:t2,:,:]),axis=(0,1)),norm=colors.Normalize(vmax=800,vmin=-800),vmax=800,vmin=-800,cmap=cmap)
#     ax.set_title(str(t1)+'-'+str(t2)+' mean minus 2020 IWP('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)

#     ax = axs[1]
#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l',ax = axs[1])
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])
#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     # cmap = plt.get_cmap('seismic')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     a=map.pcolormesh(lon, lat,np.nanmean((A_NNNall[t1:t2,:,:]-A_NNN2019[t1:t2,:,:]),axis=(0,1)),norm=colors.Normalize(vmax=800,vmin=-800),vmax=800,vmin=-800,cmap=cmap)
#     ax.set_title(str(t1)+'-'+str(t2)+' mean minus 2019 IWP('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)

#     ax = axs[2]
#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l',ax = axs[2])
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])
#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     a=map.pcolormesh(lon, lat,np.nanmean((A_NNNall[t1:t2,:,:]-A_NNN2018[t1:t2,:,:]),axis=(0,1)),norm=colors.Normalize(vmax=800,vmin=-800),vmax=800,vmin=-800,cmap=cmap)
#     fig.colorbar(a,ax=axs[:], location='right',shrink=0.9,extend = 'both')
#     ax.set_title(str(t1)+'-'+str(t2)+' mean minus 2018 IWP('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)
#     plt.show()

#     ax = axs[3]
#     #basemap设置部分
#     map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l',ax = axs[3])
#     parallels = np.arange(-90,90+30,30) #纬线
#     map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
#     ax.set_yticks(parallels,len(parallels)*[''])
#     meridians = np.arange(0,360+60,60) #经线
#     map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
#     ax.set_xticks(meridians,len(meridians)*[''])
#     map.drawcountries(linewidth=0.2)
#     map.drawcoastlines(linewidth=0.2)
#     cmap = dcmap('F://color/test8.txt')
#     cmap.set_bad('gray')
#     cmap.set_over('black',1)
#     cmap.set_under('black',1)
#     a=map.pcolormesh(lon, lat,np.nanmean((A_NNNall[t1:t2,:,:]-A_NNN2017[t1:t2,:,:]),axis=(0,1)),norm=colors.Normalize(vmax=800,vmin=-800),vmax=800,vmin=-800,cmap=cmap)
#     fig.colorbar(a,ax=axs[:], location='right',shrink=0.9,extend = 'both')
#     ax.set_title(str(t1)+'-'+str(t2)+' mean minus 2017 IWP('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)
#     plt.show()

# plotbyEOFgap(14,20,44,84)
############################## plot PCA-cldarea #########################################

# A_N2019 = np.mean(A_N112.reshape(84,180,360),axis=0)
# A_N2019C = np.mean(A_N102.reshape(84,180,360),axis=0)

# A_N2018 = np.mean(A_N111.reshape(84,180,360),axis=0)
# A_N2018C = np.mean(A_N101.reshape(84,180,360),axis=0)

# lon = np.linspace(0,360,360)
# lat = np.linspace(-90,90,180)

# fig, ax = plt.subplots(figsize=(10,10),constrained_layout=True,dpi=200)
# plt.rc('font', size=10, weight='bold')

# #basemap设置部分
# map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l')
# parallels = np.arange(-90,90+30,30) #纬线
# map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
# ax.set_yticks(parallels,len(parallels)*[''])
# meridians = np.arange(0,360+60,60) #经线
# map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
# ax.set_xticks(meridians,len(meridians)*[''])

# map.drawcountries(linewidth=1.5)
# map.drawcoastlines()
# lon, lat = np.meshgrid(lon, lat)
# cmap=dcmap('F://color/test6.txt')
# cmap.set_bad('gray')
# a=map.pcolormesh(lon, lat,DDD,
#                   norm=MidpointNormalize(midpoint=0),cmap=cmap,vmax=30,vmin=-30)
# fig.colorbar(a)
# ax.set_title('2019-2020',size=16)

# plt.show()

########################################################################################

# D_N = np.zeros((59875200))
# C_N = np.zeros((2,59875200))
# A_N1 = A_N1.reshape(59875200)
# A_N20t = A_N20t.reshape(59875200)

# # D_N = np.zeros((5443200))
# C_N = np.zeros((2, 10886400))
# C_N[0, :] = A_N113[:]
# C_N[1, :] = A_N103[:]

# D_N = pd.Series(C_N[0, :]).corr(pd.Series(C_N[1, :]), method="pearson")

# for i in range(0,180):
#     for j in range(0,360):
#         D_N[i,j] = pd.Series(C_N[0,:,i,j]).corr(pd.Series(C_N[1,:,i,j]), method='pearson')

# A_NK = A_NK.reshape(32659200)
# A_NKK = A_NKK.reshape(32659200)

# 2010-2020
mean_cldarea_perpc_2010_2020 = np.zeros((22))
mean_cldarea_perpc_2010_2020[0] = np.nanmean(
    A_N20t[np.where((A_N1 > -1.3) & (A_N1 <= -1.05))]
)
mean_cldarea_perpc_2010_2020[1] = np.nanmean(
    A_N20t[np.where((A_N1 > -1.05) & (A_N1 <= -0.8))]
)
mean_cldarea_perpc_2010_2020[2] = np.nanmean(
    A_N20t[np.where((A_N1 > -0.8) & (A_N1 <= -0.55))]
)
mean_cldarea_perpc_2010_2020[3] = np.nanmean(
    A_N20t[np.where((A_N1 > -0.55) & (A_N1 <= -0.3))]
)
mean_cldarea_perpc_2010_2020[4] = np.nanmean(
    A_N20t[np.where((A_N1 > -0.3) & (A_N1 <= -0.05))]
)
mean_cldarea_perpc_2010_2020[5] = np.nanmean(
    A_N20t[np.where((A_N1 > -0.05) & (A_N1 <= 0.2))]
)
mean_cldarea_perpc_2010_2020[6] = np.nanmean(
    A_N20t[np.where((A_N1 > 0.2) & (A_N1 <= 0.45))]
)
mean_cldarea_perpc_2010_2020[7] = np.nanmean(
    A_N20t[np.where((A_N1 > 0.45) & (A_N1 <= 0.7))]
)
mean_cldarea_perpc_2010_2020[8] = np.nanmean(
    A_N20t[np.where((A_N1 > 0.7) & (A_N1 <= 0.95))]
)
mean_cldarea_perpc_2010_2020[9] = np.nanmean(
    A_N20t[np.where((A_N1 > 0.95) & (A_N1 <= 1.2))]
)
mean_cldarea_perpc_2010_2020[10] = np.nanmean(
    A_N20t[np.where((A_N1 > 1.2) & (A_N1 <= 1.45))]
)
mean_cldarea_perpc_2010_2020[11] = np.nanmean(
    A_N20t[np.where((A_N1 > 1.45) & (A_N1 <= 1.7))]
)
mean_cldarea_perpc_2010_2020[12] = np.nanmean(
    A_N20t[np.where((A_N1 > 1.7) & (A_N1 <= 1.95))]
)
mean_cldarea_perpc_2010_2020[13] = np.nanmean(
    A_N20t[np.where((A_N1 > 1.95) & (A_N1 <= 2.2))]
)
mean_cldarea_perpc_2010_2020[14] = np.nanmean(
    A_N20t[np.where((A_N1 > 2.2) & (A_N1 <= 2.45))]
)
mean_cldarea_perpc_2010_2020[15] = np.nanmean(
    A_N20t[np.where((A_N1 > 2.45) & (A_N1 <= 2.7))]
)
mean_cldarea_perpc_2010_2020[16] = np.nanmean(
    A_N20t[np.where((A_N1 > 2.7) & (A_N1 <= 2.95))]
)
mean_cldarea_perpc_2010_2020[17] = np.nanmean(
    A_N20t[np.where((A_N1 > 2.95) & (A_N1 <= 3.2))]
)
mean_cldarea_perpc_2010_2020[18] = np.nanmean(
    A_N20t[np.where((A_N1 > 3.2) & (A_N1 <= 3.45))]
)
mean_cldarea_perpc_2010_2020[19] = np.nanmean(
    A_N20t[np.where((A_N1 > 3.45) & (A_N1 <= 3.7))]
)
mean_cldarea_perpc_2010_2020[20] = np.nanmean(
    A_N20t[np.where((A_N1 > 3.7) & (A_N1 <= 3.95))]
)
mean_cldarea_perpc_2010_2020[21] = np.nanmean(
    A_N20t[np.where((A_N1 > 3.95) & (A_N1 <= 4.2))]
)

# # 2018-2020
# mean_cldarea_perpc_2018_2020 = np.zeros((22))
# mean_cldarea_perpc_2018_2020[0] = np.nanmean(A_NK[np.where((A_NKK > -1.3) & (A_NKK <= -1.05))])
# mean_cldarea_perpc_2018_2020[1] = np.nanmean(A_NK[np.where((A_NKK > -1.05) & (A_NKK <= -0.8))])
# mean_cldarea_perpc_2018_2020[2] = np.nanmean(A_NK[np.where((A_NKK > -0.8) & (A_NKK <= -0.55))])
# mean_cldarea_perpc_2018_2020[3] = np.nanmean(A_NK[np.where((A_NKK > -0.55) & (A_NKK <= -0.3))])
# mean_cldarea_perpc_2018_2020[4] = np.nanmean(A_NK[np.where((A_NKK > -0.3) & (A_NKK <= -0.05))])
# mean_cldarea_perpc_2018_2020[5] = np.nanmean(A_NK[np.where((A_NKK > -0.05) & (A_NKK <= 0.2))])
# mean_cldarea_perpc_2018_2020[6] = np.nanmean(A_NK[np.where((A_NKK > 0.2) & (A_NKK <= 0.45))])
# mean_cldarea_perpc_2018_2020[7] = np.nanmean(A_NK[np.where((A_NKK > 0.45) & (A_NKK <= 0.7))])
# mean_cldarea_perpc_2018_2020[8] = np.nanmean(A_NK[np.where((A_NKK > 0.7) & (A_NKK <= 0.95))])
# mean_cldarea_perpc_2018_2020[9] = np.nanmean(A_NK[np.where((A_NKK > 0.95) & (A_NKK <= 1.2))])
# mean_cldarea_perpc_2018_2020[10] = np.nanmean(A_NK[np.where((A_NKK > 1.2) & (A_NKK <= 1.45))])
# mean_cldarea_perpc_2018_2020[11] = np.nanmean(A_NK[np.where((A_NKK > 1.45) & (A_NKK <= 1.7))])
# mean_cldarea_perpc_2018_2020[12] = np.nanmean(A_NK[np.where((A_NKK > 1.7) & (A_NKK <= 1.95))])
# mean_cldarea_perpc_2018_2020[13] = np.nanmean(A_NK[np.where((A_NKK > 1.95) & (A_NKK <= 2.2))])
# mean_cldarea_perpc_2018_2020[14] = np.nanmean(A_NK[np.where((A_NKK > 2.2) & (A_NKK <= 2.45))])
# mean_cldarea_perpc_2018_2020[15] = np.nanmean(A_NK[np.where((A_NKK > 2.45) & (A_NKK <= 2.7))])
# mean_cldarea_perpc_2018_2020[16] = np.nanmean(A_NK[np.where((A_NKK > 2.7) & (A_NKK <= 2.95))])
# mean_cldarea_perpc_2018_2020[17] = np.nanmean(A_NK[np.where((A_NKK > 2.95) & (A_NKK <= 3.2))])
# mean_cldarea_perpc_2018_2020[18] = np.nanmean(A_NK[np.where((A_NKK > 3.2) & (A_NKK <= 3.45))])
# mean_cldarea_perpc_2018_2020[19] = np.nanmean(A_NK[np.where((A_NKK > 3.45) & (A_NKK <= 3.7))])
# mean_cldarea_perpc_2018_2020[20] = np.nanmean(A_NK[np.where((A_NKK > 3.7) & (A_NKK <= 3.95))])
# mean_cldarea_perpc_2018_2020[21] = np.nanmean(A_NK[np.where((A_NKK > 3.95) & (A_NKK <= 4.2))])

# # 2018
# mean_cldarea_perpc_2018 = np.zeros((22))
# mean_cldarea_perpc_2018[0] = np.nanmean(A_N101[np.where((A_N111 > -1.3) & (A_N111 <= -1.05))])
# mean_cldarea_perpc_2018[1] = np.nanmean(A_N101[np.where((A_N111 > -1.05) & (A_N111 <= -0.8))])
# mean_cldarea_perpc_2018[2] = np.nanmean(A_N101[np.where((A_N111 > -0.8) & (A_N111 <= -0.55))])
# mean_cldarea_perpc_2018[3] = np.nanmean(A_N101[np.where((A_N111 > -0.55) & (A_N111 <= -0.3))])
# mean_cldarea_perpc_2018[4] = np.nanmean(A_N101[np.where((A_N111 > -0.3) & (A_N111 <= -0.05))])
# mean_cldarea_perpc_2018[5] = np.nanmean(A_N101[np.where((A_N111 > -0.05) & (A_N111 <= 0.2))])
# mean_cldarea_perpc_2018[6] = np.nanmean(A_N101[np.where((A_N111 > 0.2) & (A_N111 <= 0.45))])
# mean_cldarea_perpc_2018[7] = np.nanmean(A_N101[np.where((A_N111 > 0.45) & (A_N111 <= 0.7))])
# mean_cldarea_perpc_2018[8] = np.nanmean(A_N101[np.where((A_N111 > 0.7) & (A_N111 <= 0.95))])
# mean_cldarea_perpc_2018[9] = np.nanmean(A_N101[np.where((A_N111 > 0.95) & (A_N111 <= 1.2))])
# mean_cldarea_perpc_2018[10] = np.nanmean(A_N101[np.where((A_N111 > 1.2) & (A_N111 <= 1.45))])
# mean_cldarea_perpc_2018[11] = np.nanmean(A_N101[np.where((A_N111 > 1.45) & (A_N111 <= 1.7))])
# mean_cldarea_perpc_2018[12] = np.nanmean(A_N101[np.where((A_N111 > 1.7) & (A_N111 <= 1.95))])
# mean_cldarea_perpc_2018[13] = np.nanmean(A_N101[np.where((A_N111 > 1.95) & (A_N111 <= 2.2))])
# mean_cldarea_perpc_2018[14] = np.nanmean(A_N101[np.where((A_N111 > 2.2) & (A_N111 <= 2.45))])
# mean_cldarea_perpc_2018[15] = np.nanmean(A_N101[np.where((A_N111 > 2.45) & (A_N111 <= 2.7))])
# mean_cldarea_perpc_2018[16] = np.nanmean(A_N101[np.where((A_N111 > 2.7) & (A_N111 <= 2.95))])
# mean_cldarea_perpc_2018[17] = np.nanmean(A_N101[np.where((A_N111 > 2.95) & (A_N111 <= 3.2))])
# mean_cldarea_perpc_2018[18] = np.nanmean(A_N101[np.where((A_N111 > 3.2) & (A_N111 <= 3.45))])
# mean_cldarea_perpc_2018[19] = np.nanmean(A_N101[np.where((A_N111 > 3.45) & (A_N111 <= 3.7))])
# mean_cldarea_perpc_2018[20] = np.nanmean(A_N101[np.where((A_N111 > 3.7) & (A_N111 <= 3.95))])
# mean_cldarea_perpc_2018[21] = np.nanmean(A_N101[np.where((A_N111 > 3.95) & (A_N111 <= 4.2))])

# # 2019
# mean_cldarea_perpc_2019 = np.zeros((22))
# mean_cldarea_perpc_2019[0] = np.nanmean(A_N102[np.where((A_N112 > -1.3) & (A_N112 <= -1.05))])
# mean_cldarea_perpc_2019[1] = np.nanmean(A_N102[np.where((A_N112 > -1.05) & (A_N112 <= -0.8))])
# mean_cldarea_perpc_2019[2] = np.nanmean(A_N102[np.where((A_N112 > -0.8) & (A_N112 <= -0.55))])
# mean_cldarea_perpc_2019[3] = np.nanmean(A_N102[np.where((A_N112 > -0.55) & (A_N112 <= -0.3))])
# mean_cldarea_perpc_2019[4] = np.nanmean(A_N102[np.where((A_N112 > -0.3) & (A_N112 <= -0.05))])
# mean_cldarea_perpc_2019[5] = np.nanmean(A_N102[np.where((A_N112 > -0.05) & (A_N112 <= 0.2))])
# mean_cldarea_perpc_2019[6] = np.nanmean(A_N102[np.where((A_N112 > 0.2) & (A_N112 <= 0.45))])
# mean_cldarea_perpc_2019[7] = np.nanmean(A_N102[np.where((A_N112 > 0.45) & (A_N112 <= 0.7))])
# mean_cldarea_perpc_2019[8] = np.nanmean(A_N102[np.where((A_N112 > 0.7) & (A_N112 <= 0.95))])
# mean_cldarea_perpc_2019[9] = np.nanmean(A_N102[np.where((A_N112 > 0.95) & (A_N112 <= 1.2))])
# mean_cldarea_perpc_2019[10] = np.nanmean(A_N102[np.where((A_N112 > 1.2) & (A_N112 <= 1.45))])
# mean_cldarea_perpc_2019[11] = np.nanmean(A_N102[np.where((A_N112 > 1.45) & (A_N112 <= 1.7))])
# mean_cldarea_perpc_2019[12] = np.nanmean(A_N102[np.where((A_N112 > 1.7) & (A_N112 <= 1.95))])
# mean_cldarea_perpc_2019[13] = np.nanmean(A_N102[np.where((A_N112 > 1.95) & (A_N112 <= 2.2))])
# mean_cldarea_perpc_2019[14] = np.nanmean(A_N102[np.where((A_N112 > 2.2) & (A_N112 <= 2.45))])
# mean_cldarea_perpc_2019[15] = np.nanmean(A_N102[np.where((A_N112 > 2.45) & (A_N112 <= 2.7))])
# mean_cldarea_perpc_2019[16] = np.nanmean(A_N102[np.where((A_N112 > 2.7) & (A_N112 <= 2.95))])
# mean_cldarea_perpc_2019[17] = np.nanmean(A_N102[np.where((A_N112 > 2.95) & (A_N112 <= 3.2))])
# mean_cldarea_perpc_2019[18] = np.nanmean(A_N102[np.where((A_N112 > 3.2) & (A_N112 <= 3.45))])
# mean_cldarea_perpc_2019[19] = np.nanmean(A_N102[np.where((A_N112 > 3.45) & (A_N112 <= 3.7))])
# mean_cldarea_perpc_2019[20] = np.nanmean(A_N102[np.where((A_N112 > 3.7) & (A_N112 <= 3.95))])
# mean_cldarea_perpc_2019[21] = np.nanmean(A_N102[np.where((A_N112 > 3.95) & (A_N112 <= 4.2))])

# # 2020
# mean_cldarea_perpc_2020 = np.zeros((22))
# mean_cldarea_perpc_2020[0] = np.nanmean(A_N103[np.where((A_N113 > -1.3) & (A_N113 <= -1.05))])
# mean_cldarea_perpc_2020[1] = np.nanmean(A_N103[np.where((A_N113 > -1.05) & (A_N113 <= -0.8))])
# mean_cldarea_perpc_2020[2] = np.nanmean(A_N103[np.where((A_N113 > -0.8) & (A_N113 <= -0.55))])
# mean_cldarea_perpc_2020[3] = np.nanmean(A_N103[np.where((A_N113 > -0.55) & (A_N113 <= -0.3))])
# mean_cldarea_perpc_2020[4] = np.nanmean(A_N103[np.where((A_N113 > -0.3) & (A_N113 <= -0.05))])
# mean_cldarea_perpc_2020[5] = np.nanmean(A_N103[np.where((A_N113 > -0.05) & (A_N113 <= 0.2))])
# mean_cldarea_perpc_2020[6] = np.nanmean(A_N103[np.where((A_N113 > 0.2) & (A_N113 <= 0.45))])
# mean_cldarea_perpc_2020[7] = np.nanmean(A_N103[np.where((A_N113 > 0.45) & (A_N113 <= 0.7))])
# mean_cldarea_perpc_2020[8] = np.nanmean(A_N103[np.where((A_N113 > 0.7) & (A_N113 <= 0.95))])
# mean_cldarea_perpc_2020[9] = np.nanmean(A_N103[np.where((A_N113 > 0.95) & (A_N113 <= 1.2))])
# mean_cldarea_perpc_2020[10] = np.nanmean(A_N103[np.where((A_N113 > 1.2) & (A_N113 <= 1.45))])
# mean_cldarea_perpc_2020[11] = np.nanmean(A_N103[np.where((A_N113 > 1.45) & (A_N113 <= 1.7))])
# mean_cldarea_perpc_2020[12] = np.nanmean(A_N103[np.where((A_N113 > 1.7) & (A_N113 <= 1.95))])
# mean_cldarea_perpc_2020[13] = np.nanmean(A_N103[np.where((A_N113 > 1.95) & (A_N113 <= 2.2))])
# mean_cldarea_perpc_2020[14] = np.nanmean(A_N103[np.where((A_N113 > 2.2) & (A_N113 <= 2.45))])
# mean_cldarea_perpc_2020[15] = np.nanmean(A_N103[np.where((A_N113 > 2.45) & (A_N113 <= 2.7))])
# mean_cldarea_perpc_2020[16] = np.nanmean(A_N103[np.where((A_N113 > 2.7) & (A_N113 <= 2.95))])
# mean_cldarea_perpc_2020[17] = np.nanmean(A_N103[np.where((A_N113 > 2.95) & (A_N113 <= 3.2))])
# mean_cldarea_perpc_2020[18] = np.nanmean(A_N103[np.where((A_N113 > 3.2) & (A_N113 <= 3.45))])
# mean_cldarea_perpc_2020[19] = np.nanmean(A_N103[np.where((A_N113 > 3.45) & (A_N113 <= 3.7))])
# mean_cldarea_perpc_2020[20] = np.nanmean(A_N103[np.where((A_N113 > 3.7) & (A_N113 <= 3.95))])
# mean_cldarea_perpc_2020[21] = np.nanmean(A_N103[np.where((A_N113 > 3.95) & (A_N113 <= 4.2))])

# D_N10 = np.zeros((22))
# E_N10 = np.zeros((22))
# F_N10 = np.zeros((22))

# D_N10[0] = np.nanstd(A_N103[np.where((A_N113 > -1.3) & (A_N113 <= -1.05))])
# D_N10[1] = np.nanstd(A_N103[np.where((A_N113 > -1.05) & (A_N113 <= -0.8))])
# D_N10[2] = np.nanstd(A_N103[np.where((A_N113 > -0.8) & (A_N113 <= -0.55))])
# D_N10[3] = np.nanstd(A_N103[np.where((A_N113 > -0.55) & (A_N113 <= -0.3))])
# D_N10[4] = np.nanstd(A_N103[np.where((A_N113 > -0.3) & (A_N113 <= -0.05))])
# D_N10[5] = np.nanstd(A_N103[np.where((A_N113 > -0.05) & (A_N113 <= 0.2))])
# D_N10[6] = np.nanstd(A_N103[np.where((A_N113 > 0.2) & (A_N113 <= 0.45))])
# D_N10[7] = np.nanstd(A_N103[np.where((A_N113 > 0.45) & (A_N113 <= 0.7))])
# D_N10[8] = np.nanstd(A_N103[np.where((A_N113 > 0.7) & (A_N113 <= 0.95))])
# D_N10[9] = np.nanstd(A_N103[np.where((A_N113 > 0.95) & (A_N113 <= 1.2))])
# D_N10[10] = np.nanstd(A_N103[np.where((A_N113 > 1.2) & (A_N113 <= 1.45))])
# D_N10[11] = np.nanstd(A_N103[np.where((A_N113 > 1.45) & (A_N113 <= 1.7))])
# D_N10[12] = np.nanstd(A_N103[np.where((A_N113 > 1.7) & (A_N113 <= 1.95))])
# D_N10[13] = np.nanstd(A_N103[np.where((A_N113 > 1.95) & (A_N113 <= 2.2))])
# D_N10[14] = np.nanstd(A_N103[np.where((A_N113 > 2.2) & (A_N113 <= 2.45))])
# D_N10[15] = np.nanstd(A_N103[np.where((A_N113 > 2.45) & (A_N113 <= 2.7))])
# D_N10[16] = np.nanstd(A_N103[np.where((A_N113 > 2.7) & (A_N113 <= 2.95))])
# D_N10[17] = np.nanstd(A_N103[np.where((A_N113 > 2.95) & (A_N113 <= 3.2))])
# D_N10[18] = np.nanstd(A_N103[np.where((A_N113 > 3.2) & (A_N113 <= 3.45))])
# D_N10[19] = np.nanstd(A_N103[np.where((A_N113 > 3.45) & (A_N113 <= 3.7))])
# D_N10[20] = np.nanstd(A_N103[np.where((A_N113 > 3.7) & (A_N113 <= 3.95))])
# D_N10[21] = np.nanstd(A_N103[np.where((A_N113 > 3.95) & (A_N113 <= 4.2))])

# F_N10[0] = np.nanmax(A_N103[np.where((A_N113 > -1.3) & (A_N113 <= -1.05))]) - np.nanmin(
#     A_N103[np.where((A_N113 > -1.3) & (A_N113 <= -1.05))]
# )
# F_N10[1] = np.nanmax(A_N103[np.where((A_N113 > -1.05) & (A_N113 <= -0.8))]) - np.nanmin(
#     A_N103[np.where((A_N113 > -1.05) & (A_N113 <= -0.8))]
# )
# F_N10[2] = np.nanmax(A_N103[np.where((A_N113 > -0.8) & (A_N113 <= -0.55))]) - np.nanmin(
#     A_N103[np.where((A_N113 > -0.8) & (A_N113 <= -0.55))]
# )
# F_N10[3] = np.nanmax(A_N103[np.where((A_N113 > -0.55) & (A_N113 <= -0.3))]) - np.nanmin(
#     A_N103[np.where((A_N113 > -0.55) & (A_N113 <= -0.3))]
# )
# F_N10[4] = np.nanmax(A_N103[np.where((A_N113 > -0.3) & (A_N113 <= -0.05))]) - np.nanmin(
#     A_N103[np.where((A_N113 > -0.3) & (A_N113 <= -0.05))]
# )
# F_N10[5] = np.nanmax(A_N103[np.where((A_N113 > -0.05) & (A_N113 <= 0.2))]) - np.nanmin(
#     A_N103[np.where((A_N113 > -0.05) & (A_N113 <= 0.2))]
# )
# F_N10[6] = np.nanmax(A_N103[np.where((A_N113 > 0.2) & (A_N113 <= 0.45))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 0.2) & (A_N113 <= 0.45))]
# )
# F_N10[7] = np.nanmax(A_N103[np.where((A_N113 > 0.45) & (A_N113 <= 0.7))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 0.45) & (A_N113 <= 0.7))]
# )
# F_N10[8] = np.nanmax(A_N103[np.where((A_N113 > 0.7) & (A_N113 <= 0.95))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 0.7) & (A_N113 <= 0.95))]
# )
# F_N10[9] = np.nanmax(A_N103[np.where((A_N113 > 0.95) & (A_N113 <= 1.2))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 0.95) & (A_N113 <= 1.2))]
# )
# F_N10[10] = np.nanmax(A_N103[np.where((A_N113 > 1.2) & (A_N113 <= 1.45))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 1.2) & (A_N113 <= 1.45))]
# )
# F_N10[11] = np.nanmax(A_N103[np.where((A_N113 > 1.45) & (A_N113 <= 1.7))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 1.45) & (A_N113 <= 1.7))]
# )
# F_N10[12] = np.nanmax(A_N103[np.where((A_N113 > 1.7) & (A_N113 <= 1.95))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 1.7) & (A_N113 <= 1.95))]
# )
# F_N10[13] = np.nanmax(A_N103[np.where((A_N113 > 1.95) & (A_N113 <= 2.2))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 1.95) & (A_N113 <= 2.2))]
# )
# F_N10[14] = np.nanmax(A_N103[np.where((A_N113 > 2.2) & (A_N113 <= 2.45))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 2.2) & (A_N113 <= 2.45))]
# )
# F_N10[15] = np.nanmax(A_N103[np.where((A_N113 > 2.45) & (A_N113 <= 2.7))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 2.45) & (A_N113 <= 2.7))]
# )
# F_N10[16] = np.nanmax(A_N103[np.where((A_N113 > 2.7) & (A_N113 <= 2.95))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 2.7) & (A_N113 <= 2.95))]
# )
# F_N10[17] = np.nanmax(A_N103[np.where((A_N113 > 2.95) & (A_N113 <= 3.2))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 2.95) & (A_N113 <= 3.2))]
# )
# F_N10[18] = np.nanmax(A_N103[np.where((A_N113 > 3.2) & (A_N113 <= 3.45))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 3.2) & (A_N113 <= 3.45))]
# )
# F_N10[19] = np.nanmax(A_N103[np.where((A_N113 > 3.45) & (A_N113 <= 3.7))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 3.45) & (A_N113 <= 3.7))]
# )
# F_N10[20] = np.nanmax(A_N103[np.where((A_N113 > 3.7) & (A_N113 <= 3.95))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 3.7) & (A_N113 <= 3.95))]
# )
# F_N10[21] = np.nanmax(A_N103[np.where((A_N113 > 3.95) & (A_N113 <= 4.2))]) - np.nanmin(
#     A_N103[np.where((A_N113 > 3.95) & (A_N113 <= 4.2))]
# )

# E_N10[0] = np.nanstd(A_N113[np.where((A_N113 > -1.3) & (A_N113 <= -1.05))])
# E_N10[1] = np.nanstd(A_N113[np.where((A_N113 > -1.05) & (A_N113 <= -0.8))])
# E_N10[2] = np.nanstd(A_N113[np.where((A_N113 > -0.8) & (A_N113 <= -0.55))])
# E_N10[3] = np.nanstd(A_N113[np.where((A_N113 > -0.55) & (A_N113 <= -0.3))])
# E_N10[4] = np.nanstd(A_N113[np.where((A_N113 > -0.3) & (A_N113 <= -0.05))])
# E_N10[5] = np.nanstd(A_N113[np.where((A_N113 > -0.05) & (A_N113 <= 0.2))])
# E_N10[6] = np.nanstd(A_N113[np.where((A_N113 > 0.2) & (A_N113 <= 0.45))])
# E_N10[7] = np.nanstd(A_N113[np.where((A_N113 > 0.45) & (A_N113 <= 0.7))])
# E_N10[8] = np.nanstd(A_N113[np.where((A_N113 > 0.7) & (A_N113 <= 0.95))])
# E_N10[9] = np.nanstd(A_N113[np.where((A_N113 > 0.95) & (A_N113 <= 1.2))])
# E_N10[10] = np.nanstd(A_N113[np.where((A_N113 > 1.2) & (A_N113 <= 1.45))])
# E_N10[11] = np.nanstd(A_N113[np.where((A_N113 > 1.45) & (A_N113 <= 1.7))])
# E_N10[12] = np.nanstd(A_N113[np.where((A_N113 > 1.7) & (A_N113 <= 1.95))])
# E_N10[13] = np.nanstd(A_N113[np.where((A_N113 > 1.95) & (A_N113 <= 2.2))])
# E_N10[14] = np.nanstd(A_N113[np.where((A_N113 > 2.2) & (A_N113 <= 2.45))])
# E_N10[15] = np.nanstd(A_N113[np.where((A_N113 > 2.45) & (A_N113 <= 2.7))])
# E_N10[16] = np.nanstd(A_N113[np.where((A_N113 > 2.7) & (A_N113 <= 2.95))])
# E_N10[17] = np.nanstd(A_N113[np.where((A_N113 > 2.95) & (A_N113 <= 3.2))])
# E_N10[18] = np.nanstd(A_N113[np.where((A_N113 > 3.2) & (A_N113 <= 3.45))])
# E_N10[19] = np.nanstd(A_N113[np.where((A_N113 > 3.45) & (A_N113 <= 3.7))])
# E_N10[20] = np.nanstd(A_N113[np.where((A_N113 > 3.7) & (A_N113 <= 3.95))])
# E_N10[21] = np.nanstd(A_N113[np.where((A_N113 > 3.95) & (A_N113 <= 4.2))])

PCA = np.arange(-1.3, 4.2, 0.25)
# plt.plot(PCA, B_N,label='2010')
# plt.plot(PCA, B_N0,color='blue', label='2011-2019')
plt.figure(figsize=(8, 5))
plt.plot(
    PCA,
    mean_cldarea_perpc_2010_2020,
    color="blue",
    label="mean",
    linewidth=3,
    ls="--",
)
# plt.plot(PCA, B_N8, label="2018", alpha=0.5)
# plt.plot(PCA, B_N9, label="2019", alpha=0.5)

# plt.plot(PCA, B_N10, color="red", label="2020", linewidth=3, ls="-.")
# plt.errorbar(PCA, B_N10, fmt="bo:", yerr=F_N10, xerr=E_N10)

plt.legend()
plt.grid()
plt.title("EOF-CERES_HCF", fontsize=18)
plt.xlabel("EOF", fontsize=14)
plt.ylabel("CERES_HCF(%)", fontsize=14)

# def Least_squares(x,y):
#     x_ = x.mean()
#     y_ = y.mean()
#     m = np.zeros(1)
#     n = np.zeros(1)
#     k = np.zeros(1)
#     p = np.zeros(1)
#     for i in np.arange(22):
#         k = (x[i]-x_)* (y[i]-y_)
#         m += k
#         p = np.square( x[i]-x_ )
#         n = n + p
#     a = m/n
#     b = y_ - a* x_
#     return a,b

# if __name__ == '__main__':
#     a,b = Least_squares(PCA,B_N10)
#     print (a,b)
#     y1 = a * PCA + b
#     plt.figure(figsize=(10, 5), facecolor='w')
#     plt.plot(PCA, B_N10, 'ro', lw=2, markersize=6)
#     plt.plot(PCA, y1, 'r-', lw=2, markersize=6)
#     plt.grid(b=True, ls=':')
#     plt.xlabel(u'X', fontsize=16)
#     plt.ylabel(u'Y', fontsize=16)
#     plt.show()

# sns.displot(A_N12,kind="kde")

# plt.subplot(111)
# hist, bin_edges = np.histogram(A_N12)
# plt.plot(hist)
