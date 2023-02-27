# -*- coding: utf-8 -*-

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

# A_N13 = A_N13.reshape(33,28,180,360)

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

# A[0,:,:,:] = A_N13[0,:,:,:]
# A[1,:,:,:] = A_N14[0,:,:,:]
# A[2,:,:,:] = A_N15[0,:,:,:]
# A[3,:,:,:] = A_N16[0,:,:,:]
# A[4,:,:,:] = A_N17[0,:,:,:]
# A[5,:,:,:] = A_N18[0,:,:,:]
# A[6,:,:,:] = A_N19[0,:,:,:]
# A[7,:,:,:] = A_N110[0,:,:,:]
# A[8,:,:,:] = A_N111[0,:,:,:]
# A[9,:,:,:] = A_N112[0,:,:,:]
# A[10,:,:,:] = A_N113[0,:,:,:]

A_N1 = A_N1.reshape(119750400)
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
