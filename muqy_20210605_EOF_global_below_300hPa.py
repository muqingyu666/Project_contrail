# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:30:13 2021

@author: Mu o(*￣▽￣*)ブ
"""
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
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

# ''' t,rh,w get '''
# for i in range(0,66):

#     # year_str=str(2010).zfill(4)
#     # month_str=str(1).zfill(2)
#     # time_str0=year_str+month_str

#     ERA_file=glob.glob('E:\\ERA5_daily_stored per month_global_3\\ERA5_daily_' +'*.nc')
#     FILE_NAME_ERA = ERA_file[i]

#     for j in range(0,28):

#         file_obj = xr.open_dataset(FILE_NAME_ERA)
#         lat = file_obj.lat
#         lon = file_obj.lon
#         P = file_obj.level
#         z = file_obj.Geo

#         RH = file_obj.RH
#         T = file_obj.T
#         W = file_obj.W
#         T250 = T[:,6,:,:]
#         T300 = T[:,7,:,:]
#         RH300 = RH[:,7,:,:]
#         RH250 = RH[:,6,:,:]
#         W = W[:,7,:,:]
#         Z300 = z[:,7,:,:]
#         Z250 = z[:,6,:,:]

#         T250 = np.delete(T250, 0, axis=1)
#         T300 = np.delete(T300, 0, axis=1)
#         RH300 = np.delete(RH300, 0, axis=1)
#         RH250 = np.delete(RH250, 0, axis=1)
#         W = np.delete(W, 0, axis=1)
#         Z300 = np.delete(Z300, 0, axis=1)
#         Z250 = np.delete(Z250, 0, axis=1)

#         T250[j,:,:] = np.flipud(T250[j,:,:])
#         T300[j,:,:] = np.flipud(T300[j,:,:])
#         RH300[j,:,:] = np.flipud(RH300[j,:,:])
#         RH250[j,:,:] = np.flipud(RH250[j,:,:])
#         W[j,:,:] = np.flipud(W[j,:,:])
#         Z300[j,:,:] = np.flipud(Z300[j,:,:])
#         Z250[j,:,:] = np.flipud(Z250[j,:,:])

#         RH300_1 = np.array(RH300[j,:,:]).reshape(64800)
#         RH250_1 = np.array(RH250[j,:,:]).reshape(64800)
#         RH300_1[RH300_1<=0]=0.01
#         RH250_1[RH250_1<=0]=0.01
#         T250_1 = np.array(T250[j,:,:]).reshape(64800)
#         T300_1 = np.array(T300[j,:,:]).reshape(64800)
#         W_1 = np.array(W[j,:,:]).reshape(64800)
#         Z300_1 = np.array(Z300[j,:,:]).reshape(64800)
#         Z250_1 = np.array(Z250[j,:,:]).reshape(64800)

#         #Calculate sitaE using lifting condensation temperature,the equation is on the paper Bolton (1980)
#         # r300 = np.array(mpcalc.mixing_ratio_from_relative_humidity(300. * units.mbar, T300_1 * units.kelvin, RH300_1 * units.dimensionless))
#         # r250 = np.array(mpcalc.mixing_ratio_from_relative_humidity(250. * units.mbar, T250_1 * units.kelvin, RH250_1 * units.dimensionless))
#         # Tl300 = 1/(1/(T300_1-55)-(np.log(RH300_1/100))/2840)
#         # Tl250 = 1/(1/(T250_1-55)-(np.log(RH250_1/100))/2840)
#         # e300 = np.array(mpcalc.vapor_pressure(300. * units.mbar, r300 * units.dimensionless))
#         # e250 = np.array(mpcalc.vapor_pressure(250. * units.mbar, r250 * units.dimensionless))
#         # sitaDL300 = T300_1*(1000/(300-e300))**0.2854*(T300_1/Tl300)**(0.28*10**(-3)*r300)
#         # sitaDL250 = T250_1*(1000/(250-e300))**0.2854*(T250_1/Tl250)**(0.28*10**(-3)*r250)
#         # sitaE300 = sitaDL300*np.exp((3.036/Tl300-0.00178)*r300*(1+0.448*10**(-3)*r300))
#         # sitaE250 = sitaDL250*np.exp((3.036/Tl250-0.00178)*r250*(1+0.448*10**(-3)*r250))
#         # stab = (sitaE300-sitaE250)/(Z300_1-Z250_1)
#         dewpoint300 = np.array(mpcalc.dewpoint_from_relative_humidity(T300_1 * units.kelvin, RH300_1 * units.dimensionless))
#         dewpoint250 = np.array(mpcalc.dewpoint_from_relative_humidity(T250_1 * units.kelvin, RH250_1 * units.dimensionless))
#         sitaE300 = np.array(mpcalc.equivalent_potential_temperature(300. * units.mbar, T300_1 * units.kelvin, dewpoint300 * units.degree_Celsius))
#         sitaE250 = np.array(mpcalc.equivalent_potential_temperature(250. * units.mbar, T250_1 * units.kelvin, dewpoint250 * units.degree_Celsius))
#         stab = (sitaE300-sitaE250)/(Z300_1-Z250_1)

#         A_N10 = np.concatenate((A_N10,RH300_1),axis=0)
#         A_N11 = np.concatenate((A_N11,T300_1),axis=0)
#         A_N12 = np.concatenate((A_N12,W_1),axis=0)
#         A_N13 = np.concatenate((A_N13,stab),axis=0)

""" stablility below300"""
for i in range(0, 66):

    # year_str=str(2010).zfill(4)
    # month_str=str(1).zfill(2)
    # time_str0=year_str+month_str

    ERA_file = glob.glob(
        "E:\\ERA5_daily_stored per month_global_4\\ERA5_daily_"
        + "*.nc"
    )
    FILE_NAME_ERA = ERA_file[i]

    for j in range(0, 28):

        file_obj = xr.open_dataset(FILE_NAME_ERA)
        P = file_obj.level
        z = file_obj.Geo

        RH = file_obj.RH
        T = file_obj.T
        W = file_obj.W
        T350 = T[:, 0, :, :]
        T500 = T[:, 3, :, :]
        RH350 = RH[:, 0, :, :]
        RH500 = RH[:, 3, :, :]
        Z350 = z[:, 0, :, :]
        Z500 = z[:, 3, :, :]
        W = W[:, 0, :, :]

        T350 = np.delete(T350, 0, axis=1)
        T500 = np.delete(T500, 0, axis=1)
        RH350 = np.delete(RH350, 0, axis=1)
        RH500 = np.delete(RH500, 0, axis=1)
        Z350 = np.delete(Z350, 0, axis=1)
        Z500 = np.delete(Z500, 0, axis=1)
        W = np.delete(W, 0, axis=1)

        T350[j, :, :] = np.flipud(T350[j, :, :])
        T500[j, :, :] = np.flipud(T500[j, :, :])
        RH350[j, :, :] = np.flipud(RH350[j, :, :])
        RH500[j, :, :] = np.flipud(RH500[j, :, :])
        Z350[j, :, :] = np.flipud(Z350[j, :, :])
        Z500[j, :, :] = np.flipud(Z500[j, :, :])
        W[j, :, :] = np.flipud(W[j, :, :])

        RH350_1 = np.array(RH350[j, :, :]).reshape(64800)
        RH500_1 = np.array(RH500[j, :, :]).reshape(64800)
        RH350_1[RH350_1 <= 0] = 0.1
        RH500_1[RH500_1 <= 0] = 0.1
        RH350_1[RH350_1 >= 130] = 130
        RH500_1[RH500_1 >= 130] = 130
        T350_1 = np.array(T350[j, :, :]).reshape(64800)
        T500_1 = np.array(T500[j, :, :]).reshape(64800)
        T350_1[T350_1 <= 200] = 200
        T500_1[T500_1 <= 200] = 200
        T350_1[T350_1 >= 280] = 280
        T500_1[T500_1 >= 280] = 280
        Z350_1 = np.array(Z350[j, :, :]).reshape(64800)
        Z500_1 = np.array(Z500[j, :, :]).reshape(64800)
        W_1 = np.array(W[j, :, :]).reshape(64800)

        dewpoinT350 = np.array(
            mpcalc.dewpoint_from_relative_humidity(
                T350_1 * units.kelvin,
                RH350_1 * units.dimensionless,
            )
        )
        dewpoinT500 = np.array(
            mpcalc.dewpoint_from_relative_humidity(
                T500_1 * units.kelvin,
                RH500_1 * units.dimensionless,
            )
        )
        dewpoinT500[dewpoinT500 >= 75] = 75

        sitaE350 = np.array(
            mpcalc.equivalent_potential_temperature(
                350.0 * units.mbar,
                T350_1 * units.kelvin,
                dewpoinT350 * units.degree_Celsius,
            )
        )
        sitaE500 = np.array(
            mpcalc.equivalent_potential_temperature(
                500.0 * units.mbar,
                T500_1 * units.kelvin,
                dewpoinT500 * units.degree_Celsius,
            )
        )
        stab = (sitaE500 - sitaE350) / (Z500_1 - Z350_1)

        A_N10 = np.concatenate((A_N10, RH350_1), axis=0)
        A_N11 = np.concatenate((A_N11, T350_1), axis=0)
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
A_N1e = A_N1[0:60, :, :, :]  # 2010-2019
A_NKK = A_N1[36:66, :, :, :]  # 2016-2020
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
    "G:\\CERES_highcloud_1\\CERES_highcloud_" + "*.nc"
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
A_NK = A_N20t[36:66, :, :, :]  # 2016-2020
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

########################## set the midddlepoint for cmap ###############################################


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


#########################################################################################

# A_NNN = np.zeros((180,360))
# A_NNN = np.random.rand(180,360)
# A_NNN1 = np.random.rand(180,360)

# A_N90 = A_N90.reshape(84,180,360)
# A_N19 = A_N19.reshape(84,180,360)

# A_N101 = A_N101.reshape(84,180,360)
# A_N111 = A_N111.reshape(84,180,360)

# A_N103 = A_N103.reshape(84,180,360)
# A_N113 = A_N113.reshape(84,180,360)

# A_N102 = A_N102.reshape(84,180,360)
# A_N112 = A_N112.reshape(84,180,360)

# for i in range(0,180):
#     for j in range(0,360):
#         if A_N103[:,i,j][np.where((A_N113[:,i,j]>=0))].shape == 0:
#             A_NNN[i,j] = np.nan
#         else :
#             A_NNN[i,j] = np.nanmean(A_N103[:,i,j][np.where((A_N113[:,i,j]>=0))])

# for i in range(0,180):
#     for j in range(0,360):
#         if A_N102[:,i,j][np.where((A_N112[:,i,j]>=0))].shape == 0:
#             A_NNN1[i,j] = np.nan
#         else :
#             A_NNN1[i,j] = np.nanmean(A_N102[:,i,j][np.where((A_N112[:,i,j]>=0))])

# DDD = A_NNN1-A_NNN

# A_N103 = A_N103.reshape(3,28,180,360)
# A_N113 = A_N113.reshape(3,28,180,360)
# A_N90 = A_N90.reshape(3,28,180,360)

# A_N10 = A_N10.reshape(11,3,28,180,360)

# lon = np.linspace(0,360,360)
# lat = np.linspace(-90,90,180)

# fig, ax = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=200)
# plt.rc('font', size=10, weight='bold')

# #basemap设置部分
# map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l')
# parallels = np.arange(-90,90+30,30) #纬线
# map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
# ax.set_yticks(parallels,len(parallels)*[''])
# meridians = np.arange(0,360+60,60) #经线
# map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
# ax.set_xticks(meridians,len(meridians)*[''])

# map.drawcountries(linewidth=0.5)
# map.drawcoastlines(linewidth=0.5)
# lon, lat = np.meshgrid(lon, lat)
# cmap = dcmap('F://color/test6.txt')
# # cmap = plt.get_cmap('seismic')
# cmap.set_bad('gray')
# # a = map.imshow(DDD,cmap=cmap,norm=MidpointNormalize(midpoint=0),vmax=30,vmin=-30)
# a=map.pcolormesh(lon, lat,A_N10[6,2,0,:,:],cmap=cmap)
# fig.colorbar(a,shrink=0.6)
# ax.set_title('2019 minus 2020 Cldarea(EOF>0)',size=16)

# plt.show()

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
# a=map.pcolormesh(lon, lat,A_N2018,cmap=cmap,vmax=5.01)
# fig.colorbar(a)
# ax.set_title('2018 mean EOF',size=16)

# plt.show()

########################################################################################

# D_N = np.zeros((59875200))
# C_N = np.zeros((2,59875200))
# A_N1 = A_N1.reshape(59875200)
# A_N20t = A_N20t.reshape(59875200)

# # D_N = np.zeros((5443200))
# # C_N = np.zeros((2,5443200))
# C_N[0,:] = A_N1[:]
# C_N[1,:] = A_N20t[:]

# D_N = pd.Series(C_N[0,:]).corr(pd.Series(C_N[1,:]), method='pearson')

# for i in range(0,180):
#     for j in range(0,360):
#         D_N[i,j] = pd.Series(C_N[0,:,i,j]).corr(pd.Series(C_N[1,:,i,j]), method='pearson')


# B_N0 = np.zeros((71))

# B_N0[0] = np.nanmean(A_NM[np.where((A_N1e>-10)&(A_N1e<=-9))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1e>-9)&(A_N1e<=-8))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1e>-8)&(A_N1e<=-7))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1e>-7)&(A_N1e<=-6))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1e>-6)&(A_N1e<=-5))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1e>-5)&(A_N1e<=-4))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1e>-4)&(A_N1e<=-3))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1e>-3)&(A_N1e<=-2))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1e>-2)&(A_N1e<=-1))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1e>-1)&(A_N1e<=0))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1e>0)&(A_N1e<=1))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1e>1)&(A_N1e<=2))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1e>2)&(A_N1e<=3))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1e>3)&(A_N1e<=4))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1e>4)&(A_N1e<=5))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1e>5)&(A_N1e<=6))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1e>6)&(A_N1e<=7))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1e>7)&(A_N1e<=8))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1e>8)&(A_N1e<=9))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1e>9)&(A_N1e<=10))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1e>10)&(A_N1e<=11))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1e>11)&(A_N1e<=12))])
# B_N0[22] = np.nanmean(A_NM[np.where((A_N1e>12)&(A_N1e<=13))])
# B_N0[23] = np.nanmean(A_NM[np.where((A_N1e>13)&(A_N1e<=14))])
# B_N0[24] = np.nanmean(A_NM[np.where((A_N1e>14)&(A_N1e<=15))])
# B_N0[25] = np.nanmean(A_NM[np.where((A_N1e>15)&(A_N1e<=16))])
# B_N0[26] = np.nanmean(A_NM[np.where((A_N1e>16)&(A_N1e<=17))])
# B_N0[27] = np.nanmean(A_NM[np.where((A_N1e>17)&(A_N1e<=18))])
# B_N0[28] = np.nanmean(A_NM[np.where((A_N1e>18)&(A_N1e<=19))])
# B_N0[29] = np.nanmean(A_NM[np.where((A_N1e>19)&(A_N1e<=20))])
# B_N0[30] = np.nanmean(A_NM[np.where((A_N1e>20)&(A_N1e<=21))])
# B_N0[31] = np.nanmean(A_NM[np.where((A_N1e>21)&(A_N1e<=22))])
# B_N0[32] = np.nanmean(A_NM[np.where((A_N1e>22)&(A_N1e<=23))])
# B_N0[33] = np.nanmean(A_NM[np.where((A_N1e>23)&(A_N1e<=24))])
# B_N0[34] = np.nanmean(A_NM[np.where((A_N1e>24)&(A_N1e<=25))])
# B_N0[35] = np.nanmean(A_NM[np.where((A_N1e>25)&(A_N1e<=26))])
# B_N0[36] = np.nanmean(A_NM[np.where((A_N1e>26)&(A_N1e<=27))])
# B_N0[37] = np.nanmean(A_NM[np.where((A_N1e>27)&(A_N1e<=28))])
# B_N0[38] = np.nanmean(A_NM[np.where((A_N1e>28)&(A_N1e<=29))])
# B_N0[39] = np.nanmean(A_NM[np.where((A_N1e>29)&(A_N1e<=30))])
# B_N0[40] = np.nanmean(A_NM[np.where((A_N1e>30)&(A_N1e<=31))])
# B_N0[41] = np.nanmean(A_NM[np.where((A_N1e>31)&(A_N1e<=32))])
# B_N0[42] = np.nanmean(A_NM[np.where((A_N1e>32)&(A_N1e<=33))])
# B_N0[43] = np.nanmean(A_NM[np.where((A_N1e>33)&(A_N1e<=34))])
# B_N0[44] = np.nanmean(A_NM[np.where((A_N1e>34)&(A_N1e<=35))])
# B_N0[45] = np.nanmean(A_NM[np.where((A_N1e>35)&(A_N1e<=36))])
# B_N0[46] = np.nanmean(A_NM[np.where((A_N1e>36)&(A_N1e<=37))])
# B_N0[47] = np.nanmean(A_NM[np.where((A_N1e>37)&(A_N1e<=38))])
# B_N0[48] = np.nanmean(A_NM[np.where((A_N1e>38)&(A_N1e<=39))])
# B_N0[49] = np.nanmean(A_NM[np.where((A_N1e>39)&(A_N1e<=40))])
# B_N0[50] = np.nanmean(A_NM[np.where((A_N1e>40)&(A_N1e<=41))])
# B_N0[51] = np.nanmean(A_NM[np.where((A_N1e>41)&(A_N1e<=42))])
# B_N0[52] = np.nanmean(A_NM[np.where((A_N1e>42)&(A_N1e<=43))])
# B_N0[53] = np.nanmean(A_NM[np.where((A_N1e>43)&(A_N1e<=44))])
# B_N0[54] = np.nanmean(A_NM[np.where((A_N1e>44)&(A_N1e<=45))])
# B_N0[55] = np.nanmean(A_NM[np.where((A_N1e>45)&(A_N1e<=46))])
# B_N0[56] = np.nanmean(A_NM[np.where((A_N1e>46)&(A_N1e<=47))])
# B_N0[57] = np.nanmean(A_NM[np.where((A_N1e>47)&(A_N1e<=48))])
# B_N0[58] = np.nanmean(A_NM[np.where((A_N1e>48)&(A_N1e<=49))])
# B_N0[59] = np.nanmean(A_NM[np.where((A_N1e>49)&(A_N1e<=50))])
# B_N0[60] = np.nanmean(A_NM[np.where((A_N1e>50)&(A_N1e<=51))])
# B_N0[61] = np.nanmean(A_NM[np.where((A_N1e>51)&(A_N1e<=52))])
# B_N0[62] = np.nanmean(A_NM[np.where((A_N1e>52)&(A_N1e<=53))])
# B_N0[63] = np.nanmean(A_NM[np.where((A_N1e>53)&(A_N1e<=54))])
# B_N0[64] = np.nanmean(A_NM[np.where((A_N1e>54)&(A_N1e<=55))])
# B_N0[65] = np.nanmean(A_NM[np.where((A_N1e>55)&(A_N1e<=56))])
# B_N0[66] = np.nanmean(A_NM[np.where((A_N1e>56)&(A_N1e<=57))])
# B_N0[67] = np.nanmean(A_NM[np.where((A_N1e>57)&(A_N1e<=58))])
# B_N0[68] = np.nanmean(A_NM[np.where((A_N1e>59)&(A_N1e<=60))])
# B_N0[69] = np.nanmean(A_NM[np.where((A_N1e>60)&(A_N1e<=61))])
# B_N0[70] = np.nanmean(A_NM[np.where((A_N1e>61)&(A_N1e<=62))])

# 2010-2019
B_N0 = np.zeros((11))
B_N0[0] = np.nanmean(
    A_NM[np.where((A_N1e > -10) & (A_N1e <= -5))]
)
B_N0[1] = np.nanmean(A_NM[np.where((A_N1e > -5) & (A_N1e <= 0))])
B_N0[2] = np.nanmean(A_NM[np.where((A_N1e > 0) & (A_N1e <= 5))])
B_N0[3] = np.nanmean(A_NM[np.where((A_N1e > 5) & (A_N1e <= 10))])
B_N0[4] = np.nanmean(A_NM[np.where((A_N1e > 10) & (A_N1e <= 15))])
B_N0[5] = np.nanmean(A_NM[np.where((A_N1e > 15) & (A_N1e <= 20))])
B_N0[6] = np.nanmean(A_NM[np.where((A_N1e > 20) & (A_N1e <= 25))])
B_N0[7] = np.nanmean(A_NM[np.where((A_N1e > 25) & (A_N1e <= 30))])
B_N0[8] = np.nanmean(A_NM[np.where((A_N1e > 30) & (A_N1e <= 35))])
B_N0[9] = np.nanmean(A_NM[np.where((A_N1e > 35) & (A_N1e <= 40))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1e>40)&(A_N1e<=45))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1e>45)&(A_N1e<=50))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1e>50)&(A_N1e<=55))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1e>55)&(A_N1e<=60))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1e>60)&(A_N1e<=65))])

# 2011
B_N1 = np.zeros((11))
B_N1[0] = np.nanmean(
    A_N40[np.where((A_N14 > -10) & (A_N14 <= -5))]
)
B_N1[1] = np.nanmean(A_N40[np.where((A_N14 > -5) & (A_N14 <= 0))])
B_N1[2] = np.nanmean(A_N40[np.where((A_N14 > 0) & (A_N14 <= 5))])
B_N1[3] = np.nanmean(A_N40[np.where((A_N14 > 5) & (A_N14 <= 10))])
B_N1[4] = np.nanmean(
    A_N40[np.where((A_N14 > 10) & (A_N14 <= 15))]
)
B_N1[5] = np.nanmean(
    A_N40[np.where((A_N14 > 15) & (A_N14 <= 20))]
)
B_N1[6] = np.nanmean(
    A_N40[np.where((A_N14 > 20) & (A_N14 <= 25))]
)
B_N1[7] = np.nanmean(
    A_N40[np.where((A_N14 > 25) & (A_N14 <= 30))]
)
B_N1[8] = np.nanmean(
    A_N40[np.where((A_N14 > 30) & (A_N14 <= 35))]
)
B_N1[9] = np.nanmean(
    A_N40[np.where((A_N14 > 35) & (A_N14 <= 40))]
)
B_N1[10] = np.nanmean(
    A_N40[np.where((A_N14 > 40) & (A_N14 <= 45))]
)
# B_N1[11] = np.nanmean(A_N40[np.where((A_N14>45)&(A_N14<=50))])
# B_N1[12] = np.nanmean(A_N40[np.where((A_N14>50)&(A_N14<=55))])
# B_N1[13] = np.nanmean(A_N40[np.where((A_N14>55)&(A_N14<=60))])
# B_N1[14] = np.nanmean(A_N40[np.where((A_N14>60)&(A_N14<=65))])

# 2012
B_N2 = np.zeros((11))
B_N2[0] = np.nanmean(
    A_N50[np.where((A_N15 > -10) & (A_N15 <= -5))]
)
B_N2[1] = np.nanmean(A_N50[np.where((A_N15 > -5) & (A_N15 <= 0))])
B_N2[2] = np.nanmean(A_N50[np.where((A_N15 > 0) & (A_N15 <= 5))])
B_N2[3] = np.nanmean(A_N50[np.where((A_N15 > 5) & (A_N15 <= 10))])
B_N2[4] = np.nanmean(
    A_N50[np.where((A_N15 > 10) & (A_N15 <= 15))]
)
B_N2[5] = np.nanmean(
    A_N50[np.where((A_N15 > 15) & (A_N15 <= 20))]
)
B_N2[6] = np.nanmean(
    A_N50[np.where((A_N15 > 20) & (A_N15 <= 25))]
)
B_N2[7] = np.nanmean(
    A_N50[np.where((A_N15 > 25) & (A_N15 <= 30))]
)
B_N2[8] = np.nanmean(
    A_N50[np.where((A_N15 > 30) & (A_N15 <= 35))]
)
B_N2[9] = np.nanmean(
    A_N50[np.where((A_N15 > 35) & (A_N15 <= 40))]
)
B_N2[10] = np.nanmean(
    A_N50[np.where((A_N15 > 40) & (A_N15 <= 45))]
)
# B_N2[11] = np.nanmean(A_N50[np.where((A_N15>45)&(A_N15<=50))])
# B_N2[12] = np.nanmean(A_N50[np.where((A_N15>50)&(A_N15<=55))])
# B_N2[13] = np.nanmean(A_N50[np.where((A_N15>55)&(A_N15<=60))])
# B_N2[14] = np.nanmean(A_N50[np.where((A_N15>60)&(A_N15<=65))])

# 2013
B_N3 = np.zeros((11))
B_N3[0] = np.nanmean(
    A_N60[np.where((A_N16 > -10) & (A_N16 <= -5))]
)
B_N3[1] = np.nanmean(A_N60[np.where((A_N16 > -5) & (A_N16 <= 0))])
B_N3[2] = np.nanmean(A_N60[np.where((A_N16 > 0) & (A_N16 <= 5))])
B_N3[3] = np.nanmean(A_N60[np.where((A_N16 > 5) & (A_N16 <= 10))])
B_N3[4] = np.nanmean(
    A_N60[np.where((A_N16 > 10) & (A_N16 <= 15))]
)
B_N3[5] = np.nanmean(
    A_N60[np.where((A_N16 > 15) & (A_N16 <= 20))]
)
B_N3[6] = np.nanmean(
    A_N60[np.where((A_N16 > 20) & (A_N16 <= 25))]
)
B_N3[7] = np.nanmean(
    A_N60[np.where((A_N16 > 25) & (A_N16 <= 30))]
)
B_N3[8] = np.nanmean(
    A_N60[np.where((A_N16 > 30) & (A_N16 <= 35))]
)
B_N3[9] = np.nanmean(
    A_N60[np.where((A_N16 > 35) & (A_N16 <= 40))]
)
B_N3[10] = np.nanmean(
    A_N60[np.where((A_N16 > 40) & (A_N16 <= 45))]
)
# B_N3[11] = np.nanmean(A_N60[np.where((A_N16>45)&(A_N16<=50))])
# B_N3[12] = np.nanmean(A_N60[np.where((A_N16>50)&(A_N16<=55))])
# B_N3[13] = np.nanmean(A_N60[np.where((A_N16>55)&(A_N16<=60))])
# B_N3[14] = np.nanmean(A_N60[np.where((A_N16>60)&(A_N16<=65))])

# 2014
B_N4 = np.zeros((11))
B_N4[0] = np.nanmean(
    A_N70[np.where((A_N17 > -10) & (A_N17 <= -5))]
)
B_N4[1] = np.nanmean(A_N70[np.where((A_N17 > -5) & (A_N17 <= 0))])
B_N4[2] = np.nanmean(A_N70[np.where((A_N17 > 0) & (A_N17 <= 5))])
B_N4[3] = np.nanmean(A_N70[np.where((A_N17 > 5) & (A_N17 <= 10))])
B_N4[4] = np.nanmean(
    A_N70[np.where((A_N17 > 10) & (A_N17 <= 15))]
)
B_N4[5] = np.nanmean(
    A_N70[np.where((A_N17 > 15) & (A_N17 <= 20))]
)
B_N4[6] = np.nanmean(
    A_N70[np.where((A_N17 > 20) & (A_N17 <= 25))]
)
B_N4[7] = np.nanmean(
    A_N70[np.where((A_N17 > 25) & (A_N17 <= 30))]
)
B_N4[8] = np.nanmean(
    A_N70[np.where((A_N17 > 30) & (A_N17 <= 35))]
)
B_N4[9] = np.nanmean(
    A_N70[np.where((A_N17 > 35) & (A_N17 <= 40))]
)
B_N4[10] = np.nanmean(
    A_N70[np.where((A_N17 > 40) & (A_N17 <= 45))]
)
# B_N4[11] = np.nanmean(A_N70[np.where((A_N17>45)&(A_N17<=50))])
# B_N4[12] = np.nanmean(A_N70[np.where((A_N17>50)&(A_N17<=55))])
# B_N4[13] = np.nanmean(A_N70[np.where((A_N17>55)&(A_N17<=60))])
# B_N4[14] = np.nanmean(A_N70[np.where((A_N17>60)&(A_N17<=65))])

# 2015
B_N5 = np.zeros((11))
B_N5[0] = np.nanmean(
    A_N80[np.where((A_N18 > -10) & (A_N18 <= -5))]
)
B_N5[1] = np.nanmean(A_N80[np.where((A_N18 > -5) & (A_N18 <= 0))])
B_N5[2] = np.nanmean(A_N80[np.where((A_N18 > 0) & (A_N18 <= 5))])
B_N5[3] = np.nanmean(A_N80[np.where((A_N18 > 5) & (A_N18 <= 10))])
B_N5[4] = np.nanmean(
    A_N80[np.where((A_N18 > 10) & (A_N18 <= 15))]
)
B_N5[5] = np.nanmean(
    A_N80[np.where((A_N18 > 15) & (A_N18 <= 20))]
)
B_N5[6] = np.nanmean(
    A_N80[np.where((A_N18 > 20) & (A_N18 <= 25))]
)
B_N5[7] = np.nanmean(
    A_N80[np.where((A_N18 > 25) & (A_N18 <= 30))]
)
B_N5[8] = np.nanmean(
    A_N80[np.where((A_N18 > 30) & (A_N18 <= 35))]
)
B_N5[9] = np.nanmean(
    A_N80[np.where((A_N18 > 35) & (A_N18 <= 40))]
)
B_N5[10] = np.nanmean(
    A_N80[np.where((A_N18 > 40) & (A_N18 <= 45))]
)
# B_N5[11] = np.nanmean(A_N80[np.where((A_N18>45)&(A_N18<=50))])
# B_N5[12] = np.nanmean(A_N80[np.where((A_N18>50)&(A_N18<=55))])
# B_N5[13] = np.nanmean(A_N80[np.where((A_N18>55)&(A_N18<=60))])
# B_N5[14] = np.nanmean(A_N80[np.where((A_N18>60)&(A_N18<=65))])

# 2016
B_N6 = np.zeros((11))
B_N6[0] = np.nanmean(
    A_N90[np.where((A_N19 > -10) & (A_N19 <= -5))]
)
B_N6[1] = np.nanmean(A_N90[np.where((A_N19 > -5) & (A_N19 <= 0))])
B_N6[2] = np.nanmean(A_N90[np.where((A_N19 > 0) & (A_N19 <= 5))])
B_N6[3] = np.nanmean(A_N90[np.where((A_N19 > 5) & (A_N19 <= 10))])
B_N6[4] = np.nanmean(
    A_N90[np.where((A_N19 > 10) & (A_N19 <= 15))]
)
B_N6[5] = np.nanmean(
    A_N90[np.where((A_N19 > 15) & (A_N19 <= 20))]
)
B_N6[6] = np.nanmean(
    A_N90[np.where((A_N19 > 20) & (A_N19 <= 25))]
)
B_N6[7] = np.nanmean(
    A_N90[np.where((A_N19 > 25) & (A_N19 <= 30))]
)
B_N6[8] = np.nanmean(
    A_N90[np.where((A_N19 > 30) & (A_N19 <= 35))]
)
B_N6[9] = np.nanmean(
    A_N90[np.where((A_N19 > 35) & (A_N19 <= 40))]
)
B_N6[10] = np.nanmean(
    A_N90[np.where((A_N19 > 40) & (A_N19 <= 45))]
)
# B_N6[11] = np.nanmean(A_N90[np.where((A_N19>45)&(A_N19<=50))])
# B_N6[12] = np.nanmean(A_N90[np.where((A_N19>50)&(A_N19<=55))])
# B_N6[13] = np.nanmean(A_N90[np.where((A_N19>55)&(A_N19<=60))])
# B_N6[14] = np.nanmean(A_N90[np.where((A_N19>60)&(A_N19<=65))])

# 2017
B_N7 = np.zeros((11))
B_N7[0] = np.nanmean(
    A_N100[np.where((A_N110 > -10) & (A_N110 <= -5))]
)
B_N7[1] = np.nanmean(
    A_N100[np.where((A_N110 > -5) & (A_N110 <= 0))]
)
B_N7[2] = np.nanmean(
    A_N100[np.where((A_N110 > 0) & (A_N110 <= 5))]
)
B_N7[3] = np.nanmean(
    A_N100[np.where((A_N110 > 5) & (A_N110 <= 10))]
)
B_N7[4] = np.nanmean(
    A_N100[np.where((A_N110 > 10) & (A_N110 <= 15))]
)
B_N7[5] = np.nanmean(
    A_N100[np.where((A_N110 > 15) & (A_N110 <= 20))]
)
B_N7[6] = np.nanmean(
    A_N100[np.where((A_N110 > 20) & (A_N110 <= 25))]
)
B_N7[7] = np.nanmean(
    A_N100[np.where((A_N110 > 25) & (A_N110 <= 30))]
)
B_N7[8] = np.nanmean(
    A_N100[np.where((A_N110 > 30) & (A_N110 <= 35))]
)
B_N7[9] = np.nanmean(
    A_N100[np.where((A_N110 > 35) & (A_N110 <= 40))]
)
B_N7[10] = np.nanmean(
    A_N100[np.where((A_N110 > 40) & (A_N110 <= 45))]
)
# B_N7[11] = np.nanmean(A_N100[np.where((A_N110>45)&(A_N110<=50))])
# B_N7[12] = np.nanmean(A_N100[np.where((A_N110>50)&(A_N110<=55))])
# B_N7[13] = np.nanmean(A_N100[np.where((A_N110>55)&(A_N110<=60))])
# B_N7[14] = np.nanmean(A_N100[np.where((A_N110>60)&(A_N110<=65))])

# 2018
B_N8 = np.zeros((11))
B_N8[0] = np.nanmean(
    A_N101[np.where((A_N111 > -10) & (A_N111 <= -5))]
)
B_N8[1] = np.nanmean(
    A_N101[np.where((A_N111 > -5) & (A_N111 <= 0))]
)
B_N8[2] = np.nanmean(
    A_N101[np.where((A_N111 > 0) & (A_N111 <= 5))]
)
B_N8[3] = np.nanmean(
    A_N101[np.where((A_N111 > 5) & (A_N111 <= 10))]
)
B_N8[4] = np.nanmean(
    A_N101[np.where((A_N111 > 10) & (A_N111 <= 15))]
)
B_N8[5] = np.nanmean(
    A_N101[np.where((A_N111 > 15) & (A_N111 <= 20))]
)
B_N8[6] = np.nanmean(
    A_N101[np.where((A_N111 > 20) & (A_N111 <= 25))]
)
B_N8[7] = np.nanmean(
    A_N101[np.where((A_N111 > 25) & (A_N111 <= 30))]
)
B_N8[8] = np.nanmean(
    A_N101[np.where((A_N111 > 30) & (A_N111 <= 35))]
)
B_N8[9] = np.nanmean(
    A_N101[np.where((A_N111 > 35) & (A_N111 <= 40))]
)
B_N8[10] = np.nanmean(
    A_N101[np.where((A_N111 > 40) & (A_N111 <= 45))]
)
# B_N8[11] = np.nanmean(A_N101[np.where((A_N111>45)&(A_N111<=50))])
# B_N8[12] = np.nanmean(A_N101[np.where((A_N111>50)&(A_N111<=55))])
# B_N8[13] = np.nanmean(A_N101[np.where((A_N111>55)&(A_N111<=60))])
# B_N8[14] = np.nanmean(A_N101[np.where((A_N111>60)&(A_N111<=65))])

# 2019
B_N9 = np.zeros((11))
B_N9[0] = np.nanmean(
    A_N102[np.where((A_N112 > -10) & (A_N112 <= -5))]
)
B_N9[1] = np.nanmean(
    A_N102[np.where((A_N112 > -5) & (A_N112 <= 0))]
)
B_N9[2] = np.nanmean(
    A_N102[np.where((A_N112 > 0) & (A_N112 <= 5))]
)
B_N9[3] = np.nanmean(
    A_N102[np.where((A_N112 > 5) & (A_N112 <= 10))]
)
B_N9[4] = np.nanmean(
    A_N102[np.where((A_N112 > 10) & (A_N112 <= 15))]
)
B_N9[5] = np.nanmean(
    A_N102[np.where((A_N112 > 15) & (A_N112 <= 20))]
)
B_N9[6] = np.nanmean(
    A_N102[np.where((A_N112 > 20) & (A_N112 <= 25))]
)
B_N9[7] = np.nanmean(
    A_N102[np.where((A_N112 > 25) & (A_N112 <= 30))]
)
B_N9[8] = np.nanmean(
    A_N102[np.where((A_N112 > 30) & (A_N112 <= 35))]
)
B_N9[9] = np.nanmean(
    A_N102[np.where((A_N112 > 35) & (A_N112 <= 40))]
)
B_N9[10] = np.nanmean(
    A_N102[np.where((A_N112 > 40) & (A_N112 <= 45))]
)
# B_N9[11] = np.nanmean(A_N102[np.where((A_N112>45)&(A_N112<=50))])
# B_N9[12] = np.nanmean(A_N102[np.where((A_N112>50)&(A_N112<=55))])
# B_N9[13] = np.nanmean(A_N102[np.where((A_N112>55)&(A_N112<=60))])
# B_N9[14] = np.nanmean(A_N102[np.where((A_N112>60)&(A_N112<=65))])

# 2020
B_N10 = np.zeros((11))
B_N10[0] = np.nanmean(
    A_N103[np.where((A_N113 > -10) & (A_N113 <= -5))]
)
B_N10[1] = np.nanmean(
    A_N103[np.where((A_N113 > -5) & (A_N113 <= 0))]
)
B_N10[2] = np.nanmean(
    A_N103[np.where((A_N113 > 0) & (A_N113 <= 5))]
)
B_N10[3] = np.nanmean(
    A_N103[np.where((A_N113 > 5) & (A_N113 <= 10))]
)
B_N10[4] = np.nanmean(
    A_N103[np.where((A_N113 > 10) & (A_N113 <= 15))]
)
B_N10[5] = np.nanmean(
    A_N103[np.where((A_N113 > 15) & (A_N113 <= 20))]
)
B_N10[6] = np.nanmean(
    A_N103[np.where((A_N113 > 20) & (A_N113 <= 25))]
)
B_N10[7] = np.nanmean(
    A_N103[np.where((A_N113 > 25) & (A_N113 <= 30))]
)
B_N10[8] = np.nanmean(
    A_N103[np.where((A_N113 > 30) & (A_N113 <= 35))]
)
B_N10[9] = np.nanmean(
    A_N103[np.where((A_N113 > 35) & (A_N113 <= 40))]
)
B_N10[10] = np.nanmean(
    A_N103[np.where((A_N113 > 40) & (A_N113 <= 45))]
)
# B_N10[11] = np.nanmean(A_N103[np.where((A_N113>45)&(A_N113<=50))])
# B_N10[12] = np.nanmean(A_N103[np.where((A_N113>50)&(A_N113<=55))])
# B_N10[13] = np.nanmean(A_N103[np.where((A_N113>55)&(A_N113<=60))])
# B_N10[14] = np.nanmean(A_N103[np.where((A_N113>60)&(A_N113<=65))])


PCA = np.arange(-1.3, 4.2, 0.5)
# plt.plot(PCA, B_N,label='2010')
# plt.plot(PCA, B_N0,color='blue', label='2011-2019')
plt.plot(PCA, B_N1, label="2011", alpha=0.5)
plt.plot(PCA, B_N2, label="2012", alpha=0.5)
plt.plot(PCA, B_N3, label="2013", alpha=0.5)
plt.plot(PCA, B_N4, label="2014", alpha=0.5)
plt.plot(PCA, B_N5, label="2015", alpha=0.5)
plt.plot(PCA, B_N6, label="2016", alpha=0.5)
plt.plot(PCA, B_N7, label="2017", alpha=0.5)
plt.plot(PCA, B_N8, color="black", label="2018", alpha=0.5)
plt.plot(PCA, B_N9, color="green", label="2019", alpha=0.5)
plt.plot(
    PCA, B_N10, color="red", label="2020", linewidth=3, ls="-."
)

plt.legend()
plt.title("EOF-CERES_Cldarea", fontsize=18)
plt.xlabel("EOF", fontsize=14)
plt.ylabel("CERES_Cldarea(%)", fontsize=14)

# sns.displot(A_N12,kind="kde")

# plt.subplot(111)
# hist, bin_edges = np.histogram(A_N12)
# plt.plot(hist)
