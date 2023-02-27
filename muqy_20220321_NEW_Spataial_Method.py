# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:30:13 2021

@author: Mu o(*￣▽￣*)ブ
"""

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# from mpl_toolkits.basemap import Basemap
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
import matplotlib.colors as colors
from scipy.stats import norm
from itertools import product
import time
from numba import jit
from numba import vectorize, float64


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

for i in range(0, 33):

    # year_str=str(2010).zfill(4)
    # month_str=str(1).zfill(2)
    # time_str0=year_str+month_str

    ERA_file = glob.glob(
        "G:\\ERA5_daily_stored per month_global_2\\ERA5_daily_"
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
        RH250 = RH[:, 7, :, :]
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
        W[j, :, :] = np.flipud(W[j, :, :] * 24 * 36)
        Z300[j, :, :] = np.flipud(Z300[j, :, :])
        Z250[j, :, :] = np.flipud(Z250[j, :, :])

        RH300_1 = np.array(RH300[j, :, :]).reshape(64800)
        RH250_1 = np.array(RH250[j, :, :]).reshape(64800)
        T250_1 = np.array(T250[j, :, :]).reshape(64800)
        T300_1 = np.array(T300[j, :, :]).reshape(64800)
        W_1 = np.array(W[j, :, :]).reshape(64800)
        Z300_1 = np.array(Z300[j, :, :]).reshape(64800)
        Z250_1 = np.array(Z250[j, :, :]).reshape(64800)

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

        # sita300 = np.array(mpcalc.potential_temperature(300. * units.mbar, T300_1 * units.kelvin))
        # sita250 = np.array(mpcalc.potential_temperature(250. * units.mbar, T250_1 * units.kelvin))
        # # stab = (sita300-sita250)/(Z_1-Z1_1)

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

A_N1 = np.zeros((59875200, 4))
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
A_N1 = A_N1.reshape(59875200)
A_N1 = A_N1.reshape(33, 28, 180, 360)
A_N1e = A_N1[0:30, :, :, :]  # 2010-2019
A_N13 = A_N1[0:3, :, :, :]  # 2010
A_N14 = A_N1[3:6, :, :, :]  # 2011
A_N15 = A_N1[6:9, :, :, :]  # 2012
A_N16 = A_N1[9:12, :, :, :]  # 2013
A_N17 = A_N1[12:15, :, :, :]  # 2014
A_N18 = A_N1[15:18, :, :, :]  # 2015
A_N19 = A_N1[18:21, :, :, :]  # 2016
A_N110 = A_N1[21:24, :, :, :]  # 2017
A_N111 = A_N1[24:27, :, :, :]  # 2018
A_N112 = A_N1[27:30, :, :, :]  # 2019
A_N113 = A_N1[30:33, :, :, :]  # 2020

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

A_N1 = A_N1.reshape(59875200)
A_N1e = A_N1e.reshape(54432000)
A_N13 = A_N13.reshape(5443200)
A_N14 = A_N14.reshape(5443200)
A_N15 = A_N15.reshape(5443200)
A_N16 = A_N16.reshape(5443200)
A_N17 = A_N17.reshape(5443200)
A_N18 = A_N18.reshape(5443200)
A_N19 = A_N19.reshape(5443200)
A_N110 = A_N110.reshape(5443200)
A_N111 = A_N111.reshape(5443200)
A_N112 = A_N112.reshape(5443200)
A_N113 = A_N113.reshape(5443200)

# A_N1=A_N1.reshape(924,180,360)
# A_N1mean = np.mean(A_N1.reshape(924,180,360),axis=0)
# A_N2020 = np.mean(A_N113.reshape(84,180,360),axis=0)
# A_N2019 = np.mean(A_N112.reshape(84,180,360),axis=0)
# A = np.mean(A.reshape(308,180,360),axis=0)
# A_D = A_N2019-A_N2020
# A_N102 = A_N20[27:30,:,:,:] #2019
# A_N103 = A_N20[30:33,:,:,:] #2020
# A_N2020C = np.mean(A_N103.reshape(84,180,360),axis=0)
# A_N2019C = np.mean(A_N102.reshape(84,180,360),axis=0)


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
    "G:\\CERES_highcloud\\02-05\\CERES_highcloud_" + "*.nc"
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
        iwp = file_obj.iwp_high_daily
        cldpress = file_obj.cldpress_top_high_daily
        cldphase = file_obj.cldphase_high_daily
        cldemissir = file_obj.cldemissir_high_daily

        cldarea = np.array(cldarea)
        cldicerad = np.array(cldicerad)
        cldtau = np.array(cldtau)
        iwp = np.array(iwp)
        cldpress = np.array(cldpress)
        cldphase = np.array(cldphase)
        cldemissir = np.array(cldemissir)

        for j in range(0, 28):

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldpress1 = cldpress[j, :, :].reshape(64800)
            cldphase1 = cldphase[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldpress1), axis=0)
            A_N25 = np.concatenate((A_N25, cldphase1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(28, 56):

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldpress1 = cldpress[j, :, :].reshape(64800)
            cldphase1 = cldphase[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldpress1), axis=0)
            A_N25 = np.concatenate((A_N25, cldphase1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(59, 87):

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldpress1 = cldpress[j, :, :].reshape(64800)
            cldphase1 = cldphase[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldpress1), axis=0)
            A_N25 = np.concatenate((A_N25, cldphase1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

    elif calendar.isleap(id_name) == True:

        file_obj = xr.open_dataset(FILE_NAME)
        lat = file_obj.lat
        lon = file_obj.lon
        # t = file_obj.time
        cldarea = file_obj.cldarea_high_daily
        cldicerad = file_obj.cldicerad_high_daily
        cldtau = file_obj.cldtau_high_daily
        iwp = file_obj.iwp_high_daily
        cldpress = file_obj.cldpress_top_high_daily
        cldphase = file_obj.cldphase_high_daily
        cldemissir = file_obj.cldemissir_high_daily
        cldarea = cldarea[:, :, :]
        cldicerad = cldicerad[:, :, :]
        cldtau = cldtau[:, :, :]
        cldpress = cldpress[:, :, :]
        iwp = iwp[:, :, :]
        cldphase = cldphase[:, :, :]
        cldemissir = cldemissir[:, :, :]
        cldarea = np.array(cldarea)
        cldicerad = np.array(cldicerad)
        cldtau = np.array(cldtau)
        iwp = np.array(iwp)
        cldpress = np.array(cldpress)
        cldphase = np.array(cldphase)
        cldemissir = np.array(cldemissir)

        for j in range(0, 28):

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldpress1 = cldpress[j, :, :].reshape(64800)
            cldphase1 = cldphase[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldpress1), axis=0)
            A_N25 = np.concatenate((A_N25, cldphase1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(29, 57):

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldpress1 = cldpress[j, :, :].reshape(64800)
            cldphase1 = cldphase[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldpress1), axis=0)
            A_N25 = np.concatenate((A_N25, cldphase1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)

        for j in range(60, 88):

            cldarea1 = cldarea[j, :, :].reshape(64800)
            cldicerad1 = cldicerad[j, :, :].reshape(64800)
            cldtau1 = cldtau[j, :, :].reshape(64800)
            iwp1 = iwp[j, :, :].reshape(64800)
            cldpress1 = cldpress[j, :, :].reshape(64800)
            cldphase1 = cldphase[j, :, :].reshape(64800)
            cldemissir1 = cldemissir[j, :, :].reshape(64800)
            A_N20 = np.concatenate((A_N20, cldarea1), axis=0)
            A_N21 = np.concatenate((A_N21, cldicerad1), axis=0)
            A_N22 = np.concatenate((A_N22, cldtau1), axis=0)
            A_N23 = np.concatenate((A_N23, iwp1), axis=0)
            A_N24 = np.concatenate((A_N24, cldpress1), axis=0)
            A_N25 = np.concatenate((A_N25, cldphase1), axis=0)
            A_N26 = np.concatenate((A_N26, cldemissir1), axis=0)


A_N20 = np.delete(A_N20, 0, axis=0)  # cldarea
A_N21 = np.delete(A_N21, 0, axis=0)  # cldicerad
A_N22 = np.delete(A_N22, 0, axis=0)  # cldtau
A_N23 = np.delete(A_N23, 0, axis=0)  # iwp
A_N24 = np.delete(A_N24, 0, axis=0)  # cldpress
A_N25 = np.delete(A_N25, 0, axis=0)  # cldphase
A_N26 = np.delete(A_N26, 0, axis=0)  # cldemissir

A_N20t = A_N23.reshape(
    33, 28, 180, 360
)  # Choose the variable used in the plot
A_N20t[A_N20t == -999] = np.nan

A_NM = A_N20t[0:30, :, :, :]  # 2010-2019
A_N30 = A_N20t[0:3, :, :, :]  # 2010
A_N40 = A_N20t[3:6, :, :, :]  # 2011
A_N50 = A_N20t[6:9, :, :, :]  # 2012
A_N60 = A_N20t[9:12, :, :, :]  # 2013
A_N70 = A_N20t[12:15, :, :, :]  # 2014
A_N80 = A_N20t[15:18, :, :, :]  # 2015
A_N90 = A_N20t[18:21, :, :, :]  # 2016
A_N100 = A_N20t[21:24, :, :, :]  # 2017
A_N101 = A_N20t[24:27, :, :, :]  # 2018
A_N102 = A_N20t[27:30, :, :, :]  # 2019
A_N103 = A_N20t[30:33, :, :, :]  # 2020

A_N20t = A_N20t.reshape(59875200)
A_NM = A_NM.reshape(54432000)
A_N30 = A_N30.reshape(5443200)
A_N40 = A_N40.reshape(5443200)
A_N50 = A_N50.reshape(5443200)
A_N60 = A_N60.reshape(5443200)
A_N70 = A_N70.reshape(5443200)
A_N80 = A_N80.reshape(5443200)
A_N90 = A_N90.reshape(5443200)
A_N100 = A_N100.reshape(5443200)
A_N101 = A_N101.reshape(5443200)
A_N102 = A_N102.reshape(5443200)
A_N103 = A_N103.reshape(5443200)


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

A_NNN2020 = np.zeros((84, 180, 360))
A_NNN2019 = np.zeros((84, 180, 360))
A_NNN2018 = np.zeros((84, 180, 360))
A_NNN2017 = np.zeros((84, 180, 360))
A_NNN2017 = np.zeros((84, 180, 360))
A_temp = np.zeros((84))

A_N60 = A_N60.reshape(84, 180, 360)
A_N16 = A_N16.reshape(84, 180, 360)
A_N70 = A_N70.reshape(84, 180, 360)
A_N17 = A_N17.reshape(84, 180, 360)
A_N80 = A_N80.reshape(84, 180, 360)
A_N18 = A_N18.reshape(84, 180, 360)
A_N90 = A_N90.reshape(84, 180, 360)
A_N19 = A_N19.reshape(84, 180, 360)
A_N100 = A_N100.reshape(84, 180, 360)
A_N110 = A_N110.reshape(84, 180, 360)
A_N101 = A_N101.reshape(84, 180, 360)
A_N111 = A_N111.reshape(84, 180, 360)
A_N103 = A_N103.reshape(84, 180, 360)
A_N113 = A_N113.reshape(84, 180, 360)
A_N102 = A_N102.reshape(84, 180, 360)
A_N112 = A_N112.reshape(84, 180, 360)
A_NM = A_NM.reshape(10, 84, 180, 360)
A_N1e = A_N1e.reshape(10, 84, 180, 360)

numlist1 = [i for i in range(0, 84)]
numlist2 = [i for i in range(0, 180)]
numlist3 = [i for i in range(0, 360)]


def Compute(m, n, t1, t2):
    A_NNN2020 = np.zeros((180, 360))
    A_NNN2019 = np.zeros((180, 360))
    A_NNN2018 = np.zeros((180, 360))
    A_NNN2017 = np.zeros((180, 360))
    A_NNN2016 = np.zeros((180, 360))
    A_NNN2015 = np.zeros((180, 360))
    A_NNN2014 = np.zeros((180, 360))
    A_NNN2013 = np.zeros((180, 360))
    A_temp = np.nanmean(
        A_NM[:, t1:t2, :, :][
            np.where(
                (A_N1e[:, t1:t2, :, :] >= (m * 0.1))
                & (A_N1e[:, t1:t2, :, :] < (n * 0.1))
            )
        ]
    )
    for num2, num3 in product(numlist2, numlist3):
        if (
            A_N103[t1:t2, num2, num3][
                np.where(
                    (A_N113[t1:t2, num2, num3] >= (m * 0.1))
                    & (A_N113[t1:t2, num2, num3] < (n * 0.1))
                )
            ].shape
            != 0
        ):
            A_NNN2020[num2, num3] = A_temp - np.nanmean(
                A_N103[t1:t2, num2, num3][
                    np.where(
                        (A_N113[t1:t2, num2, num3] >= (m * 0.1))
                        & (A_N113[t1:t2, num2, num3] < (n * 0.1))
                    )
                ]
            )
        else:
            A_NNN2020[num2, num3] = np.nan

    for num2, num3 in product(numlist2, numlist3):
        if (
            A_N102[t1:t2, num2, num3][
                np.where(
                    (A_N112[t1:t2, num2, num3] >= (m * 0.1))
                    & (A_N112[t1:t2, num2, num3] < (n * 0.1))
                )
            ].shape
            != 0
        ):
            A_NNN2019[num2, num3] = A_temp - np.nanmean(
                A_N102[t1:t2, num2, num3][
                    np.where(
                        (A_N112[t1:t2, num2, num3] >= (m * 0.1))
                        & (A_N112[t1:t2, num2, num3] < (n * 0.1))
                    )
                ]
            )
        else:
            A_NNN2019[num2, num3] = np.nan

    for num2, num3 in product(numlist2, numlist3):
        if (
            A_N101[t1:t2, num2, num3][
                np.where(
                    (A_N111[t1:t2, num2, num3] >= (m * 0.1))
                    & (A_N111[t1:t2, num2, num3] < (n * 0.1))
                )
            ].shape
            != 0
        ):
            A_NNN2018[num2, num3] = A_temp - np.nanmean(
                A_N101[t1:t2, num2, num3][
                    np.where(
                        (A_N111[t1:t2, num2, num3] >= (m * 0.1))
                        & (A_N111[t1:t2, num2, num3] < (n * 0.1))
                    )
                ]
            )
        else:
            A_NNN2018[num2, num3] = np.nan

    for num2, num3 in product(numlist2, numlist3):
        if (
            A_N100[t1:t2, num2, num3][
                np.where(
                    (A_N110[t1:t2, num2, num3] >= (m * 0.1))
                    & (A_N110[t1:t2, num2, num3] < (n * 0.1))
                )
            ].shape
            != 0
        ):
            A_NNN2017[num2, num3] = A_temp - np.nanmean(
                A_N100[t1:t2, num2, num3][
                    np.where(
                        (A_N110[t1:t2, num2, num3] >= (m * 0.1))
                        & (A_N110[t1:t2, num2, num3] < (n * 0.1))
                    )
                ]
            )
        else:
            A_NNN2017[num2, num3] = np.nan

    for num2, num3 in product(numlist2, numlist3):
        if (
            A_N90[t1:t2, num2, num3][
                np.where(
                    (A_N19[t1:t2, num2, num3] >= (m * 0.1))
                    & (A_N19[t1:t2, num2, num3] < (n * 0.1))
                )
            ].shape
            != 0
        ):
            A_NNN2016[num2, num3] = A_temp - np.nanmean(
                A_N90[t1:t2, num2, num3][
                    np.where(
                        (A_N19[t1:t2, num2, num3] >= (m * 0.1))
                        & (A_N19[t1:t2, num2, num3] < (n * 0.1))
                    )
                ]
            )
        else:
            A_NNN2016[num2, num3] = np.nan

    for num2, num3 in product(numlist2, numlist3):
        if (
            A_N80[t1:t2, num2, num3][
                np.where(
                    (A_N18[t1:t2, num2, num3] >= (m * 0.1))
                    & (A_N18[t1:t2, num2, num3] < (n * 0.1))
                )
            ].shape
            != 0
        ):
            A_NNN2015[num2, num3] = A_temp - np.nanmean(
                A_N80[t1:t2, num2, num3][
                    np.where(
                        (A_N18[t1:t2, num2, num3] >= (m * 0.1))
                        & (A_N18[t1:t2, num2, num3] < (n * 0.1))
                    )
                ]
            )
        else:
            A_NNN2015[num2, num3] = np.nan

    for num2, num3 in product(numlist2, numlist3):
        if (
            A_N70[t1:t2, num2, num3][
                np.where(
                    (A_N17[t1:t2, num2, num3] >= (m * 0.1))
                    & (A_N17[t1:t2, num2, num3] < (n * 0.1))
                )
            ].shape
            != 0
        ):
            A_NNN2014[num2, num3] = A_temp - np.nanmean(
                A_N70[t1:t2, num2, num3][
                    np.where(
                        (A_N17[t1:t2, num2, num3] >= (m * 0.1))
                        & (A_N17[t1:t2, num2, num3] < (n * 0.1))
                    )
                ]
            )
        else:
            A_NNN2014[num2, num3] = np.nan

    for num2, num3 in product(numlist2, numlist3):
        if (
            A_N60[t1:t2, num2, num3][
                np.where(
                    (A_N16[t1:t2, num2, num3] >= (m * 0.1))
                    & (A_N16[t1:t2, num2, num3] < (n * 0.1))
                )
            ].shape
            != 0
        ):
            A_NNN2013[num2, num3] = A_temp - np.nanmean(
                A_N60[t1:t2, num2, num3][
                    np.where(
                        (A_N16[t1:t2, num2, num3] >= (m * 0.1))
                        & (A_N16[t1:t2, num2, num3] < (n * 0.1))
                    )
                ]
            )
        else:
            A_NNN2013[num2, num3] = np.nan

    return (
        A_NNN2020,
        A_NNN2019,
        A_NNN2018,
        A_NNN2017,
        A_NNN2016,
        A_NNN2015,
        A_NNN2014,
        A_NNN2013,
    )


(
    A_NNN2020,
    A_NNN2019,
    A_NNN2018,
    A_NNN2017,
    A_NNN2016,
    A_NNN2015,
    A_NNN2014,
    A_NNN2013,
) = Compute(14, 20, 0, 84)

# def Compute(m,n):
#     A_temp = np.zeros((84))
#     A_NNN2020 = np.zeros((84,180,360))
#     A_NNN2019 = np.zeros((84,180,360))
#     A_NNN2018 = np.zeros((84,180,360))
#     A_NNN2017 = np.zeros((84,180,360))
#     for num1 in product(numlist1):
#         A_temp[num1] = np.nanmean(A_NM[:,num1,:,:][np.where((A_N1e[:,num1,:,:]\
#         >=(m*0.1))&(A_N1e[:,num1,:,:]<(n*0.1)))])
#     for num1, num2, num3 in product(numlist1, numlist2, numlist3):
#         if (m*0.1)<=A_N113[num1,num2,num3]<(n*0.1):
#             A_NNN2020[num1,num2,num3] = A_temp[num1]- A_N103[num1,num2,num3]
#         else :
#             A_NNN2020[num1,num2,num3] = np.nan

#     for num1, num2, num3 in product(numlist1, numlist2, numlist3):
#         if (m*0.1)<=A_N112[num1,num2,num3]<(n*0.1):
#             A_NNN2019[num1,num2,num3] = A_temp[num1]- A_N102[num1,num2,num3]
#         else :
#             A_NNN2019[num1,num2,num3] = np.nan

#     for num1, num2, num3 in product(numlist1, numlist2, numlist3):
#         if (m*0.1)<=A_N111[num1,num2,num3]<(n*0.1):
#             A_NNN2018[num1,num2,num3] = A_temp[num1]- A_N101[num1,num2,num3]
#         else :
#             A_NNN2018[num1,num2,num3] = np.nan

#     for num1, num2, num3 in product(numlist1, numlist2, numlist3):
#         if (m*0.1)<=A_N110[num1,num2,num3]<(n*0.1):
#             A_NNN2017[num1,num2,num3] = A_temp[num1]- A_N100[num1,num2,num3]
#         else :
#             A_NNN2017[num1,num2,num3] = np.nan

#     return A_NNN2020,A_NNN2019,A_NNN2018,A_NNN2017

# A_NNN2020,A_NNN2019,A_NNN2018,A_NNN2017 = Compute(14,20)

# def C2019():
#     for num1, num2, num3 in product(numlist2, numlist3, numlist4):
#         if (num1*0.1)<=A_N112[num1,num2,num3]<((num1+1)*0.1):
#             A_NNN2019[num1,num1,num2,num3] = A_temp[num1, num1]\
#             - A_N102[num1,num2,num3]
#         else :
#             A_NNN2019[num1,num1,num2,num3] = np.nan
#     return A_NNN2019

# def C2018():
#     for num1, num2, num3 in product(numlist2, numlist3, numlist4):
#         if (num1*0.1)<=A_N111[num1,num2,num3]<((num1+1)*0.1):
#             A_NNN2018[num1,num1,num2,num3] = A_temp[num1, num1] \
#             - A_N101[num1,num2,num3]
#         else :
#             A_NNN2018[num1,num1,num2,num3] = np.nan
#     return A_NNN2018

# def C2017():
#     for num1, num2, num3 in product(numlist2, numlist3, numlist4):
#         if (num1*0.1)<=A_N110[num1,num2,num3]<((num1+1)*0.1):
#             A_NNN2017[num1,num1,num2,num3] = A_temp[num1, num1] \
#             - A_N100[num1,num2,num3]
#         else :
#             A_NNN2017[num1,num1,num2,num3] = np.nan
#     return A_NNN2017

# start=time.time()
# A_NNN2020 = C2020()
# A_NNN2019 = C2019()
# A_NNN2018 = C2018()
# A_NNN2017 = C2017()
# end=time.time()
# print('Running time: %s Seconds'%(end-start))

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


def plotbyEOFgap(m, n, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    fig = plt.figure(figsize=(5, 10), dpi=150)
    # fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
    plt.rc("font", size=10, weight="bold")

    ax1 = plt.subplot(
        411, projection=ccrs.PlateCarree(central_longitude=180)
    )
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("black")
    cmap.set_under("black")
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(A_NNN2020[t1:t2, :, :], axis=(0)),
        transform=ccrs.PlateCarree(),
        vmax=800,
        vmin=-800,
        cmap=cmap,
    )
    gl = ax1.gridlines(
        linestyle="--", lw=0.5, alpha=0.7, draw_labels=True
    )
    gl.xlabels_top = False
    ax1.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " mean minus 2020 Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax2 = plt.subplot(
        412, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    a = ax2.pcolormesh(
        lon,
        lat,
        np.nanmean(A_NNN2019[t1:t2, :, :], axis=(0)),
        transform=ccrs.PlateCarree(),
        vmax=800,
        vmin=-800,
        cmap=cmap,
    )
    gl = ax2.gridlines(
        linestyle="--", lw=0.5, alpha=0.7, draw_labels=True
    )
    gl.xlabels_top = False
    ax2.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " mean minus 2019 Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax3 = plt.subplot(
        413, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    a = ax3.pcolormesh(
        lon,
        lat,
        np.nanmean(A_NNN2018[t1:t2, :, :], axis=(0)),
        transform=ccrs.PlateCarree(),
        vmax=800,
        vmin=-800,
        cmap=cmap,
    )
    gl = ax3.gridlines(
        linestyle="--", lw=0.5, alpha=0.7, draw_labels=True
    )
    gl.xlabels_top = False
    ax3.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " mean minus 2018 Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax4 = plt.subplot(
        414, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax4.coastlines(resolution="50m", lw=0.3)
    ax4.set_global()
    a = ax4.pcolormesh(
        lon,
        lat,
        np.nanmean(A_NNN2017[t1:t2, :, :], axis=(0)),
        transform=ccrs.PlateCarree(),
        vmax=800,
        vmin=-800,
        cmap=cmap,
    )
    gl = ax4.gridlines(
        linestyle="--", lw=0.5, alpha=0.7, draw_labels=True
    )
    gl.xlabels_top = False
    ax4.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " mean minus 2017 Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )
    fig.colorbar(
        a,
        ax=[ax1, ax2, ax3, ax4],
        location="right",
        shrink=0.9,
        extend="both",
    )
    plt.show()


plotbyEOFgap(14, 20, 0, 84)
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

# #2010-2019
# B_N0 = np.zeros((22))
# B_N0[0] = np.nanmean(A_NM[np.where((A_N1e>-1.3)&(A_N1e<=-1.05))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1e>-1.05)&(A_N1e<=-0.8))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1e>-0.8)&(A_N1e<=-0.55))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1e>-0.55)&(A_N1e<=-0.3))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1e>-0.3)&(A_N1e<=-0.05))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1e>-0.05)&(A_N1e<=0.2))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1e>0.2)&(A_N1e<=0.45))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1e>0.45)&(A_N1e<=0.7))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1e>0.7)&(A_N1e<=0.95))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1e>0.95)&(A_N1e<=1.2))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1e>1.2)&(A_N1e<=1.45))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1e>1.45)&(A_N1e<=1.7))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1e>1.7)&(A_N1e<=1.95))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1e>1.95)&(A_N1e<=2.2))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1e>2.2)&(A_N1e<=2.45))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1e>2.45)&(A_N1e<=2.7))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1e>2.7)&(A_N1e<=2.95))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1e>2.95)&(A_N1e<=3.2))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1e>3.2)&(A_N1e<=3.45))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1e>3.45)&(A_N1e<=3.7))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1e>3.7)&(A_N1e<=3.95))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1e>3.95)&(A_N1e<=4.2))])

# #2011
# B_N1 = np.zeros((22))
# B_N1[0] = np.nanmean(A_N40[np.where((A_N14>-1.3)&(A_N14<=-1.05))])
# B_N1[1] = np.nanmean(A_N40[np.where((A_N14>-1.05)&(A_N14<=-0.8))])
# B_N1[2] = np.nanmean(A_N40[np.where((A_N14>-0.8)&(A_N14<=-0.55))])
# B_N1[3] = np.nanmean(A_N40[np.where((A_N14>-0.55)&(A_N14<=-0.3))])
# B_N1[4] = np.nanmean(A_N40[np.where((A_N14>-0.3)&(A_N14<=-0.05))])
# B_N1[5] = np.nanmean(A_N40[np.where((A_N14>-0.05)&(A_N14<=0.2))])
# B_N1[6] = np.nanmean(A_N40[np.where((A_N14>0.2)&(A_N14<=0.45))])
# B_N1[7] = np.nanmean(A_N40[np.where((A_N14>0.45)&(A_N14<=0.7))])
# B_N1[8] = np.nanmean(A_N40[np.where((A_N14>0.7)&(A_N14<=0.95))])
# B_N1[9] = np.nanmean(A_N40[np.where((A_N14>0.95)&(A_N14<=1.2))])
# B_N1[10] = np.nanmean(A_N40[np.where((A_N14>1.2)&(A_N14<=1.45))])
# B_N1[11] = np.nanmean(A_N40[np.where((A_N14>1.45)&(A_N14<=1.7))])
# B_N1[12] = np.nanmean(A_N40[np.where((A_N14>1.7)&(A_N14<=1.95))])
# B_N1[13] = np.nanmean(A_N40[np.where((A_N14>1.95)&(A_N14<=2.2))])
# B_N1[14] = np.nanmean(A_N40[np.where((A_N14>2.2)&(A_N14<=2.45))])
# B_N1[15] = np.nanmean(A_N40[np.where((A_N14>2.45)&(A_N14<=2.7))])
# B_N1[16] = np.nanmean(A_N40[np.where((A_N14>2.7)&(A_N14<=2.95))])
# B_N1[17] = np.nanmean(A_N40[np.where((A_N14>2.95)&(A_N14<=3.2))])
# B_N1[18] = np.nanmean(A_N40[np.where((A_N14>3.2)&(A_N14<=3.45))])
# B_N1[19] = np.nanmean(A_N40[np.where((A_N14>3.45)&(A_N14<=3.7))])
# B_N1[20] = np.nanmean(A_N40[np.where((A_N14>3.7)&(A_N14<=3.95))])
# B_N1[21] = np.nanmean(A_N40[np.where((A_N14>3.95)&(A_N14<=4.2))])

# #2012
# B_N2 = np.zeros((22))
# B_N2[0] = np.nanmean(A_N50[np.where((A_N15>-1.3)&(A_N15<=-1.05))])
# B_N2[1] = np.nanmean(A_N50[np.where((A_N15>-1.05)&(A_N15<=-0.8))])
# B_N2[2] = np.nanmean(A_N50[np.where((A_N15>-0.8)&(A_N15<=-0.55))])
# B_N2[3] = np.nanmean(A_N50[np.where((A_N15>-0.55)&(A_N15<=-0.3))])
# B_N2[4] = np.nanmean(A_N50[np.where((A_N15>-0.3)&(A_N15<=-0.05))])
# B_N2[5] = np.nanmean(A_N50[np.where((A_N15>-0.05)&(A_N15<=0.2))])
# B_N2[6] = np.nanmean(A_N50[np.where((A_N15>0.2)&(A_N15<=0.45))])
# B_N2[7] = np.nanmean(A_N50[np.where((A_N15>0.45)&(A_N15<=0.7))])
# B_N2[8] = np.nanmean(A_N50[np.where((A_N15>0.7)&(A_N15<=0.95))])
# B_N2[9] = np.nanmean(A_N50[np.where((A_N15>0.95)&(A_N15<=1.2))])
# B_N2[10] = np.nanmean(A_N50[np.where((A_N15>1.2)&(A_N15<=1.45))])
# B_N2[11] = np.nanmean(A_N50[np.where((A_N15>1.45)&(A_N15<=1.7))])
# B_N2[12] = np.nanmean(A_N50[np.where((A_N15>1.7)&(A_N15<=1.95))])
# B_N2[13] = np.nanmean(A_N50[np.where((A_N15>1.95)&(A_N15<=2.2))])
# B_N2[14] = np.nanmean(A_N50[np.where((A_N15>2.2)&(A_N15<=2.45))])
# B_N2[15] = np.nanmean(A_N50[np.where((A_N15>2.45)&(A_N15<=2.7))])
# B_N2[16] = np.nanmean(A_N50[np.where((A_N15>2.7)&(A_N15<=2.95))])
# B_N2[17] = np.nanmean(A_N50[np.where((A_N15>2.95)&(A_N15<=3.2))])
# B_N2[18] = np.nanmean(A_N50[np.where((A_N15>3.2)&(A_N15<=3.45))])
# B_N2[19] = np.nanmean(A_N50[np.where((A_N15>3.45)&(A_N15<=3.7))])
# B_N2[20] = np.nanmean(A_N50[np.where((A_N15>3.7)&(A_N15<=3.95))])
# B_N2[21] = np.nanmean(A_N50[np.where((A_N15>3.95)&(A_N15<=4.2))])

# #2013
# B_N3 = np.zeros((22))
# B_N3[0] = np.nanmean(A_N60[np.where((A_N16>-1.3)&(A_N16<=-1.05))])
# B_N3[1] = np.nanmean(A_N60[np.where((A_N16>-1.05)&(A_N16<=-0.8))])
# B_N3[2] = np.nanmean(A_N60[np.where((A_N16>-0.8)&(A_N16<=-0.55))])
# B_N3[3] = np.nanmean(A_N60[np.where((A_N16>-0.55)&(A_N16<=-0.3))])
# B_N3[4] = np.nanmean(A_N60[np.where((A_N16>-0.3)&(A_N16<=-0.05))])
# B_N3[5] = np.nanmean(A_N60[np.where((A_N16>-0.05)&(A_N16<=0.2))])
# B_N3[6] = np.nanmean(A_N60[np.where((A_N16>0.2)&(A_N16<=0.45))])
# B_N3[7] = np.nanmean(A_N60[np.where((A_N16>0.45)&(A_N16<=0.7))])
# B_N3[8] = np.nanmean(A_N60[np.where((A_N16>0.7)&(A_N16<=0.95))])
# B_N3[9] = np.nanmean(A_N60[np.where((A_N16>0.95)&(A_N16<=1.2))])
# B_N3[10] = np.nanmean(A_N60[np.where((A_N16>1.2)&(A_N16<=1.45))])
# B_N3[11] = np.nanmean(A_N60[np.where((A_N16>1.45)&(A_N16<=1.7))])
# B_N3[12] = np.nanmean(A_N60[np.where((A_N16>1.7)&(A_N16<=1.95))])
# B_N3[13] = np.nanmean(A_N60[np.where((A_N16>1.95)&(A_N16<=2.2))])
# B_N3[14] = np.nanmean(A_N60[np.where((A_N16>2.2)&(A_N16<=2.45))])
# B_N3[15] = np.nanmean(A_N60[np.where((A_N16>2.45)&(A_N16<=2.7))])
# B_N3[16] = np.nanmean(A_N60[np.where((A_N16>2.7)&(A_N16<=2.95))])
# B_N3[17] = np.nanmean(A_N60[np.where((A_N16>2.95)&(A_N16<=3.2))])
# B_N3[18] = np.nanmean(A_N60[np.where((A_N16>3.2)&(A_N16<=3.45))])
# B_N3[19] = np.nanmean(A_N60[np.where((A_N16>3.45)&(A_N16<=3.7))])
# B_N3[20] = np.nanmean(A_N60[np.where((A_N16>3.7)&(A_N16<=3.95))])
# B_N3[21] = np.nanmean(A_N60[np.where((A_N16>3.95)&(A_N16<=4.2))])

# #2014
# B_N4 = np.zeros((22))
# B_N4[0] = np.nanmean(A_N70[np.where((A_N17>-1.3)&(A_N17<=-1.05))])
# B_N4[1] = np.nanmean(A_N70[np.where((A_N17>-1.05)&(A_N17<=-0.8))])
# B_N4[2] = np.nanmean(A_N70[np.where((A_N17>-0.8)&(A_N17<=-0.55))])
# B_N4[3] = np.nanmean(A_N70[np.where((A_N17>-0.55)&(A_N17<=-0.3))])
# B_N4[4] = np.nanmean(A_N70[np.where((A_N17>-0.3)&(A_N17<=-0.05))])
# B_N4[5] = np.nanmean(A_N70[np.where((A_N17>-0.05)&(A_N17<=0.2))])
# B_N4[6] = np.nanmean(A_N70[np.where((A_N17>0.2)&(A_N17<=0.45))])
# B_N4[7] = np.nanmean(A_N70[np.where((A_N17>0.45)&(A_N17<=0.7))])
# B_N4[8] = np.nanmean(A_N70[np.where((A_N17>0.7)&(A_N17<=0.95))])
# B_N4[9] = np.nanmean(A_N70[np.where((A_N17>0.95)&(A_N17<=1.2))])
# B_N4[10] = np.nanmean(A_N70[np.where((A_N17>1.2)&(A_N17<=1.45))])
# B_N4[11] = np.nanmean(A_N70[np.where((A_N17>1.45)&(A_N17<=1.7))])
# B_N4[12] = np.nanmean(A_N70[np.where((A_N17>1.7)&(A_N17<=1.95))])
# B_N4[13] = np.nanmean(A_N70[np.where((A_N17>1.95)&(A_N17<=2.2))])
# B_N4[14] = np.nanmean(A_N70[np.where((A_N17>2.2)&(A_N17<=2.45))])
# B_N4[15] = np.nanmean(A_N70[np.where((A_N17>2.45)&(A_N17<=2.7))])
# B_N4[16] = np.nanmean(A_N70[np.where((A_N17>2.7)&(A_N17<=2.95))])
# B_N4[17] = np.nanmean(A_N70[np.where((A_N17>2.95)&(A_N17<=3.2))])
# B_N4[18] = np.nanmean(A_N70[np.where((A_N17>3.2)&(A_N17<=3.45))])
# B_N4[19] = np.nanmean(A_N70[np.where((A_N17>3.45)&(A_N17<=3.7))])
# B_N4[20] = np.nanmean(A_N70[np.where((A_N17>3.7)&(A_N17<=3.95))])
# B_N4[21] = np.nanmean(A_N70[np.where((A_N17>3.95)&(A_N17<=4.2))])

# #2015
# B_N5 = np.zeros((22))
# B_N5[0] = np.nanmean(A_N80[np.where((A_N18>-1.3)&(A_N18<=-1.05))])
# B_N5[1] = np.nanmean(A_N80[np.where((A_N18>-1.05)&(A_N18<=-0.8))])
# B_N5[2] = np.nanmean(A_N80[np.where((A_N18>-0.8)&(A_N18<=-0.55))])
# B_N5[3] = np.nanmean(A_N80[np.where((A_N18>-0.55)&(A_N18<=-0.3))])
# B_N5[4] = np.nanmean(A_N80[np.where((A_N18>-0.3)&(A_N18<=-0.05))])
# B_N5[5] = np.nanmean(A_N80[np.where((A_N18>-0.05)&(A_N18<=0.2))])
# B_N5[6] = np.nanmean(A_N80[np.where((A_N18>0.2)&(A_N18<=0.45))])
# B_N5[7] = np.nanmean(A_N80[np.where((A_N18>0.45)&(A_N18<=0.7))])
# B_N5[8] = np.nanmean(A_N80[np.where((A_N18>0.7)&(A_N18<=0.95))])
# B_N5[9] = np.nanmean(A_N80[np.where((A_N18>0.95)&(A_N18<=1.2))])
# B_N5[10] = np.nanmean(A_N80[np.where((A_N18>1.2)&(A_N18<=1.45))])
# B_N5[11] = np.nanmean(A_N80[np.where((A_N18>1.45)&(A_N18<=1.7))])
# B_N5[12] = np.nanmean(A_N80[np.where((A_N18>1.7)&(A_N18<=1.95))])
# B_N5[13] = np.nanmean(A_N80[np.where((A_N18>1.95)&(A_N18<=2.2))])
# B_N5[14] = np.nanmean(A_N80[np.where((A_N18>2.2)&(A_N18<=2.45))])
# B_N5[15] = np.nanmean(A_N80[np.where((A_N18>2.45)&(A_N18<=2.7))])
# B_N5[16] = np.nanmean(A_N80[np.where((A_N18>2.7)&(A_N18<=2.95))])
# B_N5[17] = np.nanmean(A_N80[np.where((A_N18>2.95)&(A_N18<=3.2))])
# B_N5[18] = np.nanmean(A_N80[np.where((A_N18>3.2)&(A_N18<=3.45))])
# B_N5[19] = np.nanmean(A_N80[np.where((A_N18>3.45)&(A_N18<=3.7))])
# B_N5[20] = np.nanmean(A_N80[np.where((A_N18>3.7)&(A_N18<=3.95))])
# B_N5[21] = np.nanmean(A_N80[np.where((A_N18>3.95)&(A_N18<=4.2))])

# #2016
# B_N6 = np.zeros((22))
# B_N6[0] = np.nanmean(A_N90[np.where((A_N19>-1.3)&(A_N19<=-1.05))])
# B_N6[1] = np.nanmean(A_N90[np.where((A_N19>-1.05)&(A_N19<=-0.8))])
# B_N6[2] = np.nanmean(A_N90[np.where((A_N19>-0.8)&(A_N19<=-0.55))])
# B_N6[3] = np.nanmean(A_N90[np.where((A_N19>-0.55)&(A_N19<=-0.3))])
# B_N6[4] = np.nanmean(A_N90[np.where((A_N19>-0.3)&(A_N19<=-0.05))])
# B_N6[5] = np.nanmean(A_N90[np.where((A_N19>-0.05)&(A_N19<=0.2))])
# B_N6[6] = np.nanmean(A_N90[np.where((A_N19>0.2)&(A_N19<=0.45))])
# B_N6[7] = np.nanmean(A_N90[np.where((A_N19>0.45)&(A_N19<=0.7))])
# B_N6[8] = np.nanmean(A_N90[np.where((A_N19>0.7)&(A_N19<=0.95))])
# B_N6[9] = np.nanmean(A_N90[np.where((A_N19>0.95)&(A_N19<=1.2))])
# B_N6[10] = np.nanmean(A_N90[np.where((A_N19>1.2)&(A_N19<=1.45))])
# B_N6[11] = np.nanmean(A_N90[np.where((A_N19>1.45)&(A_N19<=1.7))])
# B_N6[12] = np.nanmean(A_N90[np.where((A_N19>1.7)&(A_N19<=1.95))])
# B_N6[13] = np.nanmean(A_N90[np.where((A_N19>1.95)&(A_N19<=2.2))])
# B_N6[14] = np.nanmean(A_N90[np.where((A_N19>2.2)&(A_N19<=2.45))])
# B_N6[15] = np.nanmean(A_N90[np.where((A_N19>2.45)&(A_N19<=2.7))])
# B_N6[16] = np.nanmean(A_N90[np.where((A_N19>2.7)&(A_N19<=2.95))])
# B_N6[17] = np.nanmean(A_N90[np.where((A_N19>2.95)&(A_N19<=3.2))])
# B_N6[18] = np.nanmean(A_N90[np.where((A_N19>3.2)&(A_N19<=3.45))])
# B_N6[19] = np.nanmean(A_N90[np.where((A_N19>3.45)&(A_N19<=3.7))])
# B_N6[20] = np.nanmean(A_N90[np.where((A_N19>3.7)&(A_N19<=3.95))])
# B_N6[21] = np.nanmean(A_N90[np.where((A_N19>3.95)&(A_N19<=4.2))])

# #2017
# B_N7 = np.zeros((22))
# B_N7[0] = np.nanmean(A_N100[np.where((A_N110>-1.3)&(A_N110<=-1.05))])
# B_N7[1] = np.nanmean(A_N100[np.where((A_N110>-1.05)&(A_N110<=-0.8))])
# B_N7[2] = np.nanmean(A_N100[np.where((A_N110>-0.8)&(A_N110<=-0.55))])
# B_N7[3] = np.nanmean(A_N100[np.where((A_N110>-0.55)&(A_N110<=-0.3))])
# B_N7[4] = np.nanmean(A_N100[np.where((A_N110>-0.3)&(A_N110<=-0.05))])
# B_N7[5] = np.nanmean(A_N100[np.where((A_N110>-0.05)&(A_N110<=0.2))])
# B_N7[6] = np.nanmean(A_N100[np.where((A_N110>0.2)&(A_N110<=0.45))])
# B_N7[7] = np.nanmean(A_N100[np.where((A_N110>0.45)&(A_N110<=0.7))])
# B_N7[8] = np.nanmean(A_N100[np.where((A_N110>0.7)&(A_N110<=0.95))])
# B_N7[9] = np.nanmean(A_N100[np.where((A_N110>0.95)&(A_N110<=1.2))])
# B_N7[10] = np.nanmean(A_N100[np.where((A_N110>1.2)&(A_N110<=1.45))])
# B_N7[11] = np.nanmean(A_N100[np.where((A_N110>1.45)&(A_N110<=1.7))])
# B_N7[12] = np.nanmean(A_N100[np.where((A_N110>1.7)&(A_N110<=1.95))])
# B_N7[13] = np.nanmean(A_N100[np.where((A_N110>1.95)&(A_N110<=2.2))])
# B_N7[14] = np.nanmean(A_N100[np.where((A_N110>2.2)&(A_N110<=2.45))])
# B_N7[15] = np.nanmean(A_N100[np.where((A_N110>2.45)&(A_N110<=2.7))])
# B_N7[16] = np.nanmean(A_N100[np.where((A_N110>2.7)&(A_N110<=2.95))])
# B_N7[17] = np.nanmean(A_N100[np.where((A_N110>2.95)&(A_N110<=3.2))])
# B_N7[18] = np.nanmean(A_N100[np.where((A_N110>3.2)&(A_N110<=3.45))])
# B_N7[19] = np.nanmean(A_N100[np.where((A_N110>3.45)&(A_N110<=3.7))])
# B_N7[20] = np.nanmean(A_N100[np.where((A_N110>3.7)&(A_N110<=3.95))])
# B_N7[21] = np.nanmean(A_N100[np.where((A_N110>3.95)&(A_N110<=4.2))])

# #2018
# B_N8 = np.zeros((22))
# B_N8[0] = np.nanmean(A_N101[np.where((A_N111>-1.3)&(A_N111<=-1.05))])
# B_N8[1] = np.nanmean(A_N101[np.where((A_N111>-1.05)&(A_N111<=-0.8))])
# B_N8[2] = np.nanmean(A_N101[np.where((A_N111>-0.8)&(A_N111<=-0.55))])
# B_N8[3] = np.nanmean(A_N101[np.where((A_N111>-0.55)&(A_N111<=-0.3))])
# B_N8[4] = np.nanmean(A_N101[np.where((A_N111>-0.3)&(A_N111<=-0.05))])
# B_N8[5] = np.nanmean(A_N101[np.where((A_N111>-0.05)&(A_N111<=0.2))])
# B_N8[6] = np.nanmean(A_N101[np.where((A_N111>0.2)&(A_N111<=0.45))])
# B_N8[7] = np.nanmean(A_N101[np.where((A_N111>0.45)&(A_N111<=0.7))])
# B_N8[8] = np.nanmean(A_N101[np.where((A_N111>0.7)&(A_N111<=0.95))])
# B_N8[9] = np.nanmean(A_N101[np.where((A_N111>0.95)&(A_N111<=1.2))])
# B_N8[10] = np.nanmean(A_N101[np.where((A_N111>1.2)&(A_N111<=1.45))])
# B_N8[11] = np.nanmean(A_N101[np.where((A_N111>1.45)&(A_N111<=1.7))])
# B_N8[12] = np.nanmean(A_N101[np.where((A_N111>1.7)&(A_N111<=1.95))])
# B_N8[13] = np.nanmean(A_N101[np.where((A_N111>1.95)&(A_N111<=2.2))])
# B_N8[14] = np.nanmean(A_N101[np.where((A_N111>2.2)&(A_N111<=2.45))])
# B_N8[15] = np.nanmean(A_N101[np.where((A_N111>2.45)&(A_N111<=2.7))])
# B_N8[16] = np.nanmean(A_N101[np.where((A_N111>2.7)&(A_N111<=2.95))])
# B_N8[17] = np.nanmean(A_N101[np.where((A_N111>2.95)&(A_N111<=3.2))])
# B_N8[18] = np.nanmean(A_N101[np.where((A_N111>3.2)&(A_N111<=3.45))])
# B_N8[19] = np.nanmean(A_N101[np.where((A_N111>3.45)&(A_N111<=3.7))])
# B_N8[20] = np.nanmean(A_N101[np.where((A_N111>3.7)&(A_N111<=3.95))])
# B_N8[21] = np.nanmean(A_N101[np.where((A_N111>3.95)&(A_N111<=4.2))])

# #2019
# B_N9 = np.zeros((22))
# B_N9[0] = np.nanmean(A_N102[np.where((A_N112>-1.3)&(A_N112<=-1.05))])
# B_N9[1] = np.nanmean(A_N102[np.where((A_N112>-1.05)&(A_N112<=-0.8))])
# B_N9[2] = np.nanmean(A_N102[np.where((A_N112>-0.8)&(A_N112<=-0.55))])
# B_N9[3] = np.nanmean(A_N102[np.where((A_N112>-0.55)&(A_N112<=-0.3))])
# B_N9[4] = np.nanmean(A_N102[np.where((A_N112>-0.3)&(A_N112<=-0.05))])
# B_N9[5] = np.nanmean(A_N102[np.where((A_N112>-0.05)&(A_N112<=0.2))])
# B_N9[6] = np.nanmean(A_N102[np.where((A_N112>0.2)&(A_N112<=0.45))])
# B_N9[7] = np.nanmean(A_N102[np.where((A_N112>0.45)&(A_N112<=0.7))])
# B_N9[8] = np.nanmean(A_N102[np.where((A_N112>0.7)&(A_N112<=0.95))])
# B_N9[9] = np.nanmean(A_N102[np.where((A_N112>0.95)&(A_N112<=1.2))])
# B_N9[10] = np.nanmean(A_N102[np.where((A_N112>1.2)&(A_N112<=1.45))])
# B_N9[11] = np.nanmean(A_N102[np.where((A_N112>1.45)&(A_N112<=1.7))])
# B_N9[12] = np.nanmean(A_N102[np.where((A_N112>1.7)&(A_N112<=1.95))])
# B_N9[13] = np.nanmean(A_N102[np.where((A_N112>1.95)&(A_N112<=2.2))])
# B_N9[14] = np.nanmean(A_N102[np.where((A_N112>2.2)&(A_N112<=2.45))])
# B_N9[15] = np.nanmean(A_N102[np.where((A_N112>2.45)&(A_N112<=2.7))])
# B_N9[16] = np.nanmean(A_N102[np.where((A_N112>2.7)&(A_N112<=2.95))])
# B_N9[17] = np.nanmean(A_N102[np.where((A_N112>2.95)&(A_N112<=3.2))])
# B_N9[18] = np.nanmean(A_N102[np.where((A_N112>3.2)&(A_N112<=3.45))])
# B_N9[19] = np.nanmean(A_N102[np.where((A_N112>3.45)&(A_N112<=3.7))])
# B_N9[20] = np.nanmean(A_N102[np.where((A_N112>3.7)&(A_N112<=3.95))])
# B_N9[21] = np.nanmean(A_N102[np.where((A_N112>3.95)&(A_N112<=4.2))])

# #2020
# B_N10 = np.zeros((22))
# B_N10[0] = np.nanmean(A_N103[np.where((A_N113>-1.3)&(A_N113<=-1.05))])
# B_N10[1] = np.nanmean(A_N103[np.where((A_N113>-1.05)&(A_N113<=-0.8))])
# B_N10[2] = np.nanmean(A_N103[np.where((A_N113>-0.8)&(A_N113<=-0.55))])
# B_N10[3] = np.nanmean(A_N103[np.where((A_N113>-0.55)&(A_N113<=-0.3))])
# B_N10[4] = np.nanmean(A_N103[np.where((A_N113>-0.3)&(A_N113<=-0.05))])
# B_N10[5] = np.nanmean(A_N103[np.where((A_N113>-0.05)&(A_N113<=0.2))])
# B_N10[6] = np.nanmean(A_N103[np.where((A_N113>0.2)&(A_N113<=0.45))])
# B_N10[7] = np.nanmean(A_N103[np.where((A_N113>0.45)&(A_N113<=0.7))])
# B_N10[8] = np.nanmean(A_N103[np.where((A_N113>0.7)&(A_N113<=0.95))])
# B_N10[9] = np.nanmean(A_N103[np.where((A_N113>0.95)&(A_N113<=1.2))])
# B_N10[10] = np.nanmean(A_N103[np.where((A_N113>1.2)&(A_N113<=1.45))])
# B_N10[11] = np.nanmean(A_N103[np.where((A_N113>1.45)&(A_N113<=1.7))])
# B_N10[12] = np.nanmean(A_N103[np.where((A_N113>1.7)&(A_N113<=1.95))])
# B_N10[13] = np.nanmean(A_N103[np.where((A_N113>1.95)&(A_N113<=2.2))])
# B_N10[14] = np.nanmean(A_N103[np.where((A_N113>2.2)&(A_N113<=2.45))])
# B_N10[15] = np.nanmean(A_N103[np.where((A_N113>2.45)&(A_N113<=2.7))])
# B_N10[16] = np.nanmean(A_N103[np.where((A_N113>2.7)&(A_N113<=2.95))])
# B_N10[17] = np.nanmean(A_N103[np.where((A_N113>2.95)&(A_N113<=3.2))])
# B_N10[18] = np.nanmean(A_N103[np.where((A_N113>3.2)&(A_N113<=3.45))])
# B_N10[19] = np.nanmean(A_N103[np.where((A_N113>3.45)&(A_N113<=3.7))])
# B_N10[20] = np.nanmean(A_N103[np.where((A_N113>3.7)&(A_N113<=3.95))])
# B_N10[21] = np.nanmean(A_N103[np.where((A_N113>3.95)&(A_N113<=4.2))])


# PCA = np.arange(-1.3,4.2,0.25)
# # plt.plot(PCA, B_N,label='2010')
# # plt.plot(PCA, B_N0,color='blue', label='2011-2019')
# plt.plot(PCA, B_N1,label='2011',alpha = 0.5)
# plt.plot(PCA, B_N2,label='2012',alpha = 0.5)
# plt.plot(PCA, B_N3,label='2013',alpha = 0.5)
# plt.plot(PCA, B_N4,label='2014',alpha = 0.5)
# plt.plot(PCA, B_N5,label='2015',alpha = 0.5)
# plt.plot(PCA, B_N6,label='2016',alpha = 0.5)
# plt.plot(PCA, B_N7,label='2017',alpha = 0.5)
# plt.plot(PCA, B_N8,color='black',label='2018',alpha = 0.5)
# plt.plot(PCA, B_N9,color='green', label='2019',alpha = 0.5)
# plt.plot(PCA, B_N10,color='red',label='2020',linewidth = 3,ls = '-.')

# plt.legend()
# plt.title("EOF-CERES_Cldarea", fontsize=18)
# plt.xlabel("EOF", fontsize=14)
# plt.ylabel("CERES_Cldarea(%)", fontsize=14)

# sns.displot(A_N12,kind="kde")

# plt.subplot(111)
# hist, bin_edges = np.histogram(A_N12)
# plt.plot(hist)
