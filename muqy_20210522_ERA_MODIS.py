# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:37:08 2021

@author: Mu o(*￣▽￣*)ブ
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from scipy import stats
from scipy.stats import zscore
from sklearn.decomposition import PCA
import glob
import pandas as pd
import os


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

for i in range(0, 10):

    # year_str=str(2010).zfill(4)
    # month_str=str(1).zfill(2)
    # time_str0=year_str+month_str

    ERA_file = glob.glob(
        "G:\\ERA5_daily_stored per month_china_2\\ERA5_daily_"
        + "*.nc"
    )
    FILE_NAME_ERA = ERA_file[i]

    for i in range(0, 28):

        file_obj = xr.open_dataset(FILE_NAME_ERA)
        lat = file_obj.lat
        lon = file_obj.lon
        P = file_obj.level
        z = file_obj.Geo
        RH = file_obj.RH
        T = file_obj.T
        W = file_obj.W
        T = T[:, 7, :, :]
        RH = RH[:, 7, :, :]
        W = W[:, 7, :, :]

        # T[i,:,:] = np.flipud(T[i,:,:])
        # RH[i,:,:] = np.flipud(RH[i,:,:])
        # W[i,:,:] = np.flipud(W[i,:,:])

        # RH = RH[:,89:170,39:180]
        # T = T[:,89:170,39:180]
        # W = W[:,89:170,39:180]

        RH_N = stats.zscore(RH, axis=0)
        T_N = stats.zscore(T, axis=0)
        W_N = stats.zscore(W, axis=0)

        RH_N1 = RH_N[i, :, :].reshape(11421)
        T_N1 = T_N[i, :, :].reshape(11421)
        W_N1 = W_N[i, :, :].reshape(11421)
        A_N10 = np.concatenate((A_N10, RH_N1), axis=0)
        A_N11 = np.concatenate((A_N11, T_N1), axis=0)
        A_N12 = np.concatenate((A_N12, W_N1), axis=0)

A_N10 = np.delete(A_N10, 0, axis=0)
A_N11 = np.delete(A_N11, 0, axis=0)
A_N12 = np.delete(A_N12, 0, axis=0)

A_N1 = np.zeros((3197880, 3))
A_N1[:, 0] = A_N10
A_N1[:, 1] = A_N11
A_N1[:, 2] = A_N12

pca = PCA(n_components=1, whiten=True)
pca.fit(A_N1)
A_N1 = pca.fit_transform(A_N1)
# A_N1 = A_N1.reshape(11,28,81,141)

# A_N1 = A_N1[1:11,:,:,:]
# A_N1 = np.mean(A_N1[:,:,:,:], axis=0)
A_N1 = A_N1.reshape(3197880)
A_N1 = A_N1.reshape(10, 28, 81, 141)

A_N14 = A_N1[0, :, :, :]  # 2011
A_N15 = A_N1[1, :, :, :]  # 2012
A_N16 = A_N1[2, :, :, :]  # 2013
A_N17 = A_N1[3, :, :, :]  # 2014
A_N18 = A_N1[4, :, :, :]  # 2015
A_N19 = A_N1[5, :, :, :]  # 2016
A_N110 = A_N1[6, :, :, :]  # 2017
A_N111 = A_N1[7, :, :, :]  # 2018
A_N112 = A_N1[8, :, :, :]  # 2019
A_N113 = A_N1[9, :, :, :]  # 2020

A_N1 = A_N1.reshape(3197880)

A_N14 = A_N14.reshape(319788)
A_N15 = A_N15.reshape(319788)
A_N16 = A_N16.reshape(319788)
A_N17 = A_N17.reshape(319788)
A_N18 = A_N18.reshape(319788)
A_N19 = A_N19.reshape(319788)
A_N110 = A_N110.reshape(319788)
A_N111 = A_N111.reshape(319788)
A_N112 = A_N112.reshape(319788)
A_N113 = A_N113.reshape(319788)

################################# rename ERA5 #######################################

# CERES_file=glob.glob('G:\\CERES_highcloud\\02-05\\CERES_highcloud_' +'*.nc')
# path = 'G:'
# n = 2013

# for i in range(0,10):
#     oldname = CERES_file[i]
#     newname = path+ os.sep +'CERES_highcloud_'+str(n)+'0201-'+str(n)+'0531.nc'
#     n = n+1
#     os.rename(oldname,newname)

############################### Deal CERES data ###################################

A_N20 = np.zeros((1))
A_N21 = np.zeros((1))
A_N22 = np.zeros((1))
A_N23 = np.zeros((1))
A_N24 = np.zeros((1))
A_N25 = np.zeros((1))
A_N26 = np.zeros((1))
A_N27 = np.zeros((1))

path = "F://MOD08/"
files = os.listdir(path)
files.sort(key=lambda x: int(x[:-6]))
# MODIS_file=glob.glob(path +files[0]+'/MOD08_D3'+'*.hdf')

for i in range(1, 11):

    MODIS_file = glob.glob(
        path + files[i] + "/MOD08_D3" + "*.hdf"
    )

    for j in range(0, 28):

        FILE_NAME_MODIS = MODIS_file[j]

        data = nc.Dataset(FILE_NAME_MODIS)
        # lat = file_obj.lat
        # lon = file_obj.lon
        # t = file_obj.time
        cirrus = data.variables["Cirrus_Fraction_Infrared"]
        high = data.variables["High_Cloud_Fraction_Infrared"]

        cirrus = np.flipud(np.array(cirrus))
        c_1, c_2 = np.split(cirrus, 2, axis=1)
        cirrus = np.c_[c_2, c_1]
        cirrus = cirrus[89:170, 39:180]
        cirrus1 = cirrus[:, :].reshape(11421)
        cirrus1[cirrus1 < 0] = 0

        high = np.flipud(np.array(high))
        c_1, c_2 = np.split(high, 2, axis=1)
        high = np.c_[c_2, c_1]
        high = high[89:170, 39:180]
        high1 = high[:, :].reshape(11421)
        high1[high1 < 0] = 0

        A_N20 = np.concatenate((A_N20, cirrus1), axis=0)
        A_N21 = np.concatenate((A_N21, high1), axis=0)


A_N20 = np.delete(A_N20, 0, axis=0)  # Cirrus_Fraction
A_N21 = np.delete(A_N21, 0, axis=0)  # High_Cloud_Fraction
# A_N22 = np.delete(A_N22, 0, axis=0) #cldtau
# A_N23 = np.delete(A_N23, 0, axis=0) #iwp
# A_N24 = np.delete(A_N24, 0, axis=0) #cldpress
# A_N25 = np.delete(A_N25, 0, axis=0) #cldphase
# A_N26 = np.delete(A_N26, 0, axis=0) #cldemissir

A_N20 = A_N21.reshape(
    10, 28, 81, 141
)  # Choose the variable used in the plot
A_NM = A_N20[0:10, :, :, :]
# A_N30 = A_N20[0,:,:,:] #2010
A_N40 = A_N20[0, :, :, :]  # 2011
A_N50 = A_N20[1, :, :, :]  # 2012
A_N60 = A_N20[2, :, :, :]  # 2013
A_N70 = A_N20[3, :, :, :]  # 2014
A_N80 = A_N20[4, :, :, :]  # 2015
A_N90 = A_N20[5, :, :, :]  # 2016
A_N100 = A_N20[6, :, :, :]  # 2017
A_N101 = A_N20[7, :, :, :]  # 2018
A_N102 = A_N20[8, :, :, :]  # 2019
A_N103 = A_N20[9, :, :, :]  # 2020

A_N20 = A_N20.reshape(3197880)
A_NM = A_NM.reshape(3197880)
# A_N30 = A_N30.reshape(319788)
A_N40 = A_N40.reshape(319788)
A_N50 = A_N50.reshape(319788)
A_N60 = A_N60.reshape(319788)
A_N70 = A_N70.reshape(319788)
A_N80 = A_N80.reshape(319788)
A_N90 = A_N90.reshape(319788)
A_N100 = A_N100.reshape(319788)
A_N101 = A_N101.reshape(319788)
A_N102 = A_N102.reshape(319788)
A_N103 = A_N103.reshape(319788)


# A_N20 = np.nanmean(A_N20.reshape(10,319788), axis=0)
# A_N21 = np.nanmean(A_N21.reshape(10,319788), axis=0)
# A_N22 = np.nanmean(A_N22.reshape(10,319788), axis=0)
# A_N23 = np.nanmean(A_N23.reshape(10,319788), axis=0)

# A_N21 = A_N21.reshape(28,81,141)
# plt.plot(A_N1, A_N20)
#########################################################################################
# A_NNN = np.zeros((81,141))
# A_NNN = np.random.rand(81,141)
# A_NNN1 = np.random.rand(81,141)
# A_N103 = A_N103.reshape(28,81,141)
# A_N113 = A_N113.reshape(28,81,141)
# A_N102 = A_N102.reshape(28,81,141)
# A_N112 = A_N112.reshape(28,81,141)
# A_nnm = np.nanmean(A_N103,axis = 0)

# for i in range(0,81):
#     for j in range(0,141):
#         if A_N103[:,i,j][np.where((A_N113[:,i,j]>=0.5))].shape == 0:
#             A_NNN[i,j] = np.nan
#         else :
#             A_NNN[i,j] = np.nanmean(A_N103[:,i,j][np.where((A_N113[:,i,j]>=0.5))])

# for i in range(0,81):
#     for j in range(0,141):
#         if A_N102[:,i,j][np.where((A_N112[:,i,j]>=0.5))].shape == 0:
#             A_NNN1[i,j] = np.nan
#         else :
#             A_NNN1[i,j] = np.nanmean(A_N102[:,i,j][np.where((A_N112[:,i,j]>=0.5))])

# DDD = A_NNN1-A_NNN
############################## plot PCA-cldarea #########################################

B_N0 = np.zeros((28))
B_N0[0] = np.nanmean(
    A_NM[np.where((A_N1 > -1.55) & (A_N1 <= -1.45))]
)
B_N0[1] = np.nanmean(
    A_NM[np.where((A_N1 > -1.45) & (A_N1 <= -1.35))]
)
B_N0[2] = np.nanmean(
    A_NM[np.where((A_N1 > -1.35) & (A_N1 <= -1.25))]
)
B_N0[3] = np.nanmean(
    A_NM[np.where((A_N1 > -1.25) & (A_N1 <= -1.15))]
)
B_N0[4] = np.nanmean(
    A_NM[np.where((A_N1 > -1.15) & (A_N1 <= -1.05))]
)
B_N0[5] = np.nanmean(
    A_NM[np.where((A_N1 > -1.05) & (A_N1 <= -0.95))]
)
B_N0[6] = np.nanmean(
    A_NM[np.where((A_N1 > -0.95) & (A_N1 <= -0.85))]
)
B_N0[7] = np.nanmean(
    A_NM[np.where((A_N1 > -0.85) & (A_N1 <= -0.75))]
)
B_N0[8] = np.nanmean(
    A_NM[np.where((A_N1 > -0.75) & (A_N1 <= -0.65))]
)
B_N0[9] = np.nanmean(
    A_NM[np.where((A_N1 > -0.65) & (A_N1 <= -0.55))]
)
B_N0[10] = np.nanmean(
    A_NM[np.where((A_N1 > -0.55) & (A_N1 <= -0.45))]
)
B_N0[11] = np.nanmean(
    A_NM[np.where((A_N1 > -0.45) & (A_N1 <= -0.35))]
)
B_N0[12] = np.nanmean(
    A_NM[np.where((A_N1 > -0.35) & (A_N1 <= -0.25))]
)
B_N0[13] = np.nanmean(
    A_NM[np.where((A_N1 > -0.25) & (A_N1 <= -0.15))]
)
B_N0[14] = np.nanmean(
    A_NM[np.where((A_N1 > -0.15) & (A_N1 <= -0.05))]
)
B_N0[15] = np.nanmean(
    A_NM[np.where((A_N1 > -0.05) & (A_N1 <= 0.05))]
)
B_N0[16] = np.nanmean(
    A_NM[np.where((A_N1 > 0.05) & (A_N1 <= 0.15))]
)
B_N0[17] = np.nanmean(
    A_NM[np.where((A_N1 > 0.15) & (A_N1 <= 0.25))]
)
B_N0[18] = np.nanmean(
    A_NM[np.where((A_N1 > 0.25) & (A_N1 <= 0.35))]
)
B_N0[19] = np.nanmean(
    A_NM[np.where((A_N1 > 0.35) & (A_N1 <= 0.45))]
)
B_N0[20] = np.nanmean(
    A_NM[np.where((A_N1 > 0.45) & (A_N1 <= 0.55))]
)
B_N0[21] = np.nanmean(
    A_NM[np.where((A_N1 > 0.55) & (A_N1 <= 0.65))]
)
B_N0[22] = np.nanmean(
    A_NM[np.where((A_N1 > 0.65) & (A_N1 <= 0.75))]
)
B_N0[23] = np.nanmean(
    A_NM[np.where((A_N1 > 0.75) & (A_N1 <= 0.85))]
)
B_N0[24] = np.nanmean(
    A_NM[np.where((A_N1 > 0.85) & (A_N1 <= 0.95))]
)
B_N0[25] = np.nanmean(
    A_NM[np.where((A_N1 > 0.95) & (A_N1 <= 1.05))]
)
B_N0[26] = np.nanmean(
    A_NM[np.where((A_N1 > 1.05) & (A_N1 <= 1.15))]
)
B_N0[27] = np.nanmean(
    A_NM[np.where((A_N1 > 1.15) & (A_N1 <= 1.25))]
)

# 2010
# B_N = np.zeros((28))
# B_N[0] = np.nanmean(A_N30[np.where((A_N13>-1.55)&(A_N13<=-1.45))])
# B_N[1] = np.nanmean(A_N30[np.where((A_N13>-1.45)&(A_N13<=-1.35))])
# B_N[2] = np.nanmean(A_N30[np.where((A_N13>-1.35)&(A_N13<=-1.25))])
# B_N[3] = np.nanmean(A_N30[np.where((A_N13>-1.25)&(A_N13<=-1.15))])
# B_N[4] = np.nanmean(A_N30[np.where((A_N13>-1.15)&(A_N13<=-1.05))])
# B_N[5] = np.nanmean(A_N30[np.where((A_N13>-1.05)&(A_N13<=-0.95))])
# B_N[6] = np.nanmean(A_N30[np.where((A_N13>-0.95)&(A_N13<=-0.85))])
# B_N[7] = np.nanmean(A_N30[np.where((A_N13>-0.85)&(A_N13<=-0.75))])
# B_N[8] = np.nanmean(A_N30[np.where((A_N13>-0.75)&(A_N13<=-0.65))])
# B_N[9] = np.nanmean(A_N30[np.where((A_N13>-0.65)&(A_N13<=-0.55))])
# B_N[10] = np.nanmean(A_N30[np.where((A_N13>-0.55)&(A_N13<=-0.45))])
# B_N[11] = np.nanmean(A_N30[np.where((A_N13>-0.45)&(A_N13<=-0.35))])
# B_N[12] = np.nanmean(A_N30[np.where((A_N13>-0.35)&(A_N13<=-0.25))])
# B_N[13] = np.nanmean(A_N30[np.where((A_N13>-0.25)&(A_N13<=-0.15))])
# B_N[14] = np.nanmean(A_N30[np.where((A_N13>-0.15)&(A_N13<=-0.05))])
# B_N[15] = np.nanmean(A_N30[np.where((A_N13>-0.05)&(A_N13<=0.05))])
# B_N[16] = np.nanmean(A_N30[np.where((A_N13>0.05)&(A_N13<=0.15))])
# B_N[17] = np.nanmean(A_N30[np.where((A_N13>0.15)&(A_N13<=0.25))])
# B_N[18] = np.nanmean(A_N30[np.where((A_N13>0.25)&(A_N13<=0.35))])
# B_N[19] = np.nanmean(A_N30[np.where((A_N13>0.35)&(A_N13<=0.45))])
# B_N[20] = np.nanmean(A_N30[np.where((A_N13>0.45)&(A_N13<=0.55))])
# B_N[21] = np.nanmean(A_N30[np.where((A_N13>0.55)&(A_N13<=0.65))])
# B_N[22] = np.nanmean(A_N30[np.where((A_N13>0.65)&(A_N13<=0.75))])
# B_N[23] = np.nanmean(A_N30[np.where((A_N13>0.75)&(A_N13<=0.85))])
# B_N[24] = np.nanmean(A_N30[np.where((A_N13>0.85)&(A_N13<=0.95))])
# B_N[25] = np.nanmean(A_N30[np.where((A_N13>0.95)&(A_N13<=1.05))])
# B_N[26] = np.nanmean(A_N30[np.where((A_N13>1.05)&(A_N13<=1.15))])
# B_N[27] = np.nanmean(A_N30[np.where((A_N13>1.15)&(A_N13<=1.25))])

# 2011
B_N1 = np.zeros((28))
B_N1[0] = np.nanmean(
    A_N40[np.where((A_N14 > -1.55) & (A_N14 <= -1.45))]
)
B_N1[1] = np.nanmean(
    A_N40[np.where((A_N14 > -1.45) & (A_N14 <= -1.35))]
)
B_N1[2] = np.nanmean(
    A_N40[np.where((A_N14 > -1.35) & (A_N14 <= -1.25))]
)
B_N1[3] = np.nanmean(
    A_N40[np.where((A_N14 > -1.25) & (A_N14 <= -1.15))]
)
B_N1[4] = np.nanmean(
    A_N40[np.where((A_N14 > -1.15) & (A_N14 <= -1.05))]
)
B_N1[5] = np.nanmean(
    A_N40[np.where((A_N14 > -1.05) & (A_N14 <= -0.95))]
)
B_N1[6] = np.nanmean(
    A_N40[np.where((A_N14 > -0.95) & (A_N14 <= -0.85))]
)
B_N1[7] = np.nanmean(
    A_N40[np.where((A_N14 > -0.85) & (A_N14 <= -0.75))]
)
B_N1[8] = np.nanmean(
    A_N40[np.where((A_N14 > -0.75) & (A_N14 <= -0.65))]
)
B_N1[9] = np.nanmean(
    A_N40[np.where((A_N14 > -0.65) & (A_N14 <= -0.55))]
)
B_N1[10] = np.nanmean(
    A_N40[np.where((A_N14 > -0.55) & (A_N14 <= -0.45))]
)
B_N1[11] = np.nanmean(
    A_N40[np.where((A_N14 > -0.45) & (A_N14 <= -0.35))]
)
B_N1[12] = np.nanmean(
    A_N40[np.where((A_N14 > -0.35) & (A_N14 <= -0.25))]
)
B_N1[13] = np.nanmean(
    A_N40[np.where((A_N14 > -0.25) & (A_N14 <= -0.15))]
)
B_N1[14] = np.nanmean(
    A_N40[np.where((A_N14 > -0.15) & (A_N14 <= -0.05))]
)
B_N1[15] = np.nanmean(
    A_N40[np.where((A_N14 > -0.05) & (A_N14 <= 0.05))]
)
B_N1[16] = np.nanmean(
    A_N40[np.where((A_N14 > 0.05) & (A_N14 <= 0.15))]
)
B_N1[17] = np.nanmean(
    A_N40[np.where((A_N14 > 0.15) & (A_N14 <= 0.25))]
)
B_N1[18] = np.nanmean(
    A_N40[np.where((A_N14 > 0.25) & (A_N14 <= 0.35))]
)
B_N1[19] = np.nanmean(
    A_N40[np.where((A_N14 > 0.35) & (A_N14 <= 0.45))]
)
B_N1[20] = np.nanmean(
    A_N40[np.where((A_N14 > 0.45) & (A_N14 <= 0.55))]
)
B_N1[21] = np.nanmean(
    A_N40[np.where((A_N14 > 0.55) & (A_N14 <= 0.65))]
)
B_N1[22] = np.nanmean(
    A_N40[np.where((A_N14 > 0.65) & (A_N14 <= 0.75))]
)
B_N1[23] = np.nanmean(
    A_N40[np.where((A_N14 > 0.75) & (A_N14 <= 0.85))]
)
B_N1[24] = np.nanmean(
    A_N40[np.where((A_N14 > 0.85) & (A_N14 <= 0.95))]
)
B_N1[25] = np.nanmean(
    A_N40[np.where((A_N14 > 0.95) & (A_N14 <= 1.05))]
)
B_N1[26] = np.nanmean(
    A_N40[np.where((A_N14 > 1.05) & (A_N14 <= 1.15))]
)
B_N1[27] = np.nanmean(
    A_N40[np.where((A_N14 > 1.15) & (A_N14 <= 1.25))]
)

# 2012
B_N2 = np.zeros((28))
B_N2[0] = np.nanmean(
    A_N50[np.where((A_N15 > -1.55) & (A_N15 <= -1.45))]
)
B_N2[1] = np.nanmean(
    A_N50[np.where((A_N15 > -1.45) & (A_N15 <= -1.35))]
)
B_N2[2] = np.nanmean(
    A_N50[np.where((A_N15 > -1.35) & (A_N15 <= -1.25))]
)
B_N2[3] = np.nanmean(
    A_N50[np.where((A_N15 > -1.25) & (A_N15 <= -1.15))]
)
B_N2[4] = np.nanmean(
    A_N50[np.where((A_N15 > -1.15) & (A_N15 <= -1.05))]
)
B_N2[5] = np.nanmean(
    A_N50[np.where((A_N15 > -1.05) & (A_N15 <= -0.95))]
)
B_N2[6] = np.nanmean(
    A_N50[np.where((A_N15 > -0.95) & (A_N15 <= -0.85))]
)
B_N2[7] = np.nanmean(
    A_N50[np.where((A_N15 > -0.85) & (A_N15 <= -0.75))]
)
B_N2[8] = np.nanmean(
    A_N50[np.where((A_N15 > -0.75) & (A_N15 <= -0.65))]
)
B_N2[9] = np.nanmean(
    A_N50[np.where((A_N15 > -0.65) & (A_N15 <= -0.55))]
)
B_N2[10] = np.nanmean(
    A_N50[np.where((A_N15 > -0.55) & (A_N15 <= -0.45))]
)
B_N2[11] = np.nanmean(
    A_N50[np.where((A_N15 > -0.45) & (A_N15 <= -0.35))]
)
B_N2[12] = np.nanmean(
    A_N50[np.where((A_N15 > -0.35) & (A_N15 <= -0.25))]
)
B_N2[13] = np.nanmean(
    A_N50[np.where((A_N15 > -0.25) & (A_N15 <= -0.15))]
)
B_N2[14] = np.nanmean(
    A_N50[np.where((A_N15 > -0.15) & (A_N15 <= -0.05))]
)
B_N2[15] = np.nanmean(
    A_N50[np.where((A_N15 > -0.05) & (A_N15 <= 0.05))]
)
B_N2[16] = np.nanmean(
    A_N50[np.where((A_N15 > 0.05) & (A_N15 <= 0.15))]
)
B_N2[17] = np.nanmean(
    A_N50[np.where((A_N15 > 0.15) & (A_N15 <= 0.25))]
)
B_N2[18] = np.nanmean(
    A_N50[np.where((A_N15 > 0.25) & (A_N15 <= 0.35))]
)
B_N2[19] = np.nanmean(
    A_N50[np.where((A_N15 > 0.35) & (A_N15 <= 0.45))]
)
B_N2[20] = np.nanmean(
    A_N50[np.where((A_N15 > 0.45) & (A_N15 <= 0.55))]
)
B_N2[21] = np.nanmean(
    A_N50[np.where((A_N15 > 0.55) & (A_N15 <= 0.65))]
)
B_N2[22] = np.nanmean(
    A_N50[np.where((A_N15 > 0.65) & (A_N15 <= 0.75))]
)
B_N2[23] = np.nanmean(
    A_N50[np.where((A_N15 > 0.75) & (A_N15 <= 0.85))]
)
B_N2[24] = np.nanmean(
    A_N50[np.where((A_N15 > 0.85) & (A_N15 <= 0.95))]
)
B_N2[25] = np.nanmean(
    A_N50[np.where((A_N15 > 0.95) & (A_N15 <= 1.05))]
)
B_N2[26] = np.nanmean(
    A_N50[np.where((A_N15 > 1.05) & (A_N15 <= 1.15))]
)
B_N2[27] = np.nanmean(
    A_N50[np.where((A_N15 > 1.15) & (A_N15 <= 1.25))]
)

# 2013
B_N3 = np.zeros((28))
B_N3[0] = np.nanmean(
    A_N60[np.where((A_N16 > -1.55) & (A_N16 <= -1.45))]
)
B_N3[1] = np.nanmean(
    A_N60[np.where((A_N16 > -1.45) & (A_N16 <= -1.35))]
)
B_N3[2] = np.nanmean(
    A_N60[np.where((A_N16 > -1.35) & (A_N16 <= -1.25))]
)
B_N3[3] = np.nanmean(
    A_N60[np.where((A_N16 > -1.25) & (A_N16 <= -1.15))]
)
B_N3[4] = np.nanmean(
    A_N60[np.where((A_N16 > -1.15) & (A_N16 <= -1.05))]
)
B_N3[5] = np.nanmean(
    A_N60[np.where((A_N16 > -1.05) & (A_N16 <= -0.95))]
)
B_N3[6] = np.nanmean(
    A_N60[np.where((A_N16 > -0.95) & (A_N16 <= -0.85))]
)
B_N3[7] = np.nanmean(
    A_N60[np.where((A_N16 > -0.85) & (A_N16 <= -0.75))]
)
B_N3[8] = np.nanmean(
    A_N60[np.where((A_N16 > -0.75) & (A_N16 <= -0.65))]
)
B_N3[9] = np.nanmean(
    A_N60[np.where((A_N16 > -0.65) & (A_N16 <= -0.55))]
)
B_N3[10] = np.nanmean(
    A_N60[np.where((A_N16 > -0.55) & (A_N16 <= -0.45))]
)
B_N3[11] = np.nanmean(
    A_N60[np.where((A_N16 > -0.45) & (A_N16 <= -0.35))]
)
B_N3[12] = np.nanmean(
    A_N60[np.where((A_N16 > -0.35) & (A_N16 <= -0.25))]
)
B_N3[13] = np.nanmean(
    A_N60[np.where((A_N16 > -0.25) & (A_N16 <= -0.15))]
)
B_N3[14] = np.nanmean(
    A_N60[np.where((A_N16 > -0.15) & (A_N16 <= -0.05))]
)
B_N3[15] = np.nanmean(
    A_N60[np.where((A_N16 > -0.05) & (A_N16 <= 0.05))]
)
B_N3[16] = np.nanmean(
    A_N60[np.where((A_N16 > 0.05) & (A_N16 <= 0.15))]
)
B_N3[17] = np.nanmean(
    A_N60[np.where((A_N16 > 0.15) & (A_N16 <= 0.25))]
)
B_N3[18] = np.nanmean(
    A_N60[np.where((A_N16 > 0.25) & (A_N16 <= 0.35))]
)
B_N3[19] = np.nanmean(
    A_N60[np.where((A_N16 > 0.35) & (A_N16 <= 0.45))]
)
B_N3[20] = np.nanmean(
    A_N60[np.where((A_N16 > 0.45) & (A_N16 <= 0.55))]
)
B_N3[21] = np.nanmean(
    A_N60[np.where((A_N16 > 0.55) & (A_N16 <= 0.65))]
)
B_N3[22] = np.nanmean(
    A_N60[np.where((A_N16 > 0.65) & (A_N16 <= 0.75))]
)
B_N3[23] = np.nanmean(
    A_N60[np.where((A_N16 > 0.75) & (A_N16 <= 0.85))]
)
B_N3[24] = np.nanmean(
    A_N60[np.where((A_N16 > 0.85) & (A_N16 <= 0.95))]
)
B_N3[25] = np.nanmean(
    A_N60[np.where((A_N16 > 0.95) & (A_N16 <= 1.05))]
)
B_N3[26] = np.nanmean(
    A_N60[np.where((A_N16 > 1.05) & (A_N16 <= 1.15))]
)
B_N3[27] = np.nanmean(
    A_N60[np.where((A_N16 > 1.15) & (A_N16 <= 1.25))]
)

# 2014
B_N4 = np.zeros((28))
B_N4[0] = np.nanmean(
    A_N70[np.where((A_N17 > -1.55) & (A_N17 <= -1.45))]
)
B_N4[1] = np.nanmean(
    A_N70[np.where((A_N17 > -1.45) & (A_N17 <= -1.35))]
)
B_N4[2] = np.nanmean(
    A_N70[np.where((A_N17 > -1.35) & (A_N17 <= -1.25))]
)
B_N4[3] = np.nanmean(
    A_N70[np.where((A_N17 > -1.25) & (A_N17 <= -1.15))]
)
B_N4[4] = np.nanmean(
    A_N70[np.where((A_N17 > -1.15) & (A_N17 <= -1.05))]
)
B_N4[5] = np.nanmean(
    A_N70[np.where((A_N17 > -1.05) & (A_N17 <= -0.95))]
)
B_N4[6] = np.nanmean(
    A_N70[np.where((A_N17 > -0.95) & (A_N17 <= -0.85))]
)
B_N4[7] = np.nanmean(
    A_N70[np.where((A_N17 > -0.85) & (A_N17 <= -0.75))]
)
B_N4[8] = np.nanmean(
    A_N70[np.where((A_N17 > -0.75) & (A_N17 <= -0.65))]
)
B_N4[9] = np.nanmean(
    A_N70[np.where((A_N17 > -0.65) & (A_N17 <= -0.55))]
)
B_N4[10] = np.nanmean(
    A_N70[np.where((A_N17 > -0.55) & (A_N17 <= -0.45))]
)
B_N4[11] = np.nanmean(
    A_N70[np.where((A_N17 > -0.45) & (A_N17 <= -0.35))]
)
B_N4[12] = np.nanmean(
    A_N70[np.where((A_N17 > -0.35) & (A_N17 <= -0.25))]
)
B_N4[13] = np.nanmean(
    A_N70[np.where((A_N17 > -0.25) & (A_N17 <= -0.15))]
)
B_N4[14] = np.nanmean(
    A_N70[np.where((A_N17 > -0.15) & (A_N17 <= -0.05))]
)
B_N4[15] = np.nanmean(
    A_N70[np.where((A_N17 > -0.05) & (A_N17 <= 0.05))]
)
B_N4[16] = np.nanmean(
    A_N70[np.where((A_N17 > 0.05) & (A_N17 <= 0.15))]
)
B_N4[17] = np.nanmean(
    A_N70[np.where((A_N17 > 0.15) & (A_N17 <= 0.25))]
)
B_N4[18] = np.nanmean(
    A_N70[np.where((A_N17 > 0.25) & (A_N17 <= 0.35))]
)
B_N4[19] = np.nanmean(
    A_N70[np.where((A_N17 > 0.35) & (A_N17 <= 0.45))]
)
B_N4[20] = np.nanmean(
    A_N70[np.where((A_N17 > 0.45) & (A_N17 <= 0.55))]
)
B_N4[21] = np.nanmean(
    A_N70[np.where((A_N17 > 0.55) & (A_N17 <= 0.65))]
)
B_N4[22] = np.nanmean(
    A_N70[np.where((A_N17 > 0.65) & (A_N17 <= 0.75))]
)
B_N4[23] = np.nanmean(
    A_N70[np.where((A_N17 > 0.75) & (A_N17 <= 0.85))]
)
B_N4[24] = np.nanmean(
    A_N70[np.where((A_N17 > 0.85) & (A_N17 <= 0.95))]
)
B_N4[25] = np.nanmean(
    A_N70[np.where((A_N17 > 0.95) & (A_N17 <= 1.05))]
)
B_N4[26] = np.nanmean(
    A_N70[np.where((A_N17 > 1.05) & (A_N17 <= 1.15))]
)
B_N4[27] = np.nanmean(
    A_N70[np.where((A_N17 > 1.15) & (A_N17 <= 1.25))]
)

# 2015
B_N5 = np.zeros((28))
B_N5[0] = np.nanmean(
    A_N80[np.where((A_N18 > -1.55) & (A_N18 <= -1.45))]
)
B_N5[1] = np.nanmean(
    A_N80[np.where((A_N18 > -1.45) & (A_N18 <= -1.35))]
)
B_N5[2] = np.nanmean(
    A_N80[np.where((A_N18 > -1.35) & (A_N18 <= -1.25))]
)
B_N5[3] = np.nanmean(
    A_N80[np.where((A_N18 > -1.25) & (A_N18 <= -1.15))]
)
B_N5[4] = np.nanmean(
    A_N80[np.where((A_N18 > -1.15) & (A_N18 <= -1.05))]
)
B_N5[5] = np.nanmean(
    A_N80[np.where((A_N18 > -1.05) & (A_N18 <= -0.95))]
)
B_N5[6] = np.nanmean(
    A_N80[np.where((A_N18 > -0.95) & (A_N18 <= -0.85))]
)
B_N5[7] = np.nanmean(
    A_N80[np.where((A_N18 > -0.85) & (A_N18 <= -0.75))]
)
B_N5[8] = np.nanmean(
    A_N80[np.where((A_N18 > -0.75) & (A_N18 <= -0.65))]
)
B_N5[9] = np.nanmean(
    A_N80[np.where((A_N18 > -0.65) & (A_N18 <= -0.55))]
)
B_N5[10] = np.nanmean(
    A_N80[np.where((A_N18 > -0.55) & (A_N18 <= -0.45))]
)
B_N5[11] = np.nanmean(
    A_N80[np.where((A_N18 > -0.45) & (A_N18 <= -0.35))]
)
B_N5[12] = np.nanmean(
    A_N80[np.where((A_N18 > -0.35) & (A_N18 <= -0.25))]
)
B_N5[13] = np.nanmean(
    A_N80[np.where((A_N18 > -0.25) & (A_N18 <= -0.15))]
)
B_N5[14] = np.nanmean(
    A_N80[np.where((A_N18 > -0.15) & (A_N18 <= -0.05))]
)
B_N5[15] = np.nanmean(
    A_N80[np.where((A_N18 > -0.05) & (A_N18 <= 0.05))]
)
B_N5[16] = np.nanmean(
    A_N80[np.where((A_N18 > 0.05) & (A_N18 <= 0.15))]
)
B_N5[17] = np.nanmean(
    A_N80[np.where((A_N18 > 0.15) & (A_N18 <= 0.25))]
)
B_N5[18] = np.nanmean(
    A_N80[np.where((A_N18 > 0.25) & (A_N18 <= 0.35))]
)
B_N5[19] = np.nanmean(
    A_N80[np.where((A_N18 > 0.35) & (A_N18 <= 0.45))]
)
B_N5[20] = np.nanmean(
    A_N80[np.where((A_N18 > 0.45) & (A_N18 <= 0.55))]
)
B_N5[21] = np.nanmean(
    A_N80[np.where((A_N18 > 0.55) & (A_N18 <= 0.65))]
)
B_N5[22] = np.nanmean(
    A_N80[np.where((A_N18 > 0.65) & (A_N18 <= 0.75))]
)
B_N5[23] = np.nanmean(
    A_N80[np.where((A_N18 > 0.75) & (A_N18 <= 0.85))]
)
B_N5[24] = np.nanmean(
    A_N80[np.where((A_N18 > 0.85) & (A_N18 <= 0.95))]
)
B_N5[25] = np.nanmean(
    A_N80[np.where((A_N18 > 0.95) & (A_N18 <= 1.05))]
)
B_N5[26] = np.nanmean(
    A_N80[np.where((A_N18 > 1.05) & (A_N18 <= 1.15))]
)
B_N5[27] = np.nanmean(
    A_N80[np.where((A_N18 > 1.15) & (A_N18 <= 1.25))]
)

# 2016
B_N6 = np.zeros((28))
B_N6[0] = np.nanmean(
    A_N90[np.where((A_N19 > -1.55) & (A_N19 <= -1.45))]
)
B_N6[1] = np.nanmean(
    A_N90[np.where((A_N19 > -1.45) & (A_N19 <= -1.35))]
)
B_N6[2] = np.nanmean(
    A_N90[np.where((A_N19 > -1.35) & (A_N19 <= -1.25))]
)
B_N6[3] = np.nanmean(
    A_N90[np.where((A_N19 > -1.25) & (A_N19 <= -1.15))]
)
B_N6[4] = np.nanmean(
    A_N90[np.where((A_N19 > -1.15) & (A_N19 <= -1.05))]
)
B_N6[5] = np.nanmean(
    A_N90[np.where((A_N19 > -1.05) & (A_N19 <= -0.95))]
)
B_N6[6] = np.nanmean(
    A_N90[np.where((A_N19 > -0.95) & (A_N19 <= -0.85))]
)
B_N6[7] = np.nanmean(
    A_N90[np.where((A_N19 > -0.85) & (A_N19 <= -0.75))]
)
B_N6[8] = np.nanmean(
    A_N90[np.where((A_N19 > -0.75) & (A_N19 <= -0.65))]
)
B_N6[9] = np.nanmean(
    A_N90[np.where((A_N19 > -0.65) & (A_N19 <= -0.55))]
)
B_N6[10] = np.nanmean(
    A_N90[np.where((A_N19 > -0.55) & (A_N19 <= -0.45))]
)
B_N6[11] = np.nanmean(
    A_N90[np.where((A_N19 > -0.45) & (A_N19 <= -0.35))]
)
B_N6[12] = np.nanmean(
    A_N90[np.where((A_N19 > -0.35) & (A_N19 <= -0.25))]
)
B_N6[13] = np.nanmean(
    A_N90[np.where((A_N19 > -0.25) & (A_N19 <= -0.15))]
)
B_N6[14] = np.nanmean(
    A_N90[np.where((A_N19 > -0.15) & (A_N19 <= -0.05))]
)
B_N6[15] = np.nanmean(
    A_N90[np.where((A_N19 > -0.05) & (A_N19 <= 0.05))]
)
B_N6[16] = np.nanmean(
    A_N90[np.where((A_N19 > 0.05) & (A_N19 <= 0.15))]
)
B_N6[17] = np.nanmean(
    A_N90[np.where((A_N19 > 0.15) & (A_N19 <= 0.25))]
)
B_N6[18] = np.nanmean(
    A_N90[np.where((A_N19 > 0.25) & (A_N19 <= 0.35))]
)
B_N6[19] = np.nanmean(
    A_N90[np.where((A_N19 > 0.35) & (A_N19 <= 0.45))]
)
B_N6[20] = np.nanmean(
    A_N90[np.where((A_N19 > 0.45) & (A_N19 <= 0.55))]
)
B_N6[21] = np.nanmean(
    A_N90[np.where((A_N19 > 0.55) & (A_N19 <= 0.65))]
)
B_N6[22] = np.nanmean(
    A_N90[np.where((A_N19 > 0.65) & (A_N19 <= 0.75))]
)
B_N6[23] = np.nanmean(
    A_N90[np.where((A_N19 > 0.75) & (A_N19 <= 0.85))]
)
B_N6[24] = np.nanmean(
    A_N90[np.where((A_N19 > 0.85) & (A_N19 <= 0.95))]
)
B_N6[25] = np.nanmean(
    A_N90[np.where((A_N19 > 0.95) & (A_N19 <= 1.05))]
)
B_N6[26] = np.nanmean(
    A_N90[np.where((A_N19 > 1.05) & (A_N19 <= 1.15))]
)
B_N6[27] = np.nanmean(
    A_N90[np.where((A_N19 > 1.15) & (A_N19 <= 1.25))]
)

# 2017
B_N7 = np.zeros((28))
B_N7[0] = np.nanmean(
    A_N100[np.where((A_N110 > -1.55) & (A_N110 <= -1.45))]
)
B_N7[1] = np.nanmean(
    A_N100[np.where((A_N110 > -1.45) & (A_N110 <= -1.35))]
)
B_N7[2] = np.nanmean(
    A_N100[np.where((A_N110 > -1.35) & (A_N110 <= -1.25))]
)
B_N7[3] = np.nanmean(
    A_N100[np.where((A_N110 > -1.25) & (A_N110 <= -1.15))]
)
B_N7[4] = np.nanmean(
    A_N100[np.where((A_N110 > -1.15) & (A_N110 <= -1.05))]
)
B_N7[5] = np.nanmean(
    A_N100[np.where((A_N110 > -1.05) & (A_N110 <= -0.95))]
)
B_N7[6] = np.nanmean(
    A_N100[np.where((A_N110 > -0.95) & (A_N110 <= -0.85))]
)
B_N7[7] = np.nanmean(
    A_N100[np.where((A_N110 > -0.85) & (A_N110 <= -0.75))]
)
B_N7[8] = np.nanmean(
    A_N100[np.where((A_N110 > -0.75) & (A_N110 <= -0.65))]
)
B_N7[9] = np.nanmean(
    A_N100[np.where((A_N110 > -0.65) & (A_N110 <= -0.55))]
)
B_N7[10] = np.nanmean(
    A_N100[np.where((A_N110 > -0.55) & (A_N110 <= -0.45))]
)
B_N7[11] = np.nanmean(
    A_N100[np.where((A_N110 > -0.45) & (A_N110 <= -0.35))]
)
B_N7[12] = np.nanmean(
    A_N100[np.where((A_N110 > -0.35) & (A_N110 <= -0.25))]
)
B_N7[13] = np.nanmean(
    A_N100[np.where((A_N110 > -0.25) & (A_N110 <= -0.15))]
)
B_N7[14] = np.nanmean(
    A_N100[np.where((A_N110 > -0.15) & (A_N110 <= -0.05))]
)
B_N7[15] = np.nanmean(
    A_N100[np.where((A_N110 > -0.05) & (A_N110 <= 0.05))]
)
B_N7[16] = np.nanmean(
    A_N100[np.where((A_N110 > 0.05) & (A_N110 <= 0.15))]
)
B_N7[17] = np.nanmean(
    A_N100[np.where((A_N110 > 0.15) & (A_N110 <= 0.25))]
)
B_N7[18] = np.nanmean(
    A_N100[np.where((A_N110 > 0.25) & (A_N110 <= 0.35))]
)
B_N7[19] = np.nanmean(
    A_N100[np.where((A_N110 > 0.35) & (A_N110 <= 0.45))]
)
B_N7[20] = np.nanmean(
    A_N100[np.where((A_N110 > 0.45) & (A_N110 <= 0.55))]
)
B_N7[21] = np.nanmean(
    A_N100[np.where((A_N110 > 0.55) & (A_N110 <= 0.65))]
)
B_N7[22] = np.nanmean(
    A_N100[np.where((A_N110 > 0.65) & (A_N110 <= 0.75))]
)
B_N7[23] = np.nanmean(
    A_N100[np.where((A_N110 > 0.75) & (A_N110 <= 0.85))]
)
B_N7[24] = np.nanmean(
    A_N100[np.where((A_N110 > 0.85) & (A_N110 <= 0.95))]
)
B_N7[25] = np.nanmean(
    A_N100[np.where((A_N110 > 0.95) & (A_N110 <= 1.05))]
)
B_N7[26] = np.nanmean(
    A_N100[np.where((A_N110 > 1.05) & (A_N110 <= 1.15))]
)
B_N7[27] = np.nanmean(
    A_N100[np.where((A_N110 > 1.15) & (A_N110 <= 1.25))]
)

# 2018
B_N8 = np.zeros((28))
B_N8[0] = np.nanmean(
    A_N101[np.where((A_N111 > -1.55) & (A_N111 <= -1.45))]
)
B_N8[1] = np.nanmean(
    A_N101[np.where((A_N111 > -1.45) & (A_N111 <= -1.35))]
)
B_N8[2] = np.nanmean(
    A_N101[np.where((A_N111 > -1.35) & (A_N111 <= -1.25))]
)
B_N8[3] = np.nanmean(
    A_N101[np.where((A_N111 > -1.25) & (A_N111 <= -1.15))]
)
B_N8[4] = np.nanmean(
    A_N101[np.where((A_N111 > -1.15) & (A_N111 <= -1.05))]
)
B_N8[5] = np.nanmean(
    A_N101[np.where((A_N111 > -1.05) & (A_N111 <= -0.95))]
)
B_N8[6] = np.nanmean(
    A_N101[np.where((A_N111 > -0.95) & (A_N111 <= -0.85))]
)
B_N8[7] = np.nanmean(
    A_N101[np.where((A_N111 > -0.85) & (A_N111 <= -0.75))]
)
B_N8[8] = np.nanmean(
    A_N101[np.where((A_N111 > -0.75) & (A_N111 <= -0.65))]
)
B_N8[9] = np.nanmean(
    A_N101[np.where((A_N111 > -0.65) & (A_N111 <= -0.55))]
)
B_N8[10] = np.nanmean(
    A_N101[np.where((A_N111 > -0.55) & (A_N111 <= -0.45))]
)
B_N8[11] = np.nanmean(
    A_N101[np.where((A_N111 > -0.45) & (A_N111 <= -0.35))]
)
B_N8[12] = np.nanmean(
    A_N101[np.where((A_N111 > -0.35) & (A_N111 <= -0.25))]
)
B_N8[13] = np.nanmean(
    A_N101[np.where((A_N111 > -0.25) & (A_N111 <= -0.15))]
)
B_N8[14] = np.nanmean(
    A_N101[np.where((A_N111 > -0.15) & (A_N111 <= -0.05))]
)
B_N8[15] = np.nanmean(
    A_N101[np.where((A_N111 > -0.05) & (A_N111 <= 0.05))]
)
B_N8[16] = np.nanmean(
    A_N101[np.where((A_N111 > 0.05) & (A_N111 <= 0.15))]
)
B_N8[17] = np.nanmean(
    A_N101[np.where((A_N111 > 0.15) & (A_N111 <= 0.25))]
)
B_N8[18] = np.nanmean(
    A_N101[np.where((A_N111 > 0.25) & (A_N111 <= 0.35))]
)
B_N8[19] = np.nanmean(
    A_N101[np.where((A_N111 > 0.35) & (A_N111 <= 0.45))]
)
B_N8[20] = np.nanmean(
    A_N101[np.where((A_N111 > 0.45) & (A_N111 <= 0.55))]
)
B_N8[21] = np.nanmean(
    A_N101[np.where((A_N111 > 0.55) & (A_N111 <= 0.65))]
)
B_N8[22] = np.nanmean(
    A_N101[np.where((A_N111 > 0.65) & (A_N111 <= 0.75))]
)
B_N8[23] = np.nanmean(
    A_N101[np.where((A_N111 > 0.75) & (A_N111 <= 0.85))]
)
B_N8[24] = np.nanmean(
    A_N101[np.where((A_N111 > 0.85) & (A_N111 <= 0.95))]
)
B_N8[25] = np.nanmean(
    A_N101[np.where((A_N111 > 0.95) & (A_N111 <= 1.05))]
)
B_N8[26] = np.nanmean(
    A_N101[np.where((A_N111 > 1.05) & (A_N111 <= 1.15))]
)
B_N8[27] = np.nanmean(
    A_N101[np.where((A_N111 > 1.15) & (A_N111 <= 1.25))]
)

# 2019
B_N9 = np.zeros((28))
B_N9[0] = np.nanmean(
    A_N102[np.where((A_N112 > -1.55) & (A_N112 <= -1.45))]
)
B_N9[1] = np.nanmean(
    A_N102[np.where((A_N112 > -1.45) & (A_N112 <= -1.35))]
)
B_N9[2] = np.nanmean(
    A_N102[np.where((A_N112 > -1.35) & (A_N112 <= -1.25))]
)
B_N9[3] = np.nanmean(
    A_N102[np.where((A_N112 > -1.25) & (A_N112 <= -1.15))]
)
B_N9[4] = np.nanmean(
    A_N102[np.where((A_N112 > -1.15) & (A_N112 <= -1.05))]
)
B_N9[5] = np.nanmean(
    A_N102[np.where((A_N112 > -1.05) & (A_N112 <= -0.95))]
)
B_N9[6] = np.nanmean(
    A_N102[np.where((A_N112 > -0.95) & (A_N112 <= -0.85))]
)
B_N9[7] = np.nanmean(
    A_N102[np.where((A_N112 > -0.85) & (A_N112 <= -0.75))]
)
B_N9[8] = np.nanmean(
    A_N102[np.where((A_N112 > -0.75) & (A_N112 <= -0.65))]
)
B_N9[9] = np.nanmean(
    A_N102[np.where((A_N112 > -0.65) & (A_N112 <= -0.55))]
)
B_N9[10] = np.nanmean(
    A_N102[np.where((A_N112 > -0.55) & (A_N112 <= -0.45))]
)
B_N9[11] = np.nanmean(
    A_N102[np.where((A_N112 > -0.45) & (A_N112 <= -0.35))]
)
B_N9[12] = np.nanmean(
    A_N102[np.where((A_N112 > -0.35) & (A_N112 <= -0.25))]
)
B_N9[13] = np.nanmean(
    A_N102[np.where((A_N112 > -0.25) & (A_N112 <= -0.15))]
)
B_N9[14] = np.nanmean(
    A_N102[np.where((A_N112 > -0.15) & (A_N112 <= -0.05))]
)
B_N9[15] = np.nanmean(
    A_N102[np.where((A_N112 > -0.05) & (A_N112 <= 0.05))]
)
B_N9[16] = np.nanmean(
    A_N102[np.where((A_N112 > 0.05) & (A_N112 <= 0.15))]
)
B_N9[17] = np.nanmean(
    A_N102[np.where((A_N112 > 0.15) & (A_N112 <= 0.25))]
)
B_N9[18] = np.nanmean(
    A_N102[np.where((A_N112 > 0.25) & (A_N112 <= 0.35))]
)
B_N9[19] = np.nanmean(
    A_N102[np.where((A_N112 > 0.35) & (A_N112 <= 0.45))]
)
B_N9[20] = np.nanmean(
    A_N102[np.where((A_N112 > 0.45) & (A_N112 <= 0.55))]
)
B_N9[21] = np.nanmean(
    A_N102[np.where((A_N112 > 0.55) & (A_N112 <= 0.65))]
)
B_N9[22] = np.nanmean(
    A_N102[np.where((A_N112 > 0.65) & (A_N112 <= 0.75))]
)
B_N9[23] = np.nanmean(
    A_N102[np.where((A_N112 > 0.75) & (A_N112 <= 0.85))]
)
B_N9[24] = np.nanmean(
    A_N102[np.where((A_N112 > 0.85) & (A_N112 <= 0.95))]
)
B_N9[25] = np.nanmean(
    A_N102[np.where((A_N112 > 0.95) & (A_N112 <= 1.05))]
)
B_N9[26] = np.nanmean(
    A_N102[np.where((A_N112 > 1.05) & (A_N112 <= 1.15))]
)
B_N9[27] = np.nanmean(
    A_N102[np.where((A_N112 > 1.15) & (A_N112 <= 1.25))]
)

# 2020
B_N10 = np.zeros((28))
B_N10[0] = np.nanmean(
    A_N103[np.where((A_N113 > -1.55) & (A_N113 <= -1.45))]
)
B_N10[1] = np.nanmean(
    A_N103[np.where((A_N113 > -1.45) & (A_N113 <= -1.35))]
)
B_N10[2] = np.nanmean(
    A_N103[np.where((A_N113 > -1.35) & (A_N113 <= -1.25))]
)
B_N10[3] = np.nanmean(
    A_N103[np.where((A_N113 > -1.25) & (A_N113 <= -1.15))]
)
B_N10[4] = np.nanmean(
    A_N103[np.where((A_N113 > -1.15) & (A_N113 <= -1.05))]
)
B_N10[5] = np.nanmean(
    A_N103[np.where((A_N113 > -1.05) & (A_N113 <= -0.95))]
)
B_N10[6] = np.nanmean(
    A_N103[np.where((A_N113 > -0.95) & (A_N113 <= -0.85))]
)
B_N10[7] = np.nanmean(
    A_N103[np.where((A_N113 > -0.85) & (A_N113 <= -0.75))]
)
B_N10[8] = np.nanmean(
    A_N103[np.where((A_N113 > -0.75) & (A_N113 <= -0.65))]
)
B_N10[9] = np.nanmean(
    A_N103[np.where((A_N113 > -0.65) & (A_N113 <= -0.55))]
)
B_N10[10] = np.nanmean(
    A_N103[np.where((A_N113 > -0.55) & (A_N113 <= -0.45))]
)
B_N10[11] = np.nanmean(
    A_N103[np.where((A_N113 > -0.45) & (A_N113 <= -0.35))]
)
B_N10[12] = np.nanmean(
    A_N103[np.where((A_N113 > -0.35) & (A_N113 <= -0.25))]
)
B_N10[13] = np.nanmean(
    A_N103[np.where((A_N113 > -0.25) & (A_N113 <= -0.15))]
)
B_N10[14] = np.nanmean(
    A_N103[np.where((A_N113 > -0.15) & (A_N113 <= -0.05))]
)
B_N10[15] = np.nanmean(
    A_N103[np.where((A_N113 > -0.05) & (A_N113 <= 0.05))]
)
B_N10[16] = np.nanmean(
    A_N103[np.where((A_N113 > 0.05) & (A_N113 <= 0.15))]
)
B_N10[17] = np.nanmean(
    A_N103[np.where((A_N113 > 0.15) & (A_N113 <= 0.25))]
)
B_N10[18] = np.nanmean(
    A_N103[np.where((A_N113 > 0.25) & (A_N113 <= 0.35))]
)
B_N10[19] = np.nanmean(
    A_N103[np.where((A_N113 > 0.35) & (A_N113 <= 0.45))]
)
B_N10[20] = np.nanmean(
    A_N103[np.where((A_N113 > 0.45) & (A_N113 <= 0.55))]
)
B_N10[21] = np.nanmean(
    A_N103[np.where((A_N113 > 0.55) & (A_N113 <= 0.65))]
)
B_N10[22] = np.nanmean(
    A_N103[np.where((A_N113 > 0.65) & (A_N113 <= 0.75))]
)
B_N10[23] = np.nanmean(
    A_N103[np.where((A_N113 > 0.75) & (A_N113 <= 0.85))]
)
B_N10[24] = np.nanmean(
    A_N103[np.where((A_N113 > 0.85) & (A_N113 <= 0.95))]
)
B_N10[25] = np.nanmean(
    A_N103[np.where((A_N113 > 0.95) & (A_N113 <= 1.05))]
)
B_N10[26] = np.nanmean(
    A_N103[np.where((A_N113 > 1.05) & (A_N113 <= 1.15))]
)
B_N10[27] = np.nanmean(
    A_N103[np.where((A_N113 > 1.15) & (A_N113 <= 1.25))]
)


PCA = np.arange(-1.5, 1.3, 0.1)
# plt.plot(PCA, B_N,label='2010')
plt.plot(PCA, B_N0 * 100, color="blue", label="2011-2019")
# plt.plot(PCA, B_N1,label='2011')
# plt.plot(PCA, B_N2,label='2012')
# plt.plot(PCA, B_N3,label='2013')
# plt.plot(PCA, B_N4,label='2014')
# plt.plot(PCA, B_N5,label='2015')
# plt.plot(PCA, B_N6,label='2016')
# plt.plot(PCA, B_N7,label='2017')
# plt.plot(PCA, B_N8,label='2018')
plt.plot(PCA, B_N9 * 100, color="green", label="2019")
plt.plot(PCA, B_N10 * 100, color="red", label="2020")

plt.legend()
plt.title("EOF-MODIS_highcloudfraction", fontsize=18)
plt.xlabel("EOF", fontsize=14)
plt.ylabel("MODIS_highcloudfraction(%)", fontsize=14)

################################################################################

# D_N = np.zeros((81,141))
# C_N = np.zeros((2,280,81,141))
# A_N1 = A_N1.reshape(280,81,141)
# A_N10 = A_N10.reshape(280,81,141)
# A_N11 = A_N11.reshape(280,81,141)
# A_N12 = A_N12.reshape(280,81,141)
# A_N20 = A_N20.reshape(280,81,141)

# C_N[0,:,:,:] = A_N1[:,:,:]
# C_N[1,:,:,:] = A_N20[:,:,:]

# for i in range(0,81):
#     for j in range(0,141):
#         D_N[i,j] = pd.Series(C_N[0,:,i,j]).corr(pd.Series(C_N[1,:,i,j]), method='pearson')

# lon = np.linspace(40,180,141)
# lat = np.linspace(0,80,81)

# fig, ax = plt.subplots(figsize=(10,10),constrained_layout=True,dpi=200)
# plt.rc('font', size=10, weight='bold')

# #basemap设置部分
# map = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=80,llcrnrlon=40,urcrnrlon=180,resolution='l')
# parallels = np.arange(0,80+20,20) #纬线
# map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
# ax.set_yticks(parallels,len(parallels)*[''])
# meridians = np.arange(40,180+20,20) #经线
# map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
# ax.set_xticks(meridians,len(meridians)*[''])

# map.drawcountries(linewidth=1.5)
# map.drawcoastlines()
# lon, lat = np.meshgrid(lon, lat)
# cmap=dcmap('F://color/test6.txt')
# cmap.set_bad('gray')
# a=map.pcolormesh(lon, lat,D_N,cmap=cmap)
# fig.colorbar(a)
# ax.set_title('EOF-cldarea Pearson R',size=16)

# plt.show()
