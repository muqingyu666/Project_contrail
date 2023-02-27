# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 09:25:14 2021

@author: Mu o(*￣▽￣*)ブ
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from sklearn.decomposition import PCA
import glob


def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x


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


data = xr.open_dataset("f://T_era5.nc")

A_N1 = np.zeros((1, 1))


def era():

    R1 = np.zeros((30, 30))
    T1 = np.zeros((30, 30))
    T11 = np.zeros((30, 30))
    W1 = np.zeros((30, 30))
    Ins1 = np.zeros((30, 30))
    Ins1_N = np.zeros((30, 30))
    Z1 = np.zeros((30, 30))
    Z11 = np.zeros((30, 30))
    R1_N = np.zeros((30, 30))
    T1_N = np.zeros((30, 30))
    A_N10 = np.zeros((1))
    A_N11 = np.zeros((1))
    A_N12 = np.zeros((1))
    A_N13 = np.zeros((1))
    W1_N = np.zeros((30, 30))

    for i in range(1, 32):
        year_str = str(2007).zfill(4)
        month_str = str(1).zfill(2)
        day_str = str(i).zfill(2)
        time_str = year_str + month_str + day_str
        ERA_file0 = glob.glob(
            "G:\\200701\\div_geo_RH_T" + time_str + "*.nc"
        )
        FILE_NAME0 = ERA_file0[0]
        data0 = xr.open_dataset(FILE_NAME0)
        R = data0["r"]
        R = np.mean(R[:, 17, 80:200, 60:180], axis=0)
        T = data0["t"]
        T0 = np.mean(T[:, 17, 80:200, 60:180], axis=0)
        T2 = np.mean(T[:, 16, 80:200, 60:180], axis=0)
        Z = data0["z"]
        Z0 = np.mean(Z[:, 17, 80:200, 60:180], axis=0)
        Z2 = np.mean(Z[:, 16, 80:200, 60:180], axis=0)
        ERA_file1 = glob.glob(
            "G:\\200701\\U_V_W" + time_str + "*.nc"
        )
        FILE_NAME1 = ERA_file1[0]
        data1 = xr.open_dataset(FILE_NAME1)
        W = data1["w"]
        W = np.mean(W[:, 17, 0:120, 60:180], axis=0)
        T_1 = np.delete(T0, -1, axis=0)
        T_11 = np.delete(T2, -1, axis=0)
        R_1 = np.delete(R, -1, axis=0)
        W_1 = np.delete(W, -1, axis=0)
        Z_1 = np.delete(Z0, -1, axis=0)
        Z_11 = np.delete(Z2, -1, axis=0)
        T_1 = np.delete(T_1, -1, axis=1)
        T_11 = np.delete(T_11, -1, axis=1)
        R_1 = np.delete(R_1, -1, axis=1)
        W_1 = np.delete(W_1, -1, axis=1)
        Z_1 = np.delete(Z0, -1, axis=1)
        Z_11 = np.delete(Z2, -1, axis=1)
        T1[:, :] = T_1[::4, ::4]
        T11[:, :] = T_11[::4, ::4]
        R1[:, :] = R_1[::4, ::4]
        W1[:, :] = W_1[::4, ::4]
        Z1[:, :] = Z_1[::4, ::4]
        Z11[:, :] = Z_11[::4, ::4]
        Sita = T1 * (1013 / 300) ** (0.286)
        Sita1 = T1 * (1013 / 250) ** (0.286)
        Ins = (Sita - Sita1) / (Z1 - Z11)
        R1[:, :] = np.flipud(R1[:, :])
        T1[:, :] = np.flipud(T1[:, :])
        W1[:, :] = np.flipud(W1[:, :])
        Ins1[:, :] = np.flipud(Ins[:, :])
        R1_N[:, :] = ZscoreNormalization(R1[:, :])
        T1_N[:, :] = ZscoreNormalization(T1[:, :])
        T1_N[:, :] = ZscoreNormalization(T1[:, :])
        W1_N[:, :] = ZscoreNormalization(W1[:, :])
        Ins1_N[:, :] = ZscoreNormalization(Ins1[:, :])
        R1_N1 = R1_N[:, :].reshape(900)
        T1_N1 = T1_N[:, :].reshape(900)
        W1_N1 = W1_N[:, :].reshape(900)
        Ins1_N1 = Ins1_N[:, :].reshape(900)
        A_N10 = np.concatenate((A_N10, R1_N1), axis=0)
        A_N11 = np.concatenate((A_N11, T1_N1), axis=0)
        A_N12 = np.concatenate((A_N12, W1_N1), axis=0)
        A_N13 = np.concatenate((A_N13, Ins1_N1), axis=0)

    return A_N10, A_N11, A_N12, A_N13


A_N10, A_N11, A_N12, A_N13 = era()
A_N10 = np.delete(A_N10, 0, axis=0)
A_N11 = np.delete(A_N11, 0, axis=0)
A_N12 = np.delete(A_N12, 0, axis=0)
A_N13 = np.delete(A_N13, 0, axis=0)

A_N1 = np.zeros((27900, 3))
A_N1[:, 0] = A_N10
A_N1[:, 1] = A_N11
A_N1[:, 2] = A_N12
# A_N1[:,3] = A_N13

pca = PCA(n_components=1, whiten=True)
pca.fit(A_N1)
A_N1 = pca.fit_transform(A_N1)
# A_N1 = A_N1.reshape(30,30)
# def split(array):
#     A_0,A_1,A_2,A_3,A_4,A_5,A_6,A_7,A_8,A_9,A_10,A_11,A_12,A_13,A_14,A_15,A_16,A_17,A_18,A_19,A_20,A_21,A_22,A_23,A_24,A_25,A_26,A_27,A_28,A_29,A_30=np.split(array,31,axis=0)
#     return A_0,A_1,A_2,A_3,A_4,A_5,A_6,A_7,A_8,A_9,A_10,A_11,A_12,A_13,A_14,A_15,A_16,A_17,A_18,A_19,A_20,A_21,A_22,A_23,A_24,A_25,A_26,A_27,A_28,A_29,A_30

# A_0,A_1,A_2,A_3,A_4,A_5,A_6,A_7,A_8,A_9,A_10,A_11,A_12,A_13,A_14,A_15,A_16,A_17,A_18,A_19,A_20,A_21,A_22,A_23,A_24,A_25,A_26,A_27,A_28,A_29,A_30 = split(A_N1)
# A_0 = A_0.reshape(30,30)
A_N1 = A_N1.reshape(31, 30, 30)
# A_N1 = ZscoreNormalization(A_N1)
# data = xr.open_dataset('G:\\200701\div_geo_RH_T20070101.nc')
# r = data.r
# r = r[0,17,80:200,60:180]
# r = np.flipud(r)
# r1=r[::4,::4]

lon = np.linspace(75, 105, 30)
lat = np.linspace(20, 50, 30)
# fig = plt.figure(figsize=(10, 10))

fig, ax = plt.subplots(
    figsize=(10, 10), constrained_layout=True, dpi=120
)
plt.rc("font", size=10, weight="bold")

# basemap设置部分
map = Basemap(
    projection="cyl",
    llcrnrlat=20,
    urcrnrlat=50,
    llcrnrlon=75,
    urcrnrlon=105,
    resolution="l",
)
parallels = np.arange(20, 50 + 10, 10)  # 纬线
map.drawparallels(
    parallels,
    labels=[True, False, False, False],
    linewidth=0.01,
    dashes=[1, 400],
)
ax.set_yticks(parallels, len(parallels) * [""])
meridians = np.arange(75, 105 + 10, 10)  # 经线
map.drawmeridians(
    meridians,
    labels=[False, False, False, True],
    linewidth=0.01,
    dashes=[1, 400],
)
ax.set_xticks(meridians, len(meridians) * [""])

map.drawcountries(linewidth=1.5)
map.drawcoastlines()
lon, lat = np.meshgrid(lon, lat)
cmap = dcmap("F://color/test6.txt")
cmap.set_bad("gray")
a = map.pcolormesh(lon, lat, A_N1[0, :, :], cmap=cmap)
fig.colorbar(a)
ax.set_title("ERA5 300hPa T,RH,W", size=16)
plt.show()
