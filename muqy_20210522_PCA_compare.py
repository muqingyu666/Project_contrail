# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:12:48 2021

@author: Mu o(*￣▽￣*)ブ
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from scipy import stats
from scipy.stats import zscore
from sklearn.decomposition import PCA

# Assign blank array
Ins_N = np.zeros((1461, 41, 41))
RH_N = np.zeros((1461, 41, 41))
T_N = np.zeros((1461, 41, 41))
A_N10 = np.zeros((1))
A_N11 = np.zeros((1))
A_N12 = np.zeros((1))
A_N13 = np.zeros((1))
W_N = np.zeros((1461, 41, 41))


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


# Read data and variable
data = xr.open_dataset("G:\\T_RH_w_dTdZ_2007-2010_vect.nc")
W = data.w
RH = data.RH
T = data.T
Ins = data.dTdZ
W = W[:, 17, :, :]
RH = RH[:, 17, :, :]
T = T[:, 17, :, :]
Ins = Ins[:, 17, :, :]

# Normalize data
# for i in range(0,41):
#     for j in range(0,41):
#             RH_N[:,i,j] = ZscoreNormalization(RH[:,i,j])
#             T_N[:,i,j] = ZscoreNormalization(T[:,i,j])
#             W_N[:,i,j] = ZscoreNormalization(W[:,i,j])
#             Ins_N[:,i,j] = ZscoreNormalization(Ins[:,i,j])

for i in range(0, 41):
    for j in range(0, 41):
        RH_N = stats.zscore(RH, axis=0)
        T_N = stats.zscore(T, axis=0)
        W_N = stats.zscore(W, axis=0)
        Ins_N = stats.zscore(Ins, axis=0)

# Combine all data into big array 1-dimentional
for i in range(0, 1461):
    # RH_N[i,:,:] = ZscoreNormalization(RH[i,:,:])
    # T_N[i,:,:] = ZscoreNormalization(T[i,:,:])
    # W_N[i,:,:] = ZscoreNormalization(W[i,:,:])
    # Ins_N[i,:,:] = ZscoreNormalization(Ins[i,:,:])
    RH_N1 = RH_N[i, :, :].reshape(1681)
    T_N1 = T_N[i, :, :].reshape(1681)
    W_N1 = W_N[i, :, :].reshape(1681)
    Ins_N1 = Ins_N[i, :, :].reshape(1681)
    A_N10 = np.concatenate((A_N10, RH_N1), axis=0)
    A_N11 = np.concatenate((A_N11, T_N1), axis=0)
    A_N12 = np.concatenate((A_N12, W_N1), axis=0)
    A_N13 = np.concatenate((A_N13, Ins_N1), axis=0)

# Delete the first 0
A_N10 = np.delete(A_N10, 0, axis=0)
A_N11 = np.delete(A_N11, 0, axis=0)
A_N12 = np.delete(A_N12, 0, axis=0)
A_N13 = np.delete(A_N13, 0, axis=0)

A_N1 = np.zeros((2455941, 4))
A_N1[:, 0] = A_N10
A_N1[:, 1] = A_N11
A_N1[:, 2] = A_N12
A_N1[:, 3] = A_N13

# PCA function
pca = PCA(n_components=1, whiten=True)
pca.fit(A_N1)
A_N1 = pca.fit_transform(A_N1)
A_N1 = A_N1.reshape(1461, 41, 41)

# Elder martial brother's Value
data0 = xr.open_dataset("G:\\EOF_space_field.nc")
eof = data0.EOF


def PCAcompare(n):

    lon1 = data.lon
    lat1 = data.lat

    lon2 = data0.lon
    lat2 = data0.lat

    fig, axs = plt.subplots(
        1, 2, figsize=(10, 10), constrained_layout=True, dpi=200
    )
    # fig, ax = plt.subplots(figsize=(10,10),constrained_layout=True,dpi=120)
    plt.rc("font", size=10, weight="bold")

    # basemap设置部分
    ax = axs[0]
    map = Basemap(
        projection="cyl",
        llcrnrlat=20,
        urcrnrlat=50,
        llcrnrlon=75,
        urcrnrlon=105,
        resolution="l",
        ax=axs[0],
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
    lon1, lat1 = np.meshgrid(lon1, lat1)
    cmap = dcmap("F://color/test6.txt")
    cmap.set_bad("gray")
    # a=map.pcolormesh(lon, lat,ek1,cmap=cmap,vmin=-0.5,vmax=16.5)
    a = map.pcolormesh(lon1, lat1, A_N1[n, :, :], cmap=cmap)
    ax.set_title("PCA using Python", size=16)

    ax = axs[1]
    map = Basemap(
        projection="cyl",
        llcrnrlat=20,
        urcrnrlat=50,
        llcrnrlon=75,
        urcrnrlon=105,
        resolution="l",
        ax=axs[1],
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
    lon2, lat2 = np.meshgrid(lon2, lat2)
    cmap = dcmap("F://color/test6.txt")
    cmap.set_bad("gray")
    # a=map.pcolormesh(lon, lat,ek1,cmap=cmap,vmin=-0.5,vmax=16.5)
    a = map.pcolormesh(lon2, lat2, eof[n, :, :], cmap=cmap)
    ax.set_title("PCA using NCL", size=16)

    fig.colorbar(a, ax=axs[:], location="right", shrink=0.4)
    plt.savefig("G:\\PCA_compare_" + str(n), dpi=200)
    # plt.show()


for n in range(0, 11):
    PCAcompare(n)
