# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:38:12 2021

@author: Mu o(*￣▽￣*)ブ
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from scipy import stats
from scipy.stats import zscore
import netCDF4 as nc
from sklearn.decomposition import PCA
import glob
import pandas as pd
import os
import metpy.calc as mpcalc
from metpy.units import units
import seaborn as sns
import matplotlib.colors as colors
from scipy.stats import norm


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


#############################################################################################

A_N10 = np.zeros((1))
A_N11 = np.zeros((1))
A_N12 = np.zeros((1))
A_N13 = np.zeros((1))

for i in range(0, 365):

    # year_str=str(2010).zfill(4)
    # month_str=str(1).zfill(2)
    # time_str0=year_str+month_str

    ERA_file = glob.glob("E:\\ERA\\div_geo_RH_T2007" + "*.nc")
    FILE_NAME_ERA = ERA_file[i]
    file_obj = xr.open_dataset(FILE_NAME_ERA)

    z = file_obj.z
    RH = file_obj.r
    T = file_obj.t
    T250 = np.mean(T[:, 16, ::4, ::4], axis=0)
    T300 = np.mean(T[:, 17, ::4, ::4], axis=0)
    RH300 = np.mean(RH[:, 17, ::4, ::4], axis=0)
    RH250 = np.mean(RH[:, 16, ::4, ::4], axis=0)
    Z300 = np.mean(z[:, 17, ::4, ::4], axis=0)
    Z250 = np.mean(z[:, 16, ::4, ::4], axis=0)

    T250[:, :] = np.flipud(T250[:, :])
    T300[:, :] = np.flipud(T300[:, :])
    RH300[:, :] = np.flipud(RH300[:, :])
    RH250[:, :] = np.flipud(RH250[:, :])
    Z300[:, :] = np.flipud(Z300[:, :])
    Z250[:, :] = np.flipud(Z250[:, :])

    RH300_1 = np.array(RH300[0:31, 15:46]).reshape(961)
    RH250_1 = np.array(RH250[0:31, 15:46]).reshape(961)
    T250_1 = np.array(T250[0:31, 15:46]).reshape(961)
    T300_1 = np.array(T300[0:31, 15:46]).reshape(961)
    Z300_1 = np.array(Z300[0:31, 15:46]).reshape(961)
    Z250_1 = np.array(Z250[0:31, 15:46]).reshape(961)

    dewpoint300 = np.array(
        mpcalc.dewpoint_from_relative_humidity(
            T300_1 * units.kelvin, RH300_1 * units.dimensionless
        )
    )
    dewpoint250 = np.array(
        mpcalc.dewpoint_from_relative_humidity(
            T250_1 * units.kelvin, RH250_1 * units.dimensionless
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
    A_N13 = np.concatenate((A_N13, stab), axis=0)

for i in range(0, 365):

    ERA_file = glob.glob("E:\\ERA\\U_V_W2007" + "*.nc")
    FILE_NAME_ERA = ERA_file[i]
    file_obj = xr.open_dataset(FILE_NAME_ERA)

    W = file_obj.w
    W = np.mean(W[:, 17, ::4, ::4], axis=0)
    W[:, :] = np.flipud(W[:, :])
    W_1 = np.array(W[0:31, 15:46]).reshape(961)
    A_N12 = np.concatenate((A_N12, W_1), axis=0)


A_N10 = np.delete(A_N10, 0, axis=0)  # RH
A_N11 = np.delete(A_N11, 0, axis=0)  # T
A_N12 = np.delete(A_N12, 0, axis=0)  # W
A_N13 = np.delete(A_N13, 0, axis=0)  # stability sita/z

A_N10m = np.mean(A_N10.reshape(365, 31, 31), axis=0)  # RH
A_N11m = np.mean(A_N11.reshape(365, 31, 31), axis=0)  # T
A_N12m = np.mean(A_N12.reshape(365, 31, 31), axis=0)  # W
A_N13m = np.mean(
    A_N13.reshape(365, 31, 31), axis=0
)  # stability sita/z

A_N10 = A_N10.reshape(350765)
A_N11 = A_N11.reshape(350765)
A_N12 = A_N12.reshape(350765)
A_N13 = A_N13.reshape(350765)

A_N10_N = stats.zscore(A_N10)
A_N11_N = stats.zscore(A_N11)
A_N12_N = stats.zscore(A_N12)
A_N13_N = stats.zscore(A_N13)

A_N1 = np.zeros((350765, 4))
A_N1[:, 0] = A_N10_N
A_N1[:, 1] = A_N11_N
A_N1[:, 2] = A_N12_N
A_N1[:, 3] = A_N13_N

pca = PCA(n_components=1, whiten=True)
pca.fit(A_N1)
A_N1 = pca.fit_transform(A_N1)

A_N1m = np.mean(A_N1.reshape(365, 31, 31), axis=0)

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


########################################################################################

A_N20 = np.zeros((1))

path = "E://modis/"
files = os.listdir(path)
files.sort(key=lambda x: int(x[-25:-22]))
# MODIS_file=glob.glob(path +files[0]+'/MOD08_D3'+'*.hdf')

for i in range(0, 365):

    MODIS_file = files[i]
    data = nc.Dataset(path + MODIS_file)
    high = data.variables["High_Cloud_Fraction_Infrared"]

    high = high[39:70, 254:285]
    high = np.flipud(np.array(high))
    high1 = high[:, :].reshape(961)
    high1[high1 > 1] = 1
    high1[high1 < -1] = np.nan

    # high = np.flipud(np.array(high))
    # c_1,c_2 = np.split(high,2,axis=1)
    # high = np.c_[c_2,c_1]
    # high = high[89:170,39:180]
    # high1 = high[:,:].reshape(11421)
    # high1[high1<0] = 0

    A_N20 = np.concatenate((A_N20, high1), axis=0)
    # A_N21 = np.concatenate((A_N21,high1),axis=0)

A_N20 = np.delete(A_N20, 0, axis=0)  # Cirrus_Fraction
A_N20 = A_N20.reshape(365, 31, 31)

########################################################################################

D_N = np.zeros((31, 31))
C_N = np.zeros((2, 365, 31, 31))
# A_N1 = A_N1.reshape(59875200)
# A_N20t = A_N20t.reshape(59875200)

C_N[0, :] = A_N1.reshape(365, 31, 31)
C_N[1, :] = A_N20.reshape(365, 31, 31)

for i in range(0, 31):
    for j in range(0, 31):
        D_N[i, j] = pd.Series(C_N[0, :, i, j]).corr(
            pd.Series(C_N[1, :, i, j]), method="pearson"
        )


def plot(arr):
    lon = np.linspace(75, 105, 31)
    lat = np.linspace(20, 50, 31)

    fig, ax = plt.subplots(
        figsize=(10, 10), constrained_layout=True, dpi=200
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

    # map.drawcountries(linewidth=0.5)
    # map.drawcoastlines(linewidth=0.5)
    map.readshapefile("E:\\DBATP\\DBATP_Line", "tibetan")

    cmap = dcmap("F://color/test6.txt")
    cmap.set_bad("gray")
    a = map.pcolormesh(
        lon,
        lat,
        arr,
        norm=MidpointNormalize(midpoint=0),
        cmap=cmap,
    )
    fig.colorbar(a)
    ax.set_title("2007 mean EOF", size=16)

    plt.show()


plot(A_N1m)
