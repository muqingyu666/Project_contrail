# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:11:09 2021

@author: Mu o(*￣▽￣*)ブ
"""
import calendar
import os
import time
from itertools import product

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from metpy.units import units
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import norm, zscore
from sklearn.decomposition import PCA


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

data0 = xr.open_dataset("G:\\result_data.nc")

A_NNN20202 = np.array(data0["2020data"])
A_NNN20192 = np.array(data0["2019data"])
A_NNN20182 = np.array(data0["2018data"])

########################## reshape the array ###############################################################


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


################################# plot function ########################################################################

K = np.arange(-2.25, 5.8, 0.5)


def Aplot(i):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1 = plt.subplot(
        311, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(A_NNN20202[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=5,
        vmin=-5,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(a, ax=[ax1], shrink=0.9, extend="both")
    ax1.set_title(
        "2020-MEAN for EOF"
        + "("
        + str(K[i])
        + ","
        + str(K[i + 1])
        + ")",
        size=15,
    )

    ax2 = plt.subplot(
        312, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    b = ax2.pcolormesh(
        lon,
        lat,
        np.nanmean(A_NNN20192[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=5,
        vmin=-5,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
    ax2.set_title(
        "2019-MEAN for EOF"
        + "("
        + str(K[i])
        + ","
        + str(K[i + 1])
        + ")",
        size=15,
    )

    ax3 = plt.subplot(
        313, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    b = ax3.pcolormesh(
        lon,
        lat,
        np.nanmean(A_NNN20182[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=5,
        vmin=-5,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax3], shrink=0.9, extend="both")
    ax3.set_title(
        "2018-MEAN for EOF"
        + "("
        + str(K[i])
        + ","
        + str(K[i + 1])
        + ")",
        size=15,
    )
    # plt.savefig('/RAID01/data/muqy/PYTHONFIG/'+str(i)+'result.png',dpi=300,bbox_inches='tight')


Aplot(5)
for i in range(0, 16):
    Aplot(i)

