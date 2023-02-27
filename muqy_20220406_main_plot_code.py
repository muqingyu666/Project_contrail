# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:17:52 2021

@author: Mu o(*￣▽￣*)ブ
"""

import cartopy.crs as ccrs
from cartopy.mpl.ticker import (
    LongitudeFormatter,
    LatitudeFormatter,
)
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
import matplotlib.colors as colors
from scipy.stats import norm
from itertools import product
import time
from numba import jit
from numba import vectorize, float64
from scipy.signal import savgol_filter


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


A_NNN2020 = np.zeros((42, 84, 180, 360))
A_NNN2019 = np.zeros((42, 84, 180, 360))
A_NNN2018 = np.zeros((42, 84, 180, 360))
A_NNN2017 = np.zeros((42, 84, 180, 360))
A_NNN2017 = np.zeros((42, 84, 180, 360))
A_temp = np.zeros((42, 84))

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

########################### Pearson Coef for each year ###############################

B_N0 = np.zeros((16))
for i in range(0, 16):
    B_N0[i] = pd.Series(
        A_TEMP[:, :, i, :, :].reshape(32659200)
    ).corr(
        pd.Series(A_TEMP1[:, :, i, :, :].reshape(32659200)),
        method="pearson",
    )

#######################################################################################################
def plotbyEOFgap(k, m, n, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat1 = np.linspace(30, 59, 30)
    lat2 = np.linspace(-60, -31, 30)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(5, 8))
    # fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
    # plt.rc('font', size=10, weight='bold')
    A_NNN2020_mid1 = A_NNN2020[:, :, 120:150, :]
    A_NNN2020_mid2 = A_NNN2020[:, :, 30:60, :]
    A_NNN2019_mid1 = A_NNN2019[:, :, 120:150, :]
    A_NNN2019_mid2 = A_NNN2019[:, :, 30:60, :]
    A_NNN2018_mid1 = A_NNN2018[:, :, 120:150, :]
    A_NNN2018_mid2 = A_NNN2018[:, :, 30:60, :]

    ax1 = plt.subplot(
        311, projection=ccrs.PlateCarree(central_longitude=180)
    )
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1.coastlines(resolution="50m", lw=0.5)
    ax1.set_global()
    # a = ax1.pcolormesh(lon,lat1,np.nansum(A_NNN2020_mid1[m:n,t1:t2,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),vmax=20,vmin=-20,cmap=cmap)
    # a = ax1.pcolormesh(lon,lat2,np.nansum(A_NNN2020_mid2[m:n,t1:t2,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),vmax=20,vmin=-20,cmap=cmap)
    # a = ax1.pcolormesh(lon,lat,np.nansum(A_NNN2020[m:n,t1:t2,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),vmax=20,vmin=-20,cmap=cmap)
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nansum(A_NNN20202[m:n, t1:t2, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        vmax=20,
        vmin=-20,
        cmap=cmap,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax1.set_title(
        "Month "
        + str(t1)
        + "-"
        + str(t2)
        + "_("
        + str(np.round(np.array(m - 4.5) * 0.5, 2))
        + " =< EOF < "
        + str(np.round(np.array(n - 3.5) * 0.5, 2))
        + ")",
        size=12,
    )

    ax2 = plt.subplot(
        312, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    # a = ax2.pcolormesh(lon,lat1,np.nansum(A_NNN2019_mid1[m:n,t1:t2,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),vmax=20,vmin=-20,cmap=cmap)
    # a = ax2.pcolormesh(lon,lat2,np.nansum(A_NNN2019_mid2[m:n,t1:t2,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),vmax=20,vmin=-20,cmap=cmap)
    a = ax2.pcolormesh(
        lon,
        lat,
        np.nansum(A_NNN20192[m:n, t1:t2, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        vmax=20,
        vmin=-20,
        cmap=cmap,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    # ax2.set_title(str(t1)+'-'+str(t2)+' 2019 Cumulative anomaly percentage('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)

    ax3 = plt.subplot(
        313, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    # a = ax3.pcolormesh(lon,lat1,np.nansum(A_NNN2018_mid1[m:n,t1:t2,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),vmax=20,vmin=-20,cmap=cmap)
    # a = ax3.pcolormesh(lon,lat2,np.nansum(A_NNN2018_mid2[m:n,t1:t2,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),vmax=20,vmin=-20,cmap=cmap)
    a = ax3.pcolormesh(
        lon,
        lat,
        np.nansum(A_NNN20182[m:n, t1:t2, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        vmax=20,
        vmin=-20,
        cmap=cmap,
    )
    # a = ax3.pcolormesh(lon,lat,np.nansum(A_NNN2018[m:n,t1:t2,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),vmax=20,vmin=-20,cmap=cmap)
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False

    fig.colorbar(
        a,
        ax=[ax1, ax2, ax3],
        location="right",
        shrink=0.9,
        extend="both",
    )
    # plt.savefig('C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Cldarea/'+str(k)+str('%02d' % m)+'_'+str(t1)+'-'+str(t2)+'.png',dpi=300,bbox_inches='tight')

    plt.show()


plotbyEOFgap("cldarea", 7, 16, 0, 7)

A_NK1 = np.nanmean(A_NK.reshape(504, 180, 360), axis=0)
A_NKK1 = np.nanmean(A_NKK.reshape(504, 180, 360), axis=0)
A_N1031 = np.nanmean(A_N103.reshape(168, 180, 360), axis=0)
A_N1021 = np.nanmean(A_N102.reshape(168, 180, 360), axis=0)
A_N1011 = np.nanmean(A_N101.reshape(168, 180, 360), axis=0)
B_N103 = A_N1031 - A_NK1
B_N102 = A_N1021 - A_NK1
B_N101 = A_N1011 - A_NK1


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
        vmax=10,
        vmin=-10,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(a, ax=[ax1], shrink=0.9, extend="both")
    ax1.set_title("2020-MEAN", size=15)

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
        vmax=10,
        vmin=-10,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
    ax2.set_title("2019-MEAN", size=15)

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
        vmax=10,
        vmin=-10,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax3], shrink=0.9, extend="both")
    ax3.set_title("2018-MEAN ", size=15)
    plt.savefig(
        "/RAID01/data/muqy/PYTHONFIG/" + str(i) + "result.png",
        dpi=300,
        bbox_inches="tight",
    )


for i in range(0, 16):
    Aplot(i)

np.nansum(A_NNN20205, axis=0) + np.nansum(
    A_NNN20195, axis=0
) + np.nansum(A_NNN20185, axis=0)


def plotEOFCld(k, m):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    fig = plt.figure(figsize=(10, 10))
    # fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
    # plt.rc('font', size=10, weight='bold')

    ax1 = plt.subplot(
        621, projection=ccrs.PlateCarree(central_longitude=180)
    )
    cmap = dcmap("F://color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    # cmap.set_under('#191970')
    cmap.set_under("white")
    ax1.coastlines(resolution="50m", lw=0.5)
    ax1.set_global()
    # b = ax1.pcolormesh(lon,lat,np.nanmean(T_NNN2020[m:40,:,:],axis=0),transform=ccrs.PlateCarree(),vmax=(m+1)*0.2-0.09,vmin=m*0.2+0.03,cmap=cmap)
    b = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(T_NNN2020[m:40, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        vmax=1.5,
        vmin=-1.5,
        cmap=cmap,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax1.set_title(
        " 7 month mean EOF("
        + str(round((m + 1) * 0.2, 2))
        + ">EOF>="
        + str(round(m * 0.2, 2))
        + ")",
        size=12,
    )

    ax2 = plt.subplot(
        622, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    # a = ax2.pcolormesh(lon,lat,np.nanmean(Y_NNN2020[m:40,:,:],axis=0),transform=ccrs.PlateCarree(),vmax=((m+1)*0.2)*11.56+10.119,vmin=0,cmap=cmap)
    a = ax2.pcolormesh(
        lon,
        lat,
        np.nanmean(Y_NNN2020[m:40, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        vmax=30,
        vmin=0,
        cmap=cmap,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax2.set_title(
        " 7 month mean Cldarea("
        + str(round((m + 1) * 0.2, 2))
        + ">EOF>="
        + str(round(m * 0.2, 2))
        + ")",
        size=12,
    )

    ax3 = plt.subplot(
        623, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    # a = ax3.pcolormesh(lon,lat,np.nansum(T_NNN2018[m:n,:,:],axis=(0)),transform=ccrs.PlateCarree(),vmax=0.72,vmin=m*0.2,cmap=cmap)
    b = ax3.pcolormesh(
        lon,
        lat,
        T_NNN2019[m, :, :],
        transform=ccrs.PlateCarree(),
        vmax=(m + 1) * 0.2 - 0.09,
        vmin=m * 0.2 + 0.03,
        cmap=cmap,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False

    ax4 = plt.subplot(
        624, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax4.coastlines(resolution="50m", lw=0.3)
    ax4.set_global()
    # a = ax3.pcolormesh(lon,lat,np.nansum(T_NNN2018[m:n,:,:],axis=(0)),transform=ccrs.PlateCarree(),vmax=0.72,vmin=m*0.2,cmap=cmap)
    a = ax4.pcolormesh(
        lon,
        lat,
        Y_NNN2019[m, :, :],
        transform=ccrs.PlateCarree(),
        vmax=((m + 1) * 0.2) * 11.56 + 10.119,
        vmin=0,
        cmap=cmap,
    )
    gl = ax4.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False

    ax5 = plt.subplot(
        625, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax5.coastlines(resolution="50m", lw=0.3)
    ax5.set_global()
    # a = ax5.pcolormesh(lon,lat,np.nansum(T_NNN2018[m:n,:,:],axis=(0)),transform=ccrs.PlateCarree(),vmax=0.72,vmin=m*0.2,cmap=cmap)
    b = ax5.pcolormesh(
        lon,
        lat,
        T_NNN2018[m, :, :],
        transform=ccrs.PlateCarree(),
        vmax=(m + 1) * 0.2 - 0.09,
        vmin=m * 0.2 + 0.03,
        cmap=cmap,
    )
    gl = ax5.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False

    ax6 = plt.subplot(
        626, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax6.coastlines(resolution="50m", lw=0.3)
    ax6.set_global()
    # a = ax3.pcolormesh(lon,lat,np.nansum(T_NNN2018[m:n,:,:],axis=(0)),transform=ccrs.PlateCarree(),vmax=0.72,vmin=m*0.2,cmap=cmap)
    a = ax6.pcolormesh(
        lon,
        lat,
        Y_NNN2018[m, :, :],
        transform=ccrs.PlateCarree(),
        vmax=((m + 1) * 0.2) * 11.56 + 10.119,
        vmin=0,
        cmap=cmap,
    )
    gl = ax6.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False

    fig.colorbar(
        a,
        ax=[ax2, ax4, ax6],
        location="right",
        shrink=0.9,
        extend="both",
    )
    fig.colorbar(
        b,
        ax=[ax1, ax3, ax5],
        location="right",
        shrink=0.9,
        extend="both",
    )
    os.makedirs(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse/",
        exist_ok=True,
    )
    plt.savefig(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse/"
        + str(k)
        + str("%02d" % m)
        + "_"
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


plotEOFCld("cldarea", 0)

for i in range(0, 20):
    plotEOFCld("cldarea", i)


def diagnos(m, n, t1, t2):
    A_NTemp = np.zeros((3, 180, 360))
    A_NTemp[0, :, :] = np.nansum(
        A_NNN2020[m:n, t1:t2, :, :], axis=(0, 1)
    )
    A_NTemp[1, :, :] = np.nansum(
        A_NNN2019[m:n, t1:t2, :, :], axis=(0, 1)
    )
    A_NTemp[2, :, :] = np.nansum(
        A_NNN2018[m:n, t1:t2, :, :], axis=(0, 1)
    )
    A_NTemp1 = (
        np.nanmean(A_NNN2020[m:n, t1:t2, :, :], axis=(0))
        + np.nanmean(A_NNN2019[m:n, t1:t2, :, :], axis=(0))
        + np.nanmean(A_NNN2018[m:n, t1:t2, :, :], axis=(0))
    )
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax = plt.subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax.coastlines(resolution="50m", lw=0.3)
    ax.set_global()
    a = ax.pcolormesh(
        lon,
        lat,
        np.nansum(A_NTemp, axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=0.5,
        vmin=-0.5,
    )
    # a = ax.pcolormesh(lon,lat,A_NTemp1[0,:,:],transform=ccrs.PlateCarree(),cmap=cmap,vmax=0.5,vmin=-0.5)
    gl = ax.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax.set_title(str(np.nanmean(A_NTemp)), size=12)
    fig.colorbar(a, ax=ax, shrink=0.9, extend="both")
    plt.show()


diagnos(0, 16, 0, 7)


def debug(m, t):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax = plt.subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax.coastlines(resolution="50m", lw=0.3)
    ax.set_global()
    a = ax.pcolormesh(
        lon,
        lat,
        A_NNN2020[m, t, :, :]
        + A_NNN2019[m, t, :, :]
        + A_NNN2018[m, t, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=0.1,
        vmin=-0.1,
    )
    gl = ax.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(a, ax=ax, shrink=0.9, extend="both")
    plt.show()


for i in range(0, 3):
    for j in range(0, 3):
        debug(i, j)

import scipy
from scipy.signal import savgol_filter
import os

# plot by each latitude
def ploteachfiglat(arr, m, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 90, 180)

    fig = plt.figure(figsize=(24, 4))
    grid = plt.GridSpec(6, 24, wspace=0.4, hspace=0.4)

    ax1 = plt.subplot(
        grid[0:6, 0:12],
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nansum(arr[m, t1:t2, :, :], axis=(0)),
        transform=ccrs.PlateCarree(),
        vmax=14,
        vmin=-14,
        cmap=cmap,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    # ax1.set_title(str(t1)+'-'+str(t2)+' 2020 Cldtau('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)

    arr_temp = np.nan_to_num(
        np.nansum(arr[m, t1:t2, :, :], axis=(0))
    )
    A_NNN2020_temp = np.nan_to_num(
        np.nansum(A_NNN2020[m, t1:t2, :, :], axis=(0))
    )
    A_NNN2019_temp = np.nan_to_num(
        np.nansum(A_NNN2019[m, t1:t2, :, :], axis=(0))
    )
    A_NNN2018_temp = np.nan_to_num(
        np.nansum(A_NNN2018[m, t1:t2, :, :], axis=(0))
    )

    Lon = np.arange(0, 360, 1)
    ax2 = plt.subplot(
        grid[0:1, 12:24], xticklabels=[],
    )  # 和大子图共y轴
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2018_temp[150:180, :], axis=0), 21, 4
        ),
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2019_temp[150:180, :], axis=0), 21, 4
        ),
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2020_temp[150:180, :], axis=0), 21, 4
        ),
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(arr_temp[150:180, :], axis=0), 21, 4
        ),
        color="red",
        linewidth=3,
        ls="-.",
    )
    # plt.xlabel("Longitude", fontsize=14)
    ax2.set_xlim(0, 360)
    ax2.set_xticks(np.arange(0, 370, 60))
    # ax2.set_ylim(-2.5+m*0.2,2.5-m*0.2)
    # ax2.set_yticks(np.arange(-2.5+m*0.2,2.5-m*0.2,5))
    # plt.ylabel("CERES_Cldtau()", fontsize=14)
    plt.grid()

    ax3 = plt.subplot(
        grid[1:2, 12:24], xticklabels=[],
    )  # 和大子图共y轴
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2018_temp[120:150, :], axis=0), 21, 4
        ),
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2019_temp[120:150, :], axis=0), 21, 4
        ),
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2020_temp[120:150, :], axis=0), 21, 4
        ),
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(arr_temp[120:150, :], axis=0), 21, 4
        ),
        color="red",
        linewidth=3,
        ls="-.",
    )
    # plt.xlabel("Longitude", fontsize=14)
    ax3.set_xlim(0, 360)
    ax3.set_xticks(np.arange(0, 370, 60))
    # ax3.set_ylim(-2.5+m*0.2,2.5-m*0.2)
    # ax3.set_yticks(np.arange(-2.5+m*0.2,2.5-m*0.2,5))
    # plt.ylabel("CERES_Cldtau()", fontsize=14)
    plt.grid()

    ax4 = plt.subplot(
        grid[2:4, 12:24], xticklabels=[],
    )  # 和大子图共y轴
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2018_temp[60:120, :], axis=0), 21, 4
        ),
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2019_temp[60:120, :], axis=0), 21, 4
        ),
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2020_temp[60:120, :], axis=0), 21, 4
        ),
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(arr_temp[60:120, :], axis=0), 21, 4
        ),
        color="red",
        linewidth=3,
        ls="-.",
    )
    # plt.xlabel("Longitude", fontsize=14)
    ax4.set_xlim(0, 360)
    ax4.set_xticks(np.arange(0, 370, 60))
    # ax4.set_ylim(-2.5+m*0.2,2.5-m*0.2)
    # ax4.set_yticks(np.arange(-2.5+m*0.2,2.5-m*0.2,5))
    # plt.ylabel("CERES_Cldtau()", fontsize=14)
    plt.grid()

    ax5 = plt.subplot(
        grid[4:5, 12:24], xticklabels=[],
    )  # 和大子图共y轴
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2018_temp[30:60, :], axis=0), 21, 4
        ),
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2019_temp[30:60, :], axis=0), 21, 4
        ),
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2020_temp[30:60, :], axis=0), 21, 4
        ),
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(arr_temp[30:60, :], axis=0), 21, 4
        ),
        color="red",
        linewidth=3,
        ls="-.",
    )
    # plt.xlabel("Longitude", fontsize=14)
    ax5.set_xlim(0, 360)
    ax5.set_xticks(np.arange(0, 370, 60))
    # ax5.set_ylim(-2.5+m*0.2,2.5-m*0.2)
    # ax5.set_yticks(np.arange(-2.5+m*0.2,2.5-m*0.2,5))
    # plt.ylabel("CERES_Cldtau()", fontsize=14)
    plt.grid()

    ax6 = plt.subplot(
        grid[5:6, 12:24], xticklabels=[],
    )  # 和大子图共y轴
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2018_temp[0:30, :], axis=0), 21, 4
        ),
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2019_temp[0:30, :], axis=0), 21, 4
        ),
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2020_temp[0:30, :], axis=0), 21, 4
        ),
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(arr_temp[0:30, :], axis=0), 21, 4
        ),
        color="red",
        linewidth=3,
        ls="-.",
    )
    # plt.xlabel("Longitude", fontsize=14)
    ax6.set_xlim(0, 360)
    ax6.set_xticks(np.arange(0, 370, 60))
    # ax6.set_ylim(-2.5+m*0.2,2.5-m*0.2)
    # ax6.set_yticks(np.arange(-2.5+m*0.2,2.5-m*0.2,5))
    # plt.ylabel("CERES_Cldtau()", fontsize=14)
    plt.grid()

    fig.colorbar(
        a,
        ax=[ax1, ax2, ax3, ax4, ax5, ax6],
        location="left",
        shrink=0.9,
        extend="both",
    )
    os.makedirs(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse3/"
        + str(t1)
        + "-"
        + str(t2),
        exist_ok=True,
    )
    plt.savefig(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse3/"
        + str(t1)
        + "-"
        + str(t2)
        + "/"
        + "-"
        + str(m)
        + "-"
        + str(m + 1)
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


ploteachfiglat(A_NNN2020, 1, 0, 21)

for i in range(0, 20):
    ploteachfiglat(A_NNN2020, i, 0, 21)


def ploteachfig(arr, m, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    fig = plt.figure(figsize=(13, 6))
    grid = plt.GridSpec(4, 7, wspace=0.5, hspace=0.4)

    ax1 = plt.subplot(
        grid[0:3, 1:7],
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nansum(arr[m, t1:t2, :, :], axis=(0)),
        transform=ccrs.PlateCarree(),
        vmax=3,
        vmin=-3,
        cmap=cmap,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    # ax1.set_title(str(t1)+'-'+str(t2)+' 2020 Cldtau('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)

    Lon = np.arange(0, 360, 1)
    Lat = np.arange(-90, 90, 1)
    arr_temp = np.nan_to_num(
        np.nansum(arr[m, t1:t2, :, :], axis=(0))
    )
    # A_ALL = np.zeros((3,20,21,180,360))
    # A_ALL[0,:,:,:,:] = A_NNN2020
    # A_ALL[1,:,:,:,:] = A_NNN2019
    # A_ALL[2,:,:,:,:] = A_NNN2018
    # A_ALL = np.nanmean(A_ALL,axis=0)
    # A_ALL_temp = np.nan_to_num(np.nansum(A_ALL[m,t1:t2,:,:],axis=(0)))
    A_NNN2020_temp = np.nan_to_num(
        np.nansum(A_NNN2020[m, t1:t2, :, :], axis=(0))
    )
    A_NNN2019_temp = np.nan_to_num(
        np.nansum(A_NNN2019[m, t1:t2, :, :], axis=(0))
    )
    A_NNN2018_temp = np.nan_to_num(
        np.nansum(A_NNN2018[m, t1:t2, :, :], axis=(0))
    )

    ax2 = plt.subplot(grid[0:3, 0], xticklabels=[],)  # 和大子图共y轴
    plt.plot(
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2018_temp, axis=1), 51, 3
        ),
        Lat,
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2019_temp, axis=1), 51, 3
        ),
        Lat,
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2020_temp, axis=1), 51, 3
        ),
        Lat,
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(
            np.nanmean(arr_temp, axis=1), 51, 3
        ),
        Lat,
        color="red",
        linewidth=3,
        ls="-.",
    )
    # plt.xlabel("CERES_Cldtau()", fontsize=14)
    ax2.set_ylim(-90, 90)
    ax2.set_yticks(np.arange(-90, 100, 30))
    # plt.ylabel("Latitude", fontsize=14)
    plt.grid()

    ax3 = plt.subplot(grid[3, 1:7], yticklabels=[],)  # 和大子图共x轴
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2018_temp, axis=0), 51, 3
        ),
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2019_temp, axis=0), 51, 3
        ),
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(A_NNN2020_temp, axis=0), 51, 3
        ),
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(arr_temp, axis=0), 51, 3
        ),
        color="red",
        linewidth=3,
        ls="-.",
    )
    # plt.xlabel("Longitude", fontsize=14)
    ax3.set_xlim(0, 360)
    ax3.set_xticks(np.arange(0, 370, 60))
    # plt.ylabel("CERES_Cldtau()", fontsize=14)
    plt.grid()
    fig.colorbar(
        a,
        ax=[ax1, ax2, ax3],
        location="right",
        shrink=0.9,
        extend="both",
    )
    os.makedirs(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse/"
        + str(t1)
        + "-"
        + str(t2),
        exist_ok=True,
    )
    plt.savefig(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse/"
        + str(t1)
        + "-"
        + str(t2)
        + "/"
        + "-"
        + str(m)
        + "-"
        + str(m + 1)
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


ploteachfig(A_NNN2020, 12, 0, 21)

for i in range(0, 20):
    ploteachfig(A_NNN2020, i, 0, 21)

ploteachfig(A_NNN2020, 1, 0, 21)


def ploteachfig(k, m, n, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    arr = DDall[k]
    year_str = str(2020 - k).zfill(4)

    fig = plt.figure(figsize=(13, 6))
    grid = plt.GridSpec(4, 7, wspace=0.5, hspace=0.4)

    ax1 = plt.subplot(
        grid[0:3, 1:7],
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1.coastlines(resolution="50m")
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        arr,
        transform=ccrs.PlateCarree(),
        vmax=15,
        vmin=-15,
        cmap=cmap,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    # ax1.set_title(str(t1)+'-'+str(t2)+' 2020 Cldtau('+str(round(n*0.1,2))+'>EOF>='+str(round(m*0.1,2))+')',size=12)

    Lon = np.arange(0, 360, 1)
    Lat = np.arange(-90, 90, 1)
    D2013 = np.nan_to_num(np.nanmean(DD2013, axis=1))
    D2014 = np.nan_to_num(np.nanmean(DD2014, axis=1))
    D2015 = np.nan_to_num(np.nanmean(DD2015, axis=1))
    D2016 = np.nan_to_num(np.nanmean(DD2016, axis=1))
    D2017 = np.nan_to_num(np.nanmean(DD2017, axis=1))
    D2018 = np.nan_to_num(np.nanmean(DD2018, axis=1))
    D2019 = np.nan_to_num(np.nanmean(DD2019, axis=1))
    D2020 = np.nan_to_num(np.nanmean(DD2020, axis=1))
    DE = np.nan_to_num(np.nanmean(arr, axis=1))
    Dall = np.nan_to_num(np.nanmean(DDall, axis=(0, 2)))

    ax2 = plt.subplot(grid[0:3, 0], xticklabels=[],)  # 和大子图共y轴
    # fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
    plt.plot(
        scipy.signal.savgol_filter(Dall, 51, 3),
        Lat,
        color="blue",
        label="mean",
        linewidth=3,
        ls="-.",
    )

    plt.plot(
        scipy.signal.savgol_filter(D2013, 51, 3),
        Lat,
        label="2013",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(D2014, 51, 3),
        Lat,
        label="2014",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(D2015, 51, 3),
        Lat,
        label="2015",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(D2016, 51, 3),
        Lat,
        label="2016",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(D2017, 51, 3),
        Lat,
        label="2017",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(D2018, 51, 3),
        Lat,
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(D2019, 51, 3),
        Lat,
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(D2020, 51, 3),
        Lat,
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        scipy.signal.savgol_filter(DE, 51, 3),
        Lat,
        color="red",
        linewidth=3,
        ls="-.",
    )
    # plt.xlabel("CERES_Cldtau()", fontsize=14)
    ax2.set_ylim(-90, 90)
    ax2.set_yticks(np.arange(-90, 100, 30))
    # plt.ylabel("Latitude", fontsize=14)
    plt.grid()

    ax3 = plt.subplot(grid[3, 1:7], yticklabels=[],)  # 和大子图共x轴
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DDall, axis=(0, 1)), 51, 3
        ),
        color="blue",
        label="mean",
        linewidth=3,
        ls="-.",
    )

    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DD2013, axis=0), 51, 3
        ),
        label="2013",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DD2014, axis=0), 51, 3
        ),
        label="2014",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DD2015, axis=0), 51, 3
        ),
        label="2015",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DD2016, axis=0), 51, 3
        ),
        label="2016",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DD2017, axis=0), 51, 3
        ),
        label="2017",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DD2018, axis=0), 51, 3
        ),
        label="2018",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DD2019, axis=0), 51, 3
        ),
        label="2019",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(DD2020, axis=0), 51, 3
        ),
        label="2020",
        alpha=0.6,
    )
    plt.plot(
        Lon,
        scipy.signal.savgol_filter(
            np.nanmean(arr, axis=0), 51, 3
        ),
        color="red",
        linewidth=3,
        ls="-.",
    )

    # plt.xlabel("Longitude", fontsize=14)
    ax3.set_xlim(0, 360)
    ax3.set_xticks(np.arange(0, 370, 60))
    # plt.ylabel("CERES_Cldtau()", fontsize=14)
    plt.grid()
    fig.colorbar(
        a,
        ax=[ax1, ax2, ax3],
        location="right",
        shrink=0.9,
        extend="both",
    )
    os.makedirs(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/fig1/CldtauL/"
        + str(m)
        + "-"
        + str(n)
        + "_"
        + str(t1)
        + "-"
        + str(t2),
        exist_ok=True,
    )
    plt.savefig(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/fig1/CldtauL/"
        + str(m)
        + "-"
        + str(n)
        + "_"
        + str(t1)
        + "-"
        + str(t2)
        + "/"
        + year_str
        + "-"
        + str(m)
        + "-"
        + str(n)
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


ploteachfig(0, -1, 0, 0, 168)
ploteachfig(1, -1, 0, 0, 168)
ploteachfig(2, -1, 0, 0, 168)
ploteachfig(3, -1, 0, 0, 168)
ploteachfig(4, -1, 0, 0, 168)
ploteachfig(5, -1, 0, 0, 168)
ploteachfig(6, -1, 0, 0, 168)
ploteachfig(7, -1, 0, 0, 168)

fig = plt.figure(figsize=(13, 6))

ax1 = plt.subplot(
    211, projection=ccrs.PlateCarree(central_longitude=180)
)
cmap = dcmap("F://color/test.txt")
cmap.set_bad("gray")
cmap.set_over("#800000")
cmap.set_under("#191970")
ax1.coastlines(resolution="50m", lw=0.3)
ax1.set_global()
a = ax1.pcolormesh(
    lon,
    lat,
    np.nansum(A_NNN20205, axis=0)
    + np.nansum(A_NNN20195, axis=0)
    + np.nansum(A_NNN20185, axis=0),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
)
gl = ax1.gridlines(
    linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
)
gl.xlabels_top = False
gl.ylabels_left = False

ax2 = plt.subplot(
    212, projection=ccrs.PlateCarree(central_longitude=180)
)
ax2.coastlines(resolution="50m", lw=0.3)
ax2.set_global()
a = ax2.pcolormesh(
    lon,
    lat,
    A_N113[0, :, :],
    transform=ccrs.PlateCarree(),
    cmap=cmap,
)
gl = ax2.gridlines(
    linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
)
gl.xlabels_top = False
gl.ylabels_left = False


def plotbyEOFgap1(k, m, n, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    fig = plt.figure(figsize=(15, 40))
    # fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
    plt.rc("font", size=10, weight="bold")

    ax1 = plt.subplot(
        811, projection=ccrs.PlateCarree(central_longitude=180)
    )
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        DD2020,
        transform=ccrs.PlateCarree(),
        vmax=10,
        vmin=-10,
        cmap=cmap,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax1.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2020 minus mean Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax2 = plt.subplot(
        812, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    a = ax2.pcolormesh(
        lon,
        lat,
        DD2019,
        transform=ccrs.PlateCarree(),
        vmax=10,
        vmin=-10,
        cmap=cmap,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax2.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2019 minus mean Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax3 = plt.subplot(
        813, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    a = ax3.pcolormesh(
        lon,
        lat,
        DD2018,
        transform=ccrs.PlateCarree(),
        vmax=10,
        vmin=-10,
        cmap=cmap,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax3.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2018 minus mean Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax4 = plt.subplot(
        814, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax4.coastlines(resolution="50m", lw=0.3)
    ax4.set_global()
    a = ax4.pcolormesh(
        lon,
        lat,
        DD2017,
        transform=ccrs.PlateCarree(),
        vmax=10,
        vmin=-10,
        cmap=cmap,
    )
    gl = ax4.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax4.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2017 minus mean Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax5 = plt.subplot(
        815, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax5.coastlines(resolution="50m", lw=0.3)
    ax5.set_global()
    a = ax5.pcolormesh(
        lon,
        lat,
        DD2016,
        transform=ccrs.PlateCarree(),
        vmax=10,
        vmin=-10,
        cmap=cmap,
    )
    gl = ax5.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax5.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2016 minus mean Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax6 = plt.subplot(
        816, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax6.coastlines(resolution="50m", lw=0.3)
    ax6.set_global()
    a = ax6.pcolormesh(
        lon,
        lat,
        DD2015,
        transform=ccrs.PlateCarree(),
        vmax=10,
        vmin=-10,
        cmap=cmap,
    )
    gl = ax6.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax6.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2015 minus mean Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax7 = plt.subplot(
        817, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax7.coastlines(resolution="50m", lw=0.3)
    ax7.set_global()
    a = ax7.pcolormesh(
        lon,
        lat,
        DD2014,
        transform=ccrs.PlateCarree(),
        vmax=10,
        vmin=-10,
        cmap=cmap,
    )
    gl = ax7.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax7.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2014 minus mean Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax8 = plt.subplot(
        818, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax8.coastlines(resolution="50m", lw=0.3)
    ax8.set_global()
    a = ax8.pcolormesh(
        lon,
        lat,
        DD2013,
        transform=ccrs.PlateCarree(),
        vmax=10,
        vmin=-10,
        cmap=cmap,
    )
    gl = ax8.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax8.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2013 minus mean Cldtau("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    fig.colorbar(
        a,
        ax=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
        location="right",
        shrink=0.8,
        extend="both",
    )
    plt.tight_layout()
    # plt.savefig('C://Users/Administrator.YOS-94R6S19PUKV/Desktop/fig1/'+str(k)+str(m)+'-'+str(n)+'.png',dpi=300)
    plt.show()


plotbyEOFgap1("EOF-OLR", 10, 30, 0, 168)


def plotbyEOFgap1(m, n, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    fig = plt.figure(
        figsize=(10, 10), constrained_layout=True, dpi=200
    )
    # fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
    plt.rc("font", size=10, weight="bold")

    ax1 = plt.subplot(
        611, projection=ccrs.PlateCarree(central_longitude=180)
    )
    cmap = dcmap("F://color/test.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N103, axis=0),
        transform=ccrs.PlateCarree(),
        vmax=30,
        vmin=0,
        cmap=cmap,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax1.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2020 CldtauL("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax2 = plt.subplot(
        612, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    a = ax2.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N102, axis=0),
        transform=ccrs.PlateCarree(),
        vmax=30,
        vmin=0,
        cmap=cmap,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax2.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2019 CldtauL("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax3 = plt.subplot(
        613, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    a = ax3.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N101, axis=0),
        transform=ccrs.PlateCarree(),
        vmax=30,
        vmin=0,
        cmap=cmap,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax3.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2018 CldtauL("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax4 = plt.subplot(
        614, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax4.coastlines(resolution="50m", lw=0.3)
    ax4.set_global()
    a = ax4.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N100, axis=0),
        transform=ccrs.PlateCarree(),
        vmax=30,
        vmin=0,
        cmap=cmap,
    )
    gl = ax4.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax4.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2017 CldtauL("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax5 = plt.subplot(
        615, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax5.coastlines(resolution="50m", lw=0.3)
    ax5.set_global()
    a = ax5.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N90, axis=0),
        transform=ccrs.PlateCarree(),
        vmax=30,
        vmin=0,
        cmap=cmap,
    )
    gl = ax5.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax5.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2016 CldtauL("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    ax6 = plt.subplot(
        616, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax6.coastlines(resolution="50m", lw=0.3)
    ax6.set_global()
    a = ax6.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N80, axis=0),
        transform=ccrs.PlateCarree(),
        vmax=30,
        vmin=0,
        cmap=cmap,
    )
    gl = ax6.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    ax6.set_title(
        str(t1)
        + "-"
        + str(t2)
        + " 2015 CldtauL("
        + str(round(n * 0.1, 2))
        + ">EOF>="
        + str(round(m * 0.1, 2))
        + ")",
        size=12,
    )

    fig.colorbar(
        a,
        ax=[ax1, ax2, ax3, ax4, ax5, ax6],
        location="right",
        shrink=0.9,
        extend="both",
    )
    plt.show()


plotbyEOFgap1(10, 30, 0, 168)

import scipy
from scipy.signal import savgol_filter


def plotbyEOFgap1(k, m, n, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    lat1 = np.linspace(0, 69, 70)
    # lon,lat1 = np.meshgrid(lon,lat1)

    fig = plt.figure(figsize=(10, 12))
    # fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
    plt.rc("font", size=10, weight="bold")

    ax1 = plt.subplot(
        6,
        2,
        1,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    # cmap = dcmap('F://color/heibai.txt')
    # cmap.set_bad('gray')
    # cmap.set_over('white')
    # cmap.set_under('black')

    cmap = dcmap("F://color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")

    cmap1 = dcmap("F://color/b2g2r.txt")
    cmap1.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")

    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N103[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=500,
        vmin=0,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax1.set_title(" Cldarea ", size=12)

    ax2 = plt.subplot(
        6,
        2,
        2,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    b = ax2.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N113[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=2,
        vmin=-1,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax2.set_title(" Corresponding EOF ", size=12)

    ax3 = plt.subplot(
        6,
        2,
        3,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    a = ax3.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N102[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=500,
        vmin=0,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax4 = plt.subplot(
        6,
        2,
        4,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax4.coastlines(resolution="50m", lw=0.3)
    ax4.set_global()
    b = ax4.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N112[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=2,
        vmin=-1,
    )
    gl = ax4.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax5 = plt.subplot(
        6,
        2,
        5,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax5.coastlines(resolution="50m", lw=0.3)
    ax5.set_global()
    a = ax5.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N101[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=500,
        vmin=0,
    )
    gl = ax5.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax6 = plt.subplot(
        6,
        2,
        6,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax6.coastlines(resolution="50m", lw=0.3)
    ax6.set_global()
    b = ax6.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N111[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=2,
        vmin=-1,
    )
    gl = ax6.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax7 = plt.subplot(
        6,
        2,
        7,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax7.coastlines(resolution="50m", lw=0.3)
    ax7.set_global()
    a = ax7.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N100[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=500,
        vmin=0,
    )
    gl = ax7.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax8 = plt.subplot(
        6,
        2,
        8,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax8.coastlines(resolution="50m", lw=0.3)
    ax8.set_global()
    b = ax8.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N110[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=2,
        vmin=-1,
    )
    gl = ax8.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax9 = plt.subplot(
        6,
        2,
        9,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax9.coastlines(resolution="50m", lw=0.3)
    ax9.set_global()
    b = ax9.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N90[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=500,
        vmin=0,
    )
    gl = ax9.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax10 = plt.subplot(
        6,
        2,
        10,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax10.coastlines(resolution="50m", lw=0.3)
    ax10.set_global()
    b = ax10.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N19[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=2,
        vmin=-1,
    )
    gl = ax10.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax11 = plt.subplot(
        6,
        2,
        11,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax11.coastlines(resolution="50m", lw=0.3)
    ax11.set_global()
    b = ax11.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N80[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=500,
        vmin=0,
    )
    gl = ax11.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax12 = plt.subplot(
        6,
        2,
        12,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax12.coastlines(resolution="50m", lw=0.3)
    ax12.set_global()
    b = ax12.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N18[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=2,
        vmin=-1,
    )
    gl = ax12.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    fig.colorbar(
        a,
        ax=[ax1, ax3, ax5, ax7, ax9, ax11],
        location="right",
        shrink=0.9,
        extend="both",
    )
    fig.colorbar(
        b,
        ax=[ax2, ax4, ax6, ax8, ax10, ax12],
        location="right",
        shrink=0.9,
        extend="both",
    )
    # plt.savefig('C://Users/Administrator.YOS-94R6S19PUKV/Desktop/fig1/Compare/'+str(k)+'.png',dpi=300,bbox_inches='tight')
    plt.show()


plotbyEOFgap1("Cldarea", 14, 30, 0, 84)

sitaE500 = sitaE500.reshape(180, 360)
dewpoinT500 = dewpoinT500.reshape(180, 360)


def analysis(i, m, n):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("F://color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("black")
    cmap.set_under("black")
    ax1 = plt.subplot(
        411, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        A_N103[i, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=100,
        vmin=0,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(a, ax=[ax1], shrink=0.9, extend="both")
    ax1.set_title("2020 day" + str(i) + "  Cldarea ", size=13)

    ax2 = plt.subplot(
        412, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    b = ax2.pcolormesh(
        lon,
        lat,
        A_N113[i, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
    ax2.set_title(
        "2020 day" + str(i) + "  Corresponding EOF ", size=13
    )

    A_TEMP = np.array(
        np.where(
            (A_N113[i, :, :] >= (np.array(m) * 0.2))
            & (A_N113[i, :, :] < (np.array(n) * 0.2)),
            A_N103[i, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[A_TEMP == -99.9] = np.nan
    ax3 = plt.subplot(
        413, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    c = ax3.pcolormesh(
        lon,
        lat,
        A_TEMP,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=100,
        vmin=0,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(c, ax=[ax3], shrink=0.9, extend="both")
    ax3.set_title(
        "2020 day"
        + str(i)
        + "  Cldarea EOF["
        + str((m) * 0.2)
        + " , "
        + str((n) * 0.2)
        + "]",
        size=13,
    )

    A_TEMP1 = np.array(
        np.where(
            (A_N113[i, :, :] >= (np.array(m) * 0.2))
            & (A_N113[i, :, :] < (np.array(n) * 0.2)),
            A_N113[i, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[A_TEMP1 == -99.9] = np.nan
    ax4 = plt.subplot(
        414, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax4.coastlines(resolution="50m", lw=0.3)
    ax4.set_global()
    d = ax4.pcolormesh(
        lon,
        lat,
        A_TEMP1,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=np.nanmax(A_N113[i, :, :]),
        vmin=np.nanmin(A_N113[i, :, :]),
    )
    gl = ax4.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(d, ax=[ax4], shrink=0.9, extend="both")
    ax4.set_title(
        "2020 day"
        + str(i)
        + "  EOF["
        + str((m) * 0.2)
        + " , "
        + str((n) * 0.2)
        + "]",
        size=13,
    )
    os.makedirs(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse5/",
        exist_ok=True,
    )
    # os.makedirs('C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse5/'+str((m)*0.2)+'-'+str((n)*0.2),exist_ok=True)
    plt.savefig(
        "C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse5/"
        + "/day"
        + str(i)
        + "CldareaEOF"
        + str((m) * 0.2)
        + "-"
        + str((n) * 0.2)
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


analysis(0, -7.5, -5)
analysis(0, -5, -2.5)
analysis(0, -2.5, 0)
analysis(0, 0, 2.5)
analysis(0, 2.5, 5)
analysis(0, 5, 7.5)
analysis(0, 7.5, 10)
analysis(0, 10, 12.5)
analysis(0, 12.5, 15)

for i in range(0, 10):
    analysis(i, 0, 2.5)


def Meanplot(m, t):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("F://color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("black")
    cmap.set_under("black")
    ax1 = plt.subplot(
        211, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        A_NNNall[m, t, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=100,
        vmin=0,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(a, ax=[ax1], shrink=0.9, extend="both")
    ax1.set_title(
        "Week "
        + str(t)
        + "_("
        + str(np.round(np.array(m - 4.5) * 0.5, 2))
        + " =< EOF < "
        + str(np.round(np.array(m - 3.5) * 0.5, 2))
        + ") 3year mean Cldarea ",
        size=15,
    )

    ax2 = plt.subplot(
        212, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    b = ax2.pcolormesh(
        lon,
        lat,
        A_NNNall2[m, t, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
    ax2.set_title(
        "Month "
        + str(t)
        + "_("
        + str(np.round(np.array(m - 4.5) * 0.5, 2))
        + " =< EOF < "
        + str(np.round(np.array(m - 3.5) * 0.5, 2))
        + ") 3year mean Cldarea ",
        size=15,
    )
    plt.show()


Meanplot(4, 3)

lon = np.linspace(0, 359, 360)
lat = np.linspace(-90, 89, 180)
fig = plt.figure(figsize=(10, 12))
cmap = dcmap("F://color/b2g2r.txt")
cmap.set_bad("gray")
cmap.set_over("black")
cmap.set_under("black")
ax2 = plt.subplot(
    111, projection=ccrs.PlateCarree(central_longitude=180)
)
ax2.coastlines(resolution="50m", lw=0.3)
ax2.set_global()
b = ax2.pcolormesh(
    lon,
    lat,
    A_N103[0, :, :],
    transform=ccrs.PlateCarree(),
    cmap=cmap,
)
gl = ax2.gridlines(
    linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
)
gl.xlabels_top = False
gl.ylabels_left = False
fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
ax2.set_title("day" + str(i) + "  Corresponding EOF ", size=15)
plt.show()


def plotbyEOFgap1(k, m, n, t1, t2):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    lat1 = np.linspace(0, 69, 70)
    # lon,lat1 = np.meshgrid(lon,lat1)

    fig = plt.figure(figsize=(10, 12))
    # fig, axs = plt.subplots(4,1,figsize=(5,10),constrained_layout=True,dpi=200)
    plt.rc("font", size=10, weight="bold")

    ax1 = plt.subplot(
        6,
        2,
        1,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    cmap = dcmap("F://color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")

    cmap1 = dcmap("F://color/heibai.txt")
    cmap1.set_bad("gray")
    cmap1.set_over("black")
    cmap1.set_under("white")
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        A_N103[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=30,
        vmin=0,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax1.set_title(" Cldarea ", size=12)

    ax2 = plt.subplot(
        6,
        2,
        2,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    b = ax2.pcolormesh(
        lon,
        lat,
        A_N113[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=3,
        vmin=-2,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax2.set_title(" Corresponding EOF ", size=12)

    ax3 = plt.subplot(
        6,
        2,
        3,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    a = ax3.pcolormesh(
        lon,
        lat,
        A_N102[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=30,
        vmin=0,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax4 = plt.subplot(
        6,
        2,
        4,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax4.coastlines(resolution="50m", lw=0.3)
    ax4.set_global()
    b = ax4.pcolormesh(
        lon,
        lat,
        A_N112[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=3,
        vmin=-2,
    )
    gl = ax4.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax5 = plt.subplot(
        6,
        2,
        5,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax5.coastlines(resolution="50m", lw=0.3)
    ax5.set_global()
    a = ax5.pcolormesh(
        lon,
        lat,
        A_N101[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=30,
        vmin=0,
    )
    gl = ax5.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax6 = plt.subplot(
        6,
        2,
        6,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax6.coastlines(resolution="50m", lw=0.3)
    ax6.set_global()
    b = ax6.pcolormesh(
        lon,
        lat,
        A_N111[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=3,
        vmin=-2,
    )
    gl = ax6.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax7 = plt.subplot(
        6,
        2,
        7,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax7.coastlines(resolution="50m", lw=0.3)
    ax7.set_global()
    a = ax7.pcolormesh(
        lon,
        lat,
        A_N100[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=30,
        vmin=0,
    )
    gl = ax7.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax8 = plt.subplot(
        6,
        2,
        8,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax8.coastlines(resolution="50m", lw=0.3)
    ax8.set_global()
    b = ax8.pcolormesh(
        lon,
        lat,
        A_N110[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=3,
        vmin=-2,
    )
    gl = ax8.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax9 = plt.subplot(
        6,
        2,
        9,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax9.coastlines(resolution="50m", lw=0.3)
    ax9.set_global()
    b = ax9.pcolormesh(
        lon,
        lat,
        A_N90[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=30,
        vmin=0,
    )
    gl = ax9.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax10 = plt.subplot(
        6,
        2,
        10,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax10.coastlines(resolution="50m", lw=0.3)
    ax10.set_global()
    b = ax10.pcolormesh(
        lon,
        lat,
        A_N19[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=3,
        vmin=-2,
    )
    gl = ax10.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax11 = plt.subplot(
        6,
        2,
        11,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax11.coastlines(resolution="50m", lw=0.3)
    ax11.set_global()
    b = ax11.pcolormesh(
        lon,
        lat,
        A_N80[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap1,
        vmax=30,
        vmin=0,
    )
    gl = ax11.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    ax12 = plt.subplot(
        6,
        2,
        12,
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    ax12.coastlines(resolution="50m", lw=0.3)
    ax12.set_global()
    b = ax12.pcolormesh(
        lon,
        lat,
        A_N18[0, :, :],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=3,
        vmin=-2,
    )
    gl = ax12.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False

    fig.colorbar(
        a,
        ax=[ax1, ax3, ax5, ax7, ax9, ax11],
        location="right",
        shrink=0.9,
        extend="both",
    )
    fig.colorbar(
        b,
        ax=[ax2, ax4, ax6, ax8, ax10, ax12],
        location="right",
        shrink=0.9,
        extend="both",
    )
    # plt.savefig('C://Users/Administrator.YOS-94R6S19PUKV/Desktop/fig1/Compare/'+str(k)+'.png',dpi=300,bbox_inches='tight')
    plt.show()


plotbyEOFgap1("Cldarea", 14, 30, 0, 84)

################################# make ATEMP narrow gap (0.5) ########################################################################

numlist1 = [i for i in range(0, 168)]
numlist2 = [i for i in range(0, 16)]
A_TEMP = np.zeros((11, 168, 16, 180, 360))  # Cldarea
A_TEMP1 = np.zeros((11, 168, 16, 180, 360))  # EOF

for num1, num2 in product(numlist1, numlist2):
    A_TEMP[0, num1, num2, :, :] = np.array(
        np.where(
            (A_N113[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N113[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N103[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[1, num1, num2, :, :] = np.array(
        np.where(
            (A_N112[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N112[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N102[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[2, num1, num2, :, :] = np.array(
        np.where(
            (A_N111[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N111[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N101[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[3, num1, num2, :, :] = np.array(
        np.where(
            (A_N110[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N110[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N100[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[4, num1, num2, :, :] = np.array(
        np.where(
            (A_N19[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N19[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N90[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[5, num1, num2, :, :] = np.array(
        np.where(
            (A_N18[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N18[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N80[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[6, num1, num2, :, :] = np.array(
        np.where(
            (A_N17[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N17[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N70[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[7, num1, num2, :, :] = np.array(
        np.where(
            (A_N16[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N16[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N60[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[8, num1, num2, :, :] = np.array(
        np.where(
            (A_N15[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N15[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N50[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[9, num1, num2, :, :] = np.array(
        np.where(
            (A_N14[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N14[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N40[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[10, num1, num2, :, :] = np.array(
        np.where(
            (A_N13[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N13[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N30[num1, :, :].astype("float64"),
            -99.9,
        )
    )

    A_TEMP1[0, num1, num2, :, :] = np.array(
        np.where(
            (A_N113[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N113[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N113[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[1, num1, num2, :, :] = np.array(
        np.where(
            (A_N112[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N112[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N112[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[2, num1, num2, :, :] = np.array(
        np.where(
            (A_N111[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N111[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N111[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[3, num1, num2, :, :] = np.array(
        np.where(
            (A_N110[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N110[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N110[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[4, num1, num2, :, :] = np.array(
        np.where(
            (A_N19[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N19[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N19[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[5, num1, num2, :, :] = np.array(
        np.where(
            (A_N18[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N18[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N18[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[6, num1, num2, :, :] = np.array(
        np.where(
            (A_N17[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N17[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N17[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[7, num1, num2, :, :] = np.array(
        np.where(
            (A_N16[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N16[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N16[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[8, num1, num2, :, :] = np.array(
        np.where(
            (A_N15[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N15[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N15[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[9, num1, num2, :, :] = np.array(
        np.where(
            (A_N14[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N14[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N14[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[10, num1, num2, :, :] = np.array(
        np.where(
            (A_N13[num1, :, :] >= (np.array(num2 - 4.5) * 0.5))
            & (A_N13[num1, :, :] < (np.array(num2 - 3.5) * 0.5)),
            A_N13[num1, :, :].astype("float64"),
            -99.9,
        )
    )

A_TEMP[A_TEMP == -99.9] = np.nan
A_TEMP1[A_TEMP1 == -99.9] = np.nan

################################# Filt the error data ########################################################################

Box = np.zeros((119750400, 16))
for i in range(0, 16):
    Box[:, i] = A_TEMP[:, :, i, :, :].reshape(119750400)

A_P = np.zeros((11, 168, 16, 180, 360))  # Cldarea
A_P1 = np.zeros((11, 168, 16, 180, 360))  # Cldarea

for i in range(0, 16):
    A_P[:, :, i, :, :] = np.array(
        np.where(
            (
                np.nanpercentile(Box[:, i], 10)
                <= A_TEMP[:, :, i, :, :]
            )
            & (
                A_TEMP[:, :, i, :, :]
                <= np.nanpercentile(Box[:, i], 90)
            ),
            -999,
            A_TEMP[:, :, i, :, :].astype("float64"),
        )
    )
    A_P1[:, :, i, :, :] = np.array(
        np.where(
            (
                np.nanpercentile(Box[:, i], 10)
                <= A_TEMP1[:, :, i, :, :]
            )
            & (
                A_TEMP1[:, :, i, :, :]
                <= np.nanpercentile(Box[:, i], 90)
            ),
            -999,
            A_TEMP1[:, :, i, :, :].astype("float64"),
        )
    )

A_P[A_P == -999] = np.nan
A_P1[A_P1 == -999] = np.nan

###################################### save important data  ###############################################################

ds = xr.Dataset(
    {
        "A_P": (
            ("YEAR", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_P[:, :, :, :],
        ),
        "A_P1": (
            ("YEAR", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_P1[:, :, :, :],
        ),
    },
    coords={
        "lat": ("Latitude", np.linspace(-90, 89, 180)),
        "lon": ("Longitude", np.linspace(0, 359, 360)),
        "EOFGAP": ("EOFGAP", np.linspace(0, 15, 16)),
        "DAY": ("DAY", np.linspace(0, 167, 168)),
        "YEAR": ("YEAR", np.linspace(0, 10, 11)),
    },
)

# os.makedirs('F:\\PYTHONDATA\\',exist_ok=True)
ds.to_netcdf("F:\\PYTHONDATA\\2010_2020FLT_CLD" + ".nc")

###################################### Boxplot ###############################################################

Box_P = pd.DataFrame(Box)
Box_P.columns = [np.arange(-2, 5.6, 0.5)]

fig, ax = plt.subplots(figsize=(12, 9))
Box_P.boxplot(
    sym="o", whis=[10, 90], meanline=None, showmeans=True
)

fig, ax = plt.subplots(figsize=(12, 9))
Box_P.boxplot(
    sym="o", whis=[5, 95], meanline=None, showmeans=True
)

# plt.savefig('/data02/CAR_D1/Muqy/test.png')


def ploterror(i):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("F://color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("black")
    cmap.set_under("black")
    ax1 = plt.subplot(
        111, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(A_P[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=100,
        vmin=0,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(a, ax=[ax1], shrink=0.9, extend="both")
    ax1.set_title("day" + str(i) + "  Cldarea ", size=15)
    plt.savefig(
        "G://Fig/plot0/" + str(i) + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


for i in range(0, 16):
    ploterror(i)

################################# make ATEMP wide gap (1) ########################################################################

numlist1 = [i for i in range(0, 168)]
numlist2 = [i for i in range(0, 8)]
A_TEMP = np.zeros((3, 168, 8, 180, 360))  # Cldarea
A_TEMP1 = np.zeros((3, 168, 8, 180, 360))  # EOF

for num1, num2 in product(numlist1, numlist2):
    A_TEMP[0, num1, num2, :, :] = np.array(
        np.where(
            (A_N113[num1, :, :] >= (np.array(num2 - 2.5)))
            & (A_N113[num1, :, :] < (np.array(num2 - 1.5))),
            A_N103[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[1, num1, num2, :, :] = np.array(
        np.where(
            (A_N112[num1, :, :] >= (np.array(num2 - 2.5)))
            & (A_N112[num1, :, :] < (np.array(num2 - 1.5))),
            A_N102[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[2, num1, num2, :, :] = np.array(
        np.where(
            (A_N111[num1, :, :] >= (np.array(num2 - 2.5)))
            & (A_N111[num1, :, :] < (np.array(num2 - 1.5))),
            A_N101[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[0, num1, num2, :, :] = np.array(
        np.where(
            (A_N113[num1, :, :] >= (np.array(num2 - 2.5)))
            & (A_N113[num1, :, :] < (np.array(num2 - 1.5))),
            A_N113[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[1, num1, num2, :, :] = np.array(
        np.where(
            (A_N112[num1, :, :] >= (np.array(num2 - 2.5)))
            & (A_N112[num1, :, :] < (np.array(num2 - 1.5))),
            A_N112[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[2, num1, num2, :, :] = np.array(
        np.where(
            (A_N111[num1, :, :] >= (np.array(num2 - 2.5)))
            & (A_N111[num1, :, :] < (np.array(num2 - 1.5))),
            A_N111[num1, :, :].astype("float64"),
            -99.9,
        )
    )

A_TEMP[A_TEMP == -99.9] = np.nan
A_TEMP1[A_TEMP1 == -99.9] = np.nan

################################# make ATEMP wider gap (2) ########################################################################

numlist1 = [i for i in range(0, 168)]
numlist2 = [i for i in range(0, 4)]
A_TEMP = np.zeros((3, 168, 4, 180, 360))  # Cldarea
A_TEMP1 = np.zeros((3, 168, 4, 180, 360))  # EOF

for num1, num2 in product(numlist1, numlist2):
    A_TEMP[0, num1, num2, :, :] = np.array(
        np.where(
            (A_N113[num1, :, :] >= (np.array(num2 - 1.25) * 2))
            & (A_N113[num1, :, :] < (np.array(num2 - 0.25) * 2)),
            A_N103[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[1, num1, num2, :, :] = np.array(
        np.where(
            (A_N112[num1, :, :] >= (np.array(num2 - 1.25) * 2))
            & (A_N112[num1, :, :] < (np.array(num2 - 0.25) * 2)),
            A_N102[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP[2, num1, num2, :, :] = np.array(
        np.where(
            (A_N111[num1, :, :] >= (np.array(num2 - 1.25) * 2))
            & (A_N111[num1, :, :] < (np.array(num2 - 0.25) * 2)),
            A_N101[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[0, num1, num2, :, :] = np.array(
        np.where(
            (A_N113[num1, :, :] >= (np.array(num2 - 1.25) * 2))
            & (A_N113[num1, :, :] < (np.array(num2 - 0.25) * 2)),
            A_N113[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[1, num1, num2, :, :] = np.array(
        np.where(
            (A_N112[num1, :, :] >= (np.array(num2 - 1.25) * 2))
            & (A_N112[num1, :, :] < (np.array(num2 - 0.25) * 2)),
            A_N112[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1[2, num1, num2, :, :] = np.array(
        np.where(
            (A_N111[num1, :, :] >= (np.array(num2 - 1.25) * 2))
            & (A_N111[num1, :, :] < (np.array(num2 - 0.25) * 2)),
            A_N111[num1, :, :].astype("float64"),
            -99.9,
        )
    )

A_TEMP[A_TEMP == -99.9] = np.nan
A_TEMP1[A_TEMP1 == -99.9] = np.nan

################################# plot std of each Lat ########################################################################

A_TEMP_polar1 = A_TEMP[:, :, :, 150:180, :]
A_TEMP_polar2 = A_TEMP[:, :, :, 0:30, :]
A_TEMP_polar = np.concatenate(
    (A_TEMP_polar1, A_TEMP_polar2), axis=3
)
A_TEMP1_polar1 = A_TEMP1[:, :, :, 150:180, :]
A_TEMP1_polar2 = A_TEMP1[:, :, :, 0:30, :]
A_TEMP1_polar = np.concatenate(
    (A_TEMP1_polar1, A_TEMP1_polar2), axis=3
)

A_TEMP_mid1 = A_TEMP[:, :, :, 120:150, :]
A_TEMP_mid2 = A_TEMP[:, :, :, 30:60, :]
A_TEMP_mid = np.concatenate((A_TEMP_mid1, A_TEMP_mid2), axis=3)
A_TEMP1_mid1 = A_TEMP1[:, :, :, 120:150, :]
A_TEMP1_mid2 = A_TEMP1[:, :, :, 30:60, :]
A_TEMP1_mid = np.concatenate((A_TEMP1_mid1, A_TEMP1_mid2), axis=3)

A_TEMP_equator = A_TEMP[:, :, :, 60:120, :]
A_TEMP1_equator = A_TEMP1[:, :, :, 60:120, :]

# Pearson correlation coefficient
B_N0_polar = np.zeros((16))
B_N0_mid = np.zeros((16))
B_N0_equator = np.zeros((16))

E_N0 = np.zeros((16))
E_N1 = np.zeros((16))

for i in range(0, 16):
    B_N0_polar[i] = pd.Series(
        A_TEMP_polar[:, :, i, :, :].reshape(10886400)
    ).corr(
        pd.Series(A_TEMP1_polar[:, :, i, :, :].reshape(10886400)),
        method="pearson",
    )
    B_N0_mid[i] = pd.Series(
        A_TEMP_mid[:, :, i, :, :].reshape(10886400)
    ).corr(
        pd.Series(A_TEMP1_mid[:, :, i, :, :].reshape(10886400)),
        method="pearson",
    )
    B_N0_equator[i] = pd.Series(
        A_TEMP_equator[:, :, i, :, :].reshape(10886400)
    ).corr(
        pd.Series(
            A_TEMP1_equator[:, :, i, :, :].reshape(10886400)
        ),
        method="pearson",
    )

for i in range(0, 16):
    E_N0[i] = pd.Series(
        A_TEMP_polar[:, :, i, :, :].reshape(10886400)
    ).corr(
        pd.Series(A_TEMP1_polar[:, :, i, :, :].reshape(10886400)),
        method="pearson",
    )
    E_N1[i] = pd.Series(
        A_TEMP_mid[:, :, i, :, :].reshape(10886400)
    ).corr(
        pd.Series(A_TEMP1_mid[:, :, i, :, :].reshape(10886400)),
        method="pearson",
    )


for i in range(0, 16):
    E_N0[i] = np.nanstd(
        A_NNNall[i, :, :, :].reshape(1360800)
    )  # Standard deviation of Cldarea
    E_N1[i] = np.nanstd(A_NNNall2[i, :, :, :].reshape(453600))

# Std
C_N0_polar = np.zeros((16))
C_N0_mid = np.zeros((16))
C_N0_equator = np.zeros((16))
C_N01_polar = np.zeros((16))
C_N01_mid = np.zeros((16))
C_N01_equator = np.zeros((16))

for i in range(0, 16):
    C_N0_polar[i] = np.nanstd(
        A_TEMP_polar[:, :, i, :, :].reshape(10886400)
    )  # Standard deviation of Cldarea
    C_N0_mid[i] = np.nanstd(
        A_TEMP_mid[:, :, i, :, :].reshape(10886400)
    )
    C_N0_equator[i] = np.nanstd(
        A_TEMP_equator[:, :, i, :, :].reshape(10886400)
    )
    C_N01_polar[i] = np.nanstd(
        A_TEMP1_polar[:, :, i, :, :].reshape(10886400)
    )
    C_N01_mid[i] = np.nanstd(
        A_TEMP1_mid[:, :, i, :, :].reshape(10886400)
    )
    C_N01_equator[i] = np.nanstd(
        A_TEMP1_equator[:, :, i, :, :].reshape(10886400)
    )

fig, ax = plt.subplots(figsize=(12, 9))
PCA = np.arange(-2, 5.6, 0.5)
ax.plot(PCA, C_N0_polar, "blue", label="Std of Cldarea(Polar)")
ax.plot(PCA, C_N0_mid, "g", label="Std of Cldarea(Mid)")
ax.plot(PCA, C_N0_equator, "r", label="Std of Cldarea(Equator)")
ax.plot(
    PCA, C_N01_polar, "blue", label="Std of EOF(Polar)", ls="-."
)
ax.plot(PCA, C_N01_mid, "g", label="Std of EOF(Mid)", ls="-.")
ax.plot(
    PCA, C_N01_equator, "r", label="Std of EOF(Equator)", ls="-."
)
plt.legend()
plt.grid(
    axis="y", color="grey", linestyle="--", lw=0.5, alpha=0.5
)
plt.title("EOF-Std of Cldarea", fontsize=18)
ax.set_xlabel("EOF", fontsize=14)
ax.set_ylabel("Std of Cldarea", fontsize=14)
for ax in [ax]:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

fig, ax = plt.subplots(figsize=(12, 9))
PCA = np.arange(-2, 5.6, 0.5)
ax.plot(
    PCA, B_N0_polar, "blue", label="Corr of EOF-Cldarea(Polar)"
)
ax.plot(PCA, B_N0_mid, "g", label="Corr of EOF-Cldarea(Mid)")
ax.plot(
    PCA, B_N0_equator, "r", label="Corr of EOF-Cldarea(Equator)"
)
plt.legend()
plt.grid(
    axis="y", color="grey", linestyle="--", lw=0.5, alpha=0.5
)
plt.title("EOF-Corr of Cldarea", fontsize=18)
ax.set_xlabel("EOF", fontsize=14)
ax.set_ylabel("Corr of Cldarea", fontsize=14)
for ax in [ax]:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
# os.makedirs('C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse3/'+str(t1)+'-'+str(t2),exist_ok=True)
# plt.savefig('C://Users/Administrator.YOS-94R6S19PUKV/Desktop/FIG_NEW_WEEKLYMEAN/Diagnse3/'+str(t1)+'-'+str(t2)+'/'+'-'+str(m)+'-'+str(m+1)+'.png',dpi=300,bbox_inches='tight')

fig, ax = plt.subplots(figsize=(12, 9))
PCA = np.arange(-2, 5.6, 0.5)
ax.plot(PCA, E_N0, "blue", label="Weekly mean")
ax.plot(PCA, E_N1, "g", label="Monthly mean")
plt.legend()
plt.grid(
    axis="y", color="grey", linestyle="--", lw=0.5, alpha=0.5
)
plt.title("EOF-Std of Cldarea", fontsize=18)
ax.set_xlabel("EOF", fontsize=14)
ax.set_ylabel("Std of Cldarea", fontsize=14)
for ax in [ax]:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

plt.show()

###################################### test ################################################################################

A_TEMP_t = np.zeros((3, 168, 180, 360))  # Cldarea
A_TEMP1_t = np.zeros((3, 168, 180, 360))  # EOF

numlist1 = [i for i in range(0, 168)]
# numlist2 = [i for i in range(0, 4)]
for num1 in product(numlist1):
    A_TEMP_t[0, num1, :, :] = np.array(
        np.where(
            (A_N113[num1, :, :] >= (-1))
            & (A_N113[num1, :, :] < (2.5)),
            A_N103[num1, :, :].astype("float64"),
            -99.9,
        )
    )
    A_TEMP1_t[0, num1, :, :] = np.array(
        np.where(
            (A_N113[num1, :, :] >= (-1))
            & (A_N113[num1, :, :] < (2.5)),
            A_N113[num1, :, :].astype("float64"),
            -99.9,
        )
    )

A_TEMP_t[A_TEMP_t == -99.9] = np.nan
A_TEMP1_t[A_TEMP1_t == -99.9] = np.nan

A_TEMP_t_mid1 = A_TEMP_t[0, :, 120:150, :]
A_TEMP_t_mid2 = A_TEMP_t[0, :, 30:60, :]
A_TEMP_t_mid = np.concatenate(
    (A_TEMP_t_mid1, A_TEMP_t_mid2), axis=2
)
A_TEMP1_t_mid1 = A_TEMP1_t[0, :, 120:150, :]
A_TEMP1_t_mid2 = A_TEMP1_t[0, :, 30:60, :]
A_TEMP1_t_mid = np.concatenate(
    (A_TEMP1_t_mid1, A_TEMP1_t_mid2), axis=2
)

A_TEMP_t_polar1 = A_TEMP_t[0, :, 150:180, :]
A_TEMP_t_polar2 = A_TEMP_t[0, :, 0:30, :]
A_TEMP_t_polar = np.concatenate(
    (A_TEMP_t_polar1, A_TEMP_t_polar2), axis=2
)
A_TEMP1_t_polar1 = A_TEMP1_t[0, :, 150:180, :]
A_TEMP1_t_polar2 = A_TEMP1_t[0, :, 0:30, :]
A_TEMP1_t_polar = np.concatenate(
    (A_TEMP1_t_polar1, A_TEMP1_t_polar2), axis=2
)

A_TEMP_t_equator = A_TEMP_t[0, :, 60:120, :]
A_TEMP1_t_equator = A_TEMP1_t[0, :, 60:120, :]

pd.Series(A_TEMP_t_mid.reshape(3628800)).corr(
    pd.Series(A_TEMP1_t_mid.reshape(3628800)), method="pearson"
)
pd.Series(A_TEMP_t_polar.reshape(3628800)).corr(
    pd.Series(A_TEMP1_t_polar.reshape(3628800)), method="pearson"
)
pd.Series(A_TEMP_t_equator.reshape(3628800)).corr(
    pd.Series(A_TEMP1_t_equator.reshape(3628800)),
    method="pearson",
)
pd.Series(A_N113.reshape(10886400)).corr(
    pd.Series(A_N103.reshape(10886400)), method="pearson"
)
pd.Series(A_TEMP_t.reshape(32659200)).corr(
    pd.Series(A_TEMP1_t.reshape(32659200)), method="pearson"
)

np.nanstd(A_TEMP_t_mid.reshape(3628800))
np.nanstd(A_TEMP_t_polar.reshape(3628800))
np.nanstd(A_TEMP_t_equator.reshape(3628800))

np.nanstd(A_TEMP_t.reshape(32659200))

lon = np.linspace(0, 359, 360)
lat1 = np.linspace(30, 59, 30)
lat2 = np.linspace(-60, -31, 30)
fig = plt.figure(figsize=(10, 12))
cmap = dcmap("F://color/b2g2r.txt")
cmap.set_bad("gray")
cmap.set_over("black")
cmap.set_under("black")
ax1 = plt.subplot(
    111, projection=ccrs.PlateCarree(central_longitude=180)
)
ax1.coastlines(resolution="50m", lw=0.3)
ax1.set_global()
a = ax1.pcolormesh(
    lon,
    lat1,
    np.nanmean(A_TEMP_t_mid1, axis=0),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmax=100,
    vmin=0,
)
a = ax1.pcolormesh(
    lon,
    lat2,
    np.nanmean(A_TEMP_t_mid2, axis=0),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmax=100,
    vmin=0,
)
gl = ax1.gridlines(
    linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
)
gl.xlabels_top = False
gl.ylabels_left = False
fig.colorbar(a, ax=[ax1], shrink=0.9, extend="both")
ax1.set_title("day" + str(i) + "  Cldarea ", size=15)

plt.show()

#######################################################################################################################################

# Pearson correlation coefficient
B_N0 = np.zeros((16))
for i in range(0, 16):
    B_N0[i] = pd.Series(
        A_TEMP[:, :, i, :, :].reshape(32659200)
    ).corr(
        pd.Series(A_TEMP1[:, :, i, :, :].reshape(32659200)),
        method="pearson",
    )

# count the amount of data which equal to 0
A_X = A_N113.tolist()
B0 = A_X.count(0)

B_M = np.zeros((3, 16))

# count the amount of data
for num2 in product(numlist2):
    B_M[0, num2] = np.count_nonzero(
        (((np.array(num2) - 4.5) * 0.5) < A_N113)
        & (A_N113 < (((np.array(num2) - 3.5) * 0.5)))
    )
    B_M[1, num2] = np.count_nonzero(
        (((np.array(num2) - 4.5) * 0.5) < A_N112)
        & (A_N112 < (((np.array(num2) - 3.5) * 0.5)))
    )
    B_M[2, num2] = np.count_nonzero(
        (((np.array(num2) - 4.5) * 0.5) < A_N111)
        & (A_N111 < (((np.array(num2) - 3.5) * 0.5)))
    )

B_M = np.sum(B_M, axis=0)

np.count_nonzero(((-1) < A_NKK) & (A_NKK < (2.5)))
np.count_nonzero(A_NKK)

# Standard deviation
C_N0 = np.zeros((2, 16))
for i in range(0, 16):
    C_N0[0, i] = np.nanstd(
        A_TEMP[0, :, i, :, :].reshape(10886400)
    )  # Standard deviation of Cldarea
    C_N0[1, i] = np.nanstd(
        A_TEMP1[0, :, i, :, :].reshape(10886400)
    )  # Standard deviation of EOF

PCA = np.arange(-2, 5.6, 0.5)

fig, ax = plt.subplots(figsize=(12, 9))
ax.plot(PCA, C_N0[0, :], "b", label="Std of Cldarea each EOFgap")
ax.plot(PCA, C_N0[1, :], "g", label="Std of EOF each EOFgap")
plt.legend()
plt.grid(
    axis="y", color="grey", linestyle="--", lw=0.5, alpha=0.5
)
plt.title("EOF-Std of Cldarea", fontsize=18)
ax.set_xlabel("EOF", fontsize=14)
ax.set_ylabel("Std of Cldarea", fontsize=14)
for ax in [ax]:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

fig, ax1 = plt.subplots(figsize=(12, 9))
title = "Pearson coefficient/Amount of data in each EOFgap"
plt.title(title, fontsize=20)
plt.grid(
    axis="y", color="grey", linestyle="--", lw=0.5, alpha=0.5
)
plt.tick_params(axis="both", labelsize=14)
plot1 = ax1.plot(PCA, B_N0, "b", label="pearson coefficient")
ax1.set_ylabel("Corr", fontsize=18)
# ax1.set_ylim(0,240)
for tl in ax1.get_yticklabels():
    tl.set_color("b")
ax2 = ax1.twinx()
plot2 = ax2.plot(PCA, B_M, "g", label="amount of data")
ax2.set_ylabel("amount of data", fontsize=18)
# ax2.set_ylim(0,0.08)
ax2.tick_params(axis="y", labelsize=14)
for tl in ax2.get_yticklabels():
    tl.set_color("g")
# ax2.set_xlim(1966,2014.15)
lines = plot1 + plot2
ax1.legend(lines, [l.get_label() for l in lines])
ax1.set_yticks(
    np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 9)
)
ax2.set_yticks(
    np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 9)
)
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

