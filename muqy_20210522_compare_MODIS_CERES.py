# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:27:44 2021

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


path = "F://MOD08/"
files = os.listdir(path)
# files.sort(key = lambda x:int(x[:-6]))
# MODIS_file=glob.glob(path +files[0]+'/MOD08_D3'+'*.hdf')

###############################################################################


def MAC(year, day):

    i = year - 2010
    j = day

    MODIS_file = glob.glob(
        path + files[i] + "/MOD08_D3" + "*.hdf"
    )

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
    # cirrus = cirrus[89:170,39:180]
    # cirrus1 = cirrus[:,:].reshape(11421)
    # cirrus1[cirrus1<0] = 0

    high = np.flipud(np.array(high))
    c_1, c_2 = np.split(high, 2, axis=1)
    high = np.c_[c_2, c_1]
    # high = high[89:170,39:180]
    # high1 = high[:,:].reshape(11421)
    # high1[high1<0] = 0

    #########################################################################################

    CERES_file = glob.glob(
        "G:\\CERES_highcloud\\02-05\\CERES_highcloud_" + "*.nc"
    )

    FILE_NAME_ERA = CERES_file[i]
    # id_name = int(os.path.basename(CERES_file[i])[17:21])

    file_obj = xr.open_dataset(FILE_NAME_ERA)
    # lat = file_obj.lat
    # lon = file_obj.lon
    # t = file_obj.time
    cldarea = file_obj.cldarea_high_daily
    cldicerad = file_obj.cldicerad_high_daily
    cldtau = file_obj.cldtau_high_daily
    iwp = file_obj.iwp_high_daily
    cldpress = file_obj.cldpress_top_high_daily
    cldphase = file_obj.cldphase_high_daily
    cldemissir = file_obj.cldemissir_high_daily

    cldarea = np.array(cldarea[j, :, :])
    cldicerad = np.array(cldicerad[j, :, :])
    cldtau = np.array(cldtau[j, :, :])
    iwp = np.array(iwp[j, :, :])
    cldpress = np.array(cldpress[j, :, :])
    cldphase = np.array(cldphase[j, :, :])
    cldemissir = np.array(cldemissir[j, :, :])

    #############################################################################################

    lon = np.linspace(0, 360, 360)
    lat = np.linspace(-90, 90, 180)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    # plt.rc('font', size=10, weight='bold')

    # basemap设置部分
    ax = axs[0]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=0,
        urcrnrlon=360,
        resolution="l",
        ax=axs[0],
    )
    parallels = np.arange(-90, 90 + 30, 30)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(0, 360 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])

    map.drawcountries(linewidth=1.5)
    map.drawcoastlines()

    cmap = dcmap("F://color/test6.txt")
    cmap.set_bad("gray")
    a = map.pcolormesh(lon, lat, high, cmap=cmap, vmax=1, vmin=0)
    ax.set_title("MODIS high_cldfraction", size=16)

    # basemap设置部分
    ax = axs[1]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=0,
        urcrnrlon=360,
        resolution="l",
        ax=axs[1],
    )
    parallels = np.arange(-90, 90 + 30, 30)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(0, 360 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])

    map.drawcountries(linewidth=1.5)
    map.drawcoastlines()

    cmap = dcmap("F://color/test6.txt")
    cmap.set_bad("gray")
    a = map.pcolormesh(
        lon, lat, cldarea / 100, cmap=cmap, vmax=1, vmin=0
    )
    fig.colorbar(a, ax=axs[:], location="right", shrink=0.9)
    ax.set_title("CERES high_cldarea", size=16)

    plt.show()


MAC(2010, 1)
# MAC(2011,1)
# MAC(2012,1)
