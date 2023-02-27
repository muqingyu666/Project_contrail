# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:39:31 2020

@author: Mu o(*￣▽￣*)ブ

compare 2018,2019,2020 ceres high cloud
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import pandas as pd
import imageio


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


# 0 means 2020,1 means 2019,2 means 2018
data0 = xr.open_dataset(
    "G://CERES_highcloud/02-04/CERES_highcloud_202002-04.nc"
)
data1 = xr.open_dataset(
    "G://CERES_highcloud/02-04/CERES_highcloud_201902-04.nc"
)
data2 = xr.open_dataset(
    "G://CERES_highcloud/02-04/CERES_highcloud_201802-04.nc"
)
dataa = xr.open_dataset(
    "G://CERES_highcloud/CERES_highcloud_202003-04.nc"
)
datab = xr.open_dataset(
    "G://CERES_highcloud/CERES_highcloud_201903-04.nc"
)
datac = xr.open_dataset(
    "G://CERES_highcloud/CERES_highcloud_201803-04.nc"
)

cldarea0 = data0["cldarea_high_daily"]  # 2020
cldarea1 = data1["cldarea_high_daily"]  # 2019
cldarea2 = data2["cldarea_high_daily"]  # 2018
cldtau0 = data0["cldtau_high_daily"]
cldtau1 = data1["cldtau_high_daily"]
cldtau2 = data2["cldtau_high_daily"]
cldiwp0 = dataa["iwp_high_daily"]
cldiwp1 = datab["iwp_high_daily"]
cldiwp2 = datac["iwp_high_daily"]
cldicerad0 = dataa["cldicerad_high_daily"]
cldicerad1 = datab["cldicerad_high_daily"]
cldicerad2 = datac["cldicerad_high_daily"]

dates0 = pd.DatetimeIndex(np.array(cldiwp0["time"]))
dates1 = pd.DatetimeIndex(np.array(cldiwp1["time"]))
dates2 = pd.DatetimeIndex(np.array(cldiwp2["time"]))


def comparearea(n):
    lon = np.linspace(-179.5, 179.5, 360)
    lat = np.linspace(-89.5, 89.5, 180)
    lons, lats = np.meshgrid(lon, lat)

    fig, axs = plt.subplots(
        3, 1, constrained_layout=True, figsize=(5, 6), dpi=170
    )

    # Figure1, 2020 high cloud
    ax = axs[0]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
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
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    cmap = dcmap("F://color/test2.txt")
    cmap.set_under("white")
    a = map.pcolormesh(lons, lats, cldarea0[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates0.year[n])
        + "_"
        + str(dates0.month[n])
        + "_"
        + str(dates0.day[n])
        + "_"
        + "High Cloud Area",
        size=12,
    )

    # Figure2, 2019 high cloud
    ax = axs[1]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
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
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    a = map.pcolormesh(lons, lats, cldarea1[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates1.year[n])
        + "_"
        + str(dates1.month[n])
        + "_"
        + str(dates1.day[n])
        + "_"
        + "High Cloud Area",
        size=12,
    )

    # Figure3, 2018 high cloud
    ax = axs[2]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
        resolution="l",
        ax=axs[2],
    )
    parallels = np.arange(-90, 90 + 30, 30)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    a = map.pcolormesh(lons, lats, cldarea2[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates2.year[n])
        + "_"
        + str(dates2.month[n])
        + "_"
        + str(dates2.day[n])
        + "_"
        + "High Cloud Area",
        size=12,
    )

    fig.colorbar(a, ax=axs[:], location="right", shrink=0.8)
    plt.savefig(
        "G://Highcloud_comparation/"
        + str(dates2.month[n])
        + str(dates2.day[n]).zfill(2)
        + "_"
        + "High Cloud Area.png",
        dpi=170,
    )
    # plt.show()


def comparetau(n):
    lon = np.linspace(-179.5, 179.5, 360)
    lat = np.linspace(-89.5, 89.5, 180)
    lons, lats = np.meshgrid(lon, lat)

    fig, axs = plt.subplots(
        3, 1, constrained_layout=True, figsize=(5, 6), dpi=170
    )

    # Figure1, 2020 high cloud
    ax = axs[0]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
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
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    cmap = dcmap("F://color/test2.txt")
    cmap.set_bad("gray")
    a = map.pcolormesh(lons, lats, cldtau0[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates0.year[n + 1])
        + "_"
        + str(dates0.month[n + 1])
        + "_"
        + str(dates0.day[n + 1])
        + "_"
        + "High Cloud Visible Optical Depth",
        size=12,
    )

    # Figure2, 2019 high cloud
    ax = axs[1]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
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
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    a = map.pcolormesh(lons, lats, cldtau1[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates1.year[n])
        + "_"
        + str(dates1.month[n])
        + "_"
        + str(dates1.day[n])
        + "_"
        + "High Cloud Visible Optical Depth",
        size=12,
    )

    # Figure3, 2018 high cloud
    ax = axs[2]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
        resolution="l",
        ax=axs[2],
    )
    parallels = np.arange(-90, 90 + 30, 30)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    a = map.pcolormesh(lons, lats, cldtau2[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates2.year[n])
        + "_"
        + str(dates2.month[n])
        + "_"
        + str(dates2.day[n])
        + "_"
        + "High Cloud Visible Optical Depth",
        size=12,
    )

    fig.colorbar(a, ax=axs[:], location="right", shrink=0.8)
    plt.savefig(
        "G://Highcloud_comparation/"
        + str(dates2.month[n])
        + str(dates2.day[n]).zfill(2)
        + "_"
        + "High Cloud Visible Optical Depth.png",
        dpi=170,
    )
    # plt.show()


def compareiwp(n):
    lon = np.linspace(-179.5, 179.5, 360)
    lat = np.linspace(-89.5, 89.5, 180)
    lons, lats = np.meshgrid(lon, lat)

    fig, axs = plt.subplots(
        3, 1, constrained_layout=True, figsize=(5, 6), dpi=170
    )

    # Figure1, 2020 high cloud
    ax = axs[0]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
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
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    cmap = dcmap("F://color/test2.txt")
    cmap.set_bad("gray")
    a = map.pcolormesh(lons, lats, cldiwp0[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates0.year[n])
        + "_"
        + str(dates0.month[n])
        + "_"
        + str(dates0.day[n])
        + "_"
        + "High Cloud Ice Water Path",
        size=12,
    )

    # Figure2, 2019 high cloud
    ax = axs[1]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
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
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    a = map.pcolormesh(lons, lats, cldiwp1[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates1.year[n])
        + "_"
        + str(dates1.month[n])
        + "_"
        + str(dates1.day[n])
        + "_"
        + "High Cloud Ice Water Path",
        size=12,
    )

    # Figure3, 2018 high cloud
    ax = axs[2]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
        resolution="l",
        ax=axs[2],
    )
    parallels = np.arange(-90, 90 + 30, 30)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    a = map.pcolormesh(lons, lats, cldiwp2[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates2.year[n])
        + "_"
        + str(dates2.month[n])
        + "_"
        + str(dates2.day[n])
        + "_"
        + "High Cloud Ice Water Path",
        size=12,
    )

    fig.colorbar(a, ax=axs[:], location="right", shrink=0.8)
    plt.savefig(
        "G://Highcloud_comparation/cldiwp/"
        + str(dates2.month[n])
        + str(dates2.day[n]).zfill(2)
        + "_"
        + "High Cloud Ice Water Path.png",
        dpi=170,
    )
    # plt.show()


def compareicerad(n):
    lon = np.linspace(-179.5, 179.5, 360)
    lat = np.linspace(-89.5, 89.5, 180)
    lons, lats = np.meshgrid(lon, lat)

    fig, axs = plt.subplots(
        3, 1, constrained_layout=True, figsize=(5, 6), dpi=170
    )

    # Figure1, 2020 high cloud
    ax = axs[0]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
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
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    cmap = dcmap("F://color/test2.txt")
    cmap.set_bad("gray")
    a = map.pcolormesh(lons, lats, cldicerad0[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates0.year[n])
        + "_"
        + str(dates0.month[n])
        + "_"
        + str(dates0.day[n])
        + "_"
        + "High Cloud Ice Particle Radius",
        size=12,
    )

    # Figure2, 2019 high cloud
    ax = axs[1]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
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
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    a = map.pcolormesh(lons, lats, cldicerad1[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates1.year[n])
        + "_"
        + str(dates1.month[n])
        + "_"
        + str(dates1.day[n])
        + "_"
        + "High Cloud Ice Particle Radius",
        size=12,
    )

    # Figure3, 2018 high cloud
    ax = axs[2]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
        resolution="l",
        ax=axs[2],
    )
    parallels = np.arange(-90, 90 + 30, 30)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(-180, 180 + 60, 60)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    a = map.pcolormesh(lons, lats, cldicerad2[n, :, :], cmap=cmap)
    ax.set_title(
        str(dates2.year[n])
        + "_"
        + str(dates2.month[n])
        + "_"
        + str(dates2.day[n])
        + "_"
        + "High Cloud Ice Particle Radius",
        size=12,
    )

    fig.colorbar(a, ax=axs[:], location="right", shrink=0.8)
    plt.savefig(
        "G://Highcloud_comparation/cldicerad/"
        + str(dates2.month[n])
        + str(dates2.day[n]).zfill(2)
        + "_"
        + "High Cloud Ice Particle Radius.png",
        dpi=170,
    )
    # plt.show()


for n in range(0, 61):
    compareicerad(n)
