# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:17:13 2020

@author: Mu o(*￣▽￣*)ブ
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import glob


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


def chd(month, day, hour, minute):
    year1_str = str(2020).zfill(4)
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    time1_str = year1_str + month_str + day_str

    hour_str = str(hour).zfill(2)
    min_str = str(minute).zfill(2)
    time2_str = hour_str + min_str

    file1 = glob.glob(
        "g://Himawari-8/"
        + "NC_H08_"
        + time1_str
        + "_"
        + time2_str
        + "_R21_FLDK.06001_06001"
        + ".nc"
    )
    data1 = xr.open_dataset(file1[0])
    ch9 = data1[
        "tbb_09"
    ]  # channel 9 brightness temperature 6.9um
    ch10 = data1[
        "tbb_10"
    ]  # channel 10 brightness temperature 7.3um
    ch11 = data1[
        "tbb_11"
    ]  # channel 11 brightness temperature 8.6um
    ch14 = data1[
        "tbb_14"
    ]  # channel 14 brightness temperature 11.2um
    ch15 = data1[
        "tbb_15"
    ]  # channel 15 brightness temperature 12.4um
    ch16 = data1[
        "tbb_16"
    ]  # channel 16 brightness temperature 13.3um
    chh = ch15 - ch14  # brightness temperature difference
    # chh1 = (ch14-ch16) + (ch11-ch15) + (ch11-ch16)
    chh1 = (ch14 - ch16) + (ch11 - ch15) + (ch11 - ch16)
    chh2 = ch10 - ch9
    chh = chh[0:4001, 0:4001]
    chh1 = chh1[0:4001, 0:4001]
    chh2 = chh2[0:4001, 0:4001]
    ch9 = ch9[0:4001, 0:4001]
    ch10 = ch10[0:4001, 0:4001]
    ch15 = ch15[0:4001, 0:4001]
    ch16 = ch16[0:4001, 0:4001]
    ch14 = ch14[0:4001, 0:4001]

    lon = np.linspace(80, 160, 4001)
    lat = np.linspace(60, -20, 4001)
    lons, lats = np.meshgrid(lon, lat)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=1000)

    map = Basemap(
        projection="cyl",
        llcrnrlat=-20,
        urcrnrlat=60,
        llcrnrlon=80,
        urcrnrlon=160,
        resolution="l",
    )
    parallels = np.arange(-20, 60 + 20, 20)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(80, 160 + 20, 20)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    map.drawcountries()
    cmap = dcmap("F://color/test4.txt")
    cmap.set_under("gray")
    a = map.pcolormesh(lons, lats, chh2, cmap=cmap)
    fig.colorbar(a)
    ax.set_title(
        "Himawari-8_"
        + time1_str
        + "_"
        + time2_str
        + "_"
        + "Brightness Temperature Difference",
        size=15,
    )
    # plt.savefig('G://Himawari_fig/'+time1_str+'_'+time2_str+'_'+'Composite Brightness Temperature Difference',dpi=1000)
    plt.savefig(
        "G://Himawari_fig/"
        + time1_str
        + "_"
        + time2_str
        + "_"
        + "ch10-ch9",
        dpi=1000,
    )
    plt.show()


chd(11, 27, 11, 20)

