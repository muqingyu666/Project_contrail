# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:48:07 2020

@author: Mu o(*￣▽￣*)ブ

Forward! Himawari-8 Satellite!
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
    ch14 = data1[
        "tbb_14"
    ]  # channel 14 brightness temperature 11.2um
    ch15 = data1[
        "tbb_15"
    ]  # channel 15 brightness temperature 12.4um
    ch16 = data1[
        "tbb_16"
    ]  # channel 16 brightness temperature 13.3um
    chh = ch14 - ch15  # brightness temperature difference
    # chh1 = ch14-ch16
    chh = chh[0:3001, 0:3001]
    ch15 = ch15[0:3001, 0:3001]

    ######################################################################################################################################################

    lon = np.linspace(80, 140, 3001)
    lat = np.linspace(60, 0, 3001)
    lons, lats = np.meshgrid(lon, lat)
    fig, axs = plt.subplots(
        2, 1, figsize=(10, 10), constrained_layout=True, dpi=1000
    )

    ax = axs[0]
    map = Basemap(
        projection="cyl",
        llcrnrlat=0,
        urcrnrlat=60,
        llcrnrlon=80,
        urcrnrlon=140,
        resolution="l",
        ax=axs[0],
    )
    parallels = np.arange(0, 60 + 20, 20)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(80, 140 + 20, 20)  # 经线
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
    a = map.pcolormesh(lons, lats, ch15, cmap=cmap)
    # ax.set_title('Himawari-8_'+time1_str+'_'+time2_str+'_'+'Brightness Temperature Difference',size=15)
    ax.set_title(
        "Himawari-8_"
        + time1_str
        + "_"
        + time2_str
        + "_"
        + "BT(Band15)",
        size=15,
    )

    year2_str = str(2019).zfill(4)
    month_str = str(month).zfill(2)
    day_str = str(day).zfill(2)
    time3_str = year2_str + month_str + day_str

    hour_str = str(hour).zfill(2)
    min_str = str(minute).zfill(2)
    time4_str = hour_str + min_str

    file2 = glob.glob(
        "g://Himawari-8/"
        + "NC_H08_"
        + time3_str
        + "_"
        + time4_str
        + "_R21_FLDK.06001_06001"
        + ".nc"
    )
    data2 = xr.open_dataset(file2[0])
    ch141 = data2[
        "tbb_14"
    ]  # channel 14 brightness temperature 11.2um
    ch151 = data2[
        "tbb_15"
    ]  # channel 15 brightness temperature 12.4um
    ch161 = data2[
        "tbb_16"
    ]  # channel 16 brightness temperature 13.3um
    chh1 = ch141 - ch151  # brightness temperature difference
    # chh2 = ch141-ch161
    chh1 = chh1[0:3001, 0:3001]
    ch151 = ch151[0:3001, 0:3001]

    ax = axs[1]
    map = Basemap(
        projection="cyl",
        llcrnrlat=0,
        urcrnrlat=60,
        llcrnrlon=80,
        urcrnrlon=140,
        resolution="l",
        ax=axs[1],
    )
    parallels = np.arange(0, 60 + 20, 20)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(80, 140 + 20, 20)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    map.drawcountries()
    a = map.pcolormesh(lons, lats, ch151, cmap=cmap)
    # ax.set_title('Himawari-8_'+time3_str+'_'+time4_str+'_'+'Brightness Temperature Difference',size=15)
    ax.set_title(
        "Himawari-8_"
        + time3_str
        + "_"
        + time4_str
        + "_"
        + "BT(Band15)",
        size=15,
    )

    fig.colorbar(a, ax=axs[:], location="right", shrink=0.9)

    # plt.savefig('G://Himawari_fig/'+year1_str+'_'+year2_str+'_'+time2_str+'_'+'Brightness Temperature Difference',dpi=1000)
    plt.savefig(
        "G://Himawari_fig/2020-2019_daily_Band15_compare/"
        + year1_str
        + "_"
        + year2_str
        + "_"
        + month_str
        + day_str
        + "_"
        + "BT(Band15)",
        dpi=1000,
    )
    # plt.show()


# chd(2020,3,1,1,0)
# chd(2020,3,1,1,10)
# chd(2020,3,1,1,20)
# chd(2020,3,1,1,30)
# chd(2020,3,1,1,40)
# chd(2020,3,1,1,50)
# chd(2020,3,1,2,0)
# chd(2020,3,1,2,10)
# chd(2020,3,1,2,20)
chd(3, 1, 4, 0)
chd(3, 2, 4, 0)
chd(3, 3, 4, 0)
chd(3, 4, 4, 0)
chd(3, 5, 4, 0)
chd(3, 6, 4, 0)

