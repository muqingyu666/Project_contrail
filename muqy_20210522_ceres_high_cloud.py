# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:48:06 2020

@author: Mu o(*￣▽￣*)ブ
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


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
    "I://CERES_highcloud/02-05/CERES_highcloud_20100201-20100531.nc"
)
data1 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20110201-20110531.nc"
)
data2 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20120201-20120531.nc"
)
data3 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20130201-20130531.nc"
)
data4 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20140201-20140531.nc"
)
data5 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20150201-20150531.nc"
)
data6 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20160201-20160531.nc"
)
data7 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20170201-20170531.nc"
)
data8 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20180201-20180531.nc"
)
data9 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20190201-20190531.nc"
)
data10 = xr.open_dataset(
    "I://CERES_highcloud/02-05/CERES_highcloud_20200201-20200531.nc"
)

cldarea0 = data0["cldarea_high_daily"]  # 2020
cldarea1 = data1["cldarea_high_daily"]  # 2019
cldarea2 = data2["cldarea_high_daily"]  # 2018
cldarea3 = data3["cldarea_high_daily"]  # 2017
cldarea4 = data4["cldarea_high_daily"]  # 2016
cldarea5 = data5["cldarea_high_daily"]  # 2015
cldarea6 = data6["cldarea_high_daily"]  # 2014
cldarea7 = data7["cldarea_high_daily"]  # 2013
cldarea8 = data8["cldarea_high_daily"]  # 2012
cldarea9 = data9["cldarea_high_daily"]  # 2011
cldarea10 = data10["cldarea_high_daily"]  # 2010

cldarea0 = np.array(cldarea0)
cldarea1 = np.array(cldarea1)
cldarea2 = np.array(cldarea2)
cldarea3 = np.array(cldarea3)
cldarea4 = np.array(cldarea4)
cldarea5 = np.array(cldarea5)
cldarea6 = np.array(cldarea6)
cldarea7 = np.array(cldarea7)
cldarea8 = np.array(cldarea8)
cldarea9 = np.array(cldarea9)
cldarea10 = np.array(cldarea10)

# for i in range(0,50):
#     cldarea0[i,:,:][cldarea0[i,:,:]==-999]=0
#     cldarea1[i,:,:][cldarea1[i,:,:]==-999]=0
#     cldarea2[i,:,:][cldarea2[i,:,:]==-999]=0
#     cldarea3[i,:,:][cldarea3[i,:,:]==-999]=0
#     cldarea4[i,:,:][cldarea4[i,:,:]==-999]=0
#     cldarea5[i,:,:][cldarea5[i,:,:]==-999]=0
#     cldarea6[i,:,:][cldarea6[i,:,:]==-999]=0
#     cldarea7[i,:,:][cldarea7[i,:,:]==-999]=0
#     cldarea8[i,:,:][cldarea8[i,:,:]==-999]=0
#     cldarea9[i,:,:][cldarea9[i,:,:]==-999]=0
#     cldarea10[i,:,:][cldarea10[i,:,:]==-999]=0

# for i in range(0,50):
#     cldarea0[i,:,:][cldarea0[i,:,:]!=2]=0
#     cldarea1[i,:,:][cldarea1[i,:,:]!=2]=0
#     cldarea2[i,:,:][cldarea2[i,:,:]!=2]=0
#     cldarea3[i,:,:][cldarea3[i,:,:]!=2]=0
#     cldarea4[i,:,:][cldarea4[i,:,:]!=2]=0
#     cldarea5[i,:,:][cldarea5[i,:,:]!=2]=0
#     cldarea6[i,:,:][cldarea6[i,:,:]!=2]=0
#     cldarea7[i,:,:][cldarea7[i,:,:]!=2]=0
#     cldarea8[i,:,:][cldarea8[i,:,:]!=2]=0
#     cldarea9[i,:,:][cldarea9[i,:,:]!=2]=0
#     cldarea10[i,:,:][cldarea10[i,:,:]!=2]=0

# dates0 = pd.DatetimeIndex(np.array(cldarea0['time']))
# dates1 = pd.DatetimeIndex(np.array(cldarea1['time']))
# dates2 = pd.DatetimeIndex(np.array(cldarea2['time']))

comp = np.zeros((10, 180, 360))
comp0 = np.nanmean(cldarea0[0:28, :, :], axis=0)
comp1 = np.nanmean(cldarea1[0:28, :, :], axis=0)
comp2 = np.nanmean(cldarea2[0:28, :, :], axis=0)
comp3 = np.nanmean(cldarea3[0:28, :, :], axis=0)
comp4 = np.nanmean(cldarea4[0:28, :, :], axis=0)
comp5 = np.nanmean(cldarea5[0:28, :, :], axis=0)
comp6 = np.nanmean(cldarea6[0:28, :, :], axis=0)
comp7 = np.nanmean(cldarea7[0:28, :, :], axis=0)
comp8 = np.nanmean(cldarea8[0:28, :, :], axis=0)
comp9 = np.nanmean(cldarea9[0:28, :, :], axis=0)
comp10 = np.nanmean(cldarea10[0:28, :, :], axis=0)

comp[0, :, :] = np.nanmean(cldarea0[0:28, :, :], axis=0)
comp[1, :, :] = np.nanmean(cldarea1[0:28, :, :], axis=0)
comp[2, :, :] = np.nanmean(cldarea2[0:28, :, :], axis=0)
comp[3, :, :] = np.nanmean(cldarea3[0:28, :, :], axis=0)
comp[4, :, :] = np.nanmean(cldarea4[0:28, :, :], axis=0)
comp[5, :, :] = np.nanmean(cldarea5[0:28, :, :], axis=0)
comp[6, :, :] = np.nanmean(cldarea6[0:28, :, :], axis=0)
comp[7, :, :] = np.nanmean(cldarea7[0:28, :, :], axis=0)
comp[8, :, :] = np.nanmean(cldarea8[0:28, :, :], axis=0)
comp[9, :, :] = np.nanmean(cldarea9[0:28, :, :], axis=0)
# comp[10,:,:] = np.nanmean(cldarea10[0:28,:,:], axis=0)
comp = np.nanmean(comp, axis=0)
D = comp10 - comp
################################################################


def comparearea():
    lon = np.linspace(-179.5, 179.5, 360)
    lat = np.linspace(-89.5, 89.5, 180)
    lons, lats = np.meshgrid(lon, lat)

    fig, axs = plt.subplots(
        2, 1, constrained_layout=True, figsize=(5, 6), dpi=170
    )

    # Figure1, 2020 high cloud
    ax = axs[0]
    map = Basemap(
        projection="cyl",
        llcrnrlat=-20,
        urcrnrlat=80,
        llcrnrlon=40,
        urcrnrlon=180,
        resolution="l",
        ax=axs[0],
    )
    parallels = np.arange(-20, 80 + 20, 20)  # 纬线
    map.drawparallels(
        parallels,
        labels=[True, False, False, False],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_yticks(parallels, len(parallels) * [""])
    meridians = np.arange(40, 180 + 20, 20)  # 经线
    map.drawmeridians(
        meridians,
        labels=[False, False, False, True],
        linewidth=0.01,
        dashes=[1, 400],
    )
    ax.set_xticks(meridians, len(meridians) * [""])
    map.drawcoastlines(linewidth=0.5)
    map.drawcountries(linewidth=0.5)
    cmap = dcmap("F://color/test2.txt")
    cmap.set_under("gray")
    a = map.pcolormesh(lons, lats, D, cmap=cmap)
    # ax.set_title(str(dates0.year[n])+'_'+str(dates0.month[n])+'_'+str(dates0.day[n])+'_'+'High Cloud Area',size=12)
    ax.set_title(
        "2020_02 cldarae minus 2010_02-2019_02 cldarae", size=10
    )

    # Figure2, 2019 high cloud
    # ax = axs[1]
    # map = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=80,llcrnrlon=40,urcrnrlon=180,resolution='l',ax=axs[1])
    # parallels = np.arange(-20,80+20,20) #纬线
    # map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
    # ax.set_yticks(parallels,len(parallels)*[''])
    # meridians = np.arange(40,180+20,20) #经线
    # map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
    # ax.set_xticks(meridians,len(meridians)*[''])
    # map.drawcoastlines(linewidth=0.5)
    # map.drawcountries(linewidth=0.5)
    # a=map.pcolormesh(lons,lats,comp,cmap=cmap)
    # # ax.set_title(str(dates1.year[n])+'_'+str(dates1.month[n])+'_'+str(dates1.day[n])+'_'+'High Cloud Area',size=12)
    # ax.set_title('2010_02-2019_02 Average High Cloud Ice Water Path',size=10)

    # Figure3, 2018 high cloud
    # ax = axs[2]
    # map = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=80,llcrnrlon=40,urcrnrlon=180,resolution='l',ax=axs[2])
    # parallels = np.arange(-20,80+20,20) #纬线
    # map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
    # ax.set_yticks(parallels,len(parallels)*[''])
    # meridians = np.arange(40,180+20,20) #经线
    # map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
    # ax.set_xticks(meridians,len(meridians)*[''])
    # map.drawcoastlines(linewidth=0.5)
    # a=map.pcolormesh(lons,lats,comp2,cmap=cmap)
    # # ax.set_title(str(dates2.year[n])+'_'+str(dates2.month[n])+'_'+str(dates2.day[n])+'_'+'High Cloud Area',size=12)
    # ax.set_title('201802 High Cloud Area',size=12)

    fig.colorbar(a, ax=axs[:], location="right", shrink=0.8)
    # plt.savefig('G://Highcloud_comparation/'+str(dates2.month[n])+str(dates2.day[n]).zfill(2)+'_'+'High Cloud Area.png',dpi=170)
    plt.show()


comparearea()
