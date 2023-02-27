# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 10:36:12 2021

@author: Mu o(*￣▽￣*)ブ
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from matplotlib.pyplot import MultipleLocator
from symbol import *
import matplotlib.ticker
from scipy import interpolate
import pylab as pl
import math
from sklearn.metrics import mean_squared_error


K = np.arange(-2.25, 5.8, 0.5)


def test_plot(i):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("/RAID01/data/muqy/color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1 = plt.subplot(
        211, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2020[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=90,
        vmin=0,
    )
    # a = ax1.pcolormesh(lon,lat,np.nanmean(A_Z[:,:,i,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),cmap=cmap,vmax=100,vmin=0)
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(a, ax=[ax1], shrink=0.9, extend="both")
    ax1.set_title(
        "2020 Cld for EOF"
        + "("
        + str(K[i])
        + ","
        + str(K[i + 1])
        + ")"
        + "  Cldarea ",
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
        np.nanmean(Cld_2019[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=90,
        vmin=0,
    )
    # b = ax2.pcolormesh(lon,lat,np.nanmean(A_Z1[:,:,i,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),cmap=cmap,vmax=12,vmin=-7)
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
    ax2.set_title(
        "2019 Cld for EOF"
        + "("
        + str(K[i])
        + ","
        + str(K[i + 1])
        + ")"
        + "  Cldarea ",
        size=15,
    )
    # plt.savefig('/RAID01/data/muqy/PYTHONFIG/eachgap_allday1_'+str(i)+'.png',dpi=300,bbox_inches='tight')
    plt.savefig(
        "/RAID01/data/huxy/muqy_plot/test"
        + str(i).zfill(2)
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )


for i in range(0, 16):
    test_plot(i)

lon = np.linspace(0, 359, 360)
lat = np.linspace(-90, 89, 180)
fig = plt.figure(figsize=(10, 12))
# cmap = dcmap('/RAID01/data/muqy/color/b2g2r.txt')
# cmap.set_bad('gray')
# cmap.set_over('#800000')
# cmap.set_under('#191970')
ax = plt.subplot(
    111, projection=ccrs.PlateCarree(central_longitude=180)
)
ax.coastlines(resolution="50m", lw=0.3)
ax.set_global()
# a = ax1.pcolormesh(lon,lat,np.nanmean(A_Z[:,:,i,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),cmap=cmap,vmax=90,vmin=0)
for i in range(10):
    ax.plot(
        data2[2, i],
        data2[1, i],
        "o",
        transform=ccrs.PlateCarree(),
        label=str(i + 1),
    )
ax.plot(
    data2[2, 58],
    data2[1, 58],
    "*",
    transform=ccrs.PlateCarree(),
    color="red",
)
# a = ax1.pcolormesh(lon,lat,np.nanmean(A_Z[:,:,i,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),cmap=cmap,vmax=100,vmin=0)
gl = ax.gridlines(
    linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
)
gl.xlabels_top = False
gl.ylabels_left = False
plt.legend()
plt.show()

# fig.colorbar(a,ax=[ax1], shrink=0.9,extend = 'both')
# ax1.set_title('Cld for EOF'+'('+str(K[i])+','+str(K[i+1])+')',size=15)

# ax2 = plt.subplot(212,projection=ccrs.PlateCarree(central_longitude=180))
# ax2.coastlines(resolution='50m',lw=0.3)
# ax2.set_global()
# b = ax2.pcolormesh(lon,lat,np.nanmean(A_Z1[:,:,i,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),cmap=cmap,vmax=6,vmin=-3)
# # b = ax2.pcolormesh(lon,lat,np.nanmean(A_Z1[:,:,i,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),cmap=cmap,vmax=12,vmin=-7)
# gl = ax2.gridlines(linestyle='-.',lw=0.2, alpha = 0.5,draw_labels=True)
# gl.xlabels_top = False
# gl.ylabels_left = False
# fig.colorbar(b,ax=[ax2], shrink=0.9,extend = 'both')
# ax2.set_title('EOF'+'('+str(K[i])+','+str(K[i+1])+')',size=15)
# plt.savefig('/RAID01/data/muqy/PYTHONFIG/eachgap_allday_'+str(i)+'.png',dpi=300,bbox_inches='tight')
# plt.savefig('/RAID01/data/huxy/muqy_plot/eachgap_allday_'+str(i).zfill(2)+'error.png',dpi=300,bbox_inches='tight')


def Alphaplot(i):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("/RAID01/data/muqy/color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1 = plt.subplot(
        911, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2020[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax1.set_title("2020-MEAN", size=15)

    ax2 = plt.subplot(
        912, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.3)
    ax2.set_global()
    b = ax2.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2019[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax2.set_title("2019-MEAN", size=15)

    ax3 = plt.subplot(
        913, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax3.coastlines(resolution="50m", lw=0.3)
    ax3.set_global()
    b = ax3.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2018[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax3.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax3.set_title("2018-MEAN ", size=15)

    ax4 = plt.subplot(
        914, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax4.coastlines(resolution="50m", lw=0.3)
    ax4.set_global()
    b = ax4.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2017[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax4.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax4.set_title("2017-MEAN ", size=15)

    ax5 = plt.subplot(
        915, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax5.coastlines(resolution="50m", lw=0.3)
    ax5.set_global()
    b = ax5.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2016[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax5.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax5.set_title("2016-MEAN ", size=15)

    ax6 = plt.subplot(
        916, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax6.coastlines(resolution="50m", lw=0.3)
    ax6.set_global()
    b = ax6.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2015[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax6.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax6.set_title("2015-MEAN ", size=15)

    ax7 = plt.subplot(
        917, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax7.coastlines(resolution="50m", lw=0.3)
    ax7.set_global()
    b = ax7.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2014[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax7.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax7.set_title("2014-MEAN ", size=15)

    ax8 = plt.subplot(
        918, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax8.coastlines(resolution="50m", lw=0.3)
    ax8.set_global()
    b = ax8.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2013[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax8.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    ax8.set_title("2013-MEAN ", size=15)

    ax9 = plt.subplot(
        919, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax9.coastlines(resolution="50m", lw=0.3)
    ax9.set_global()
    b = ax9.pcolormesh(
        lon,
        lat,
        np.nanmean(Cld_2012[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-6,
    )
    gl = ax9.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(
        b,
        ax=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
        shrink=0.9,
        extend="both",
    )
    ax9.set_title("2012-MEAN ", size=15)
    plt.savefig(
        "/RAID01/data/muqy/PYTHONFIG/"
        + str(i).zfill(2)
        + "Cld_test.png",
        dpi=300,
        bbox_inches="tight",
    )


num1 = 0
num2 = 5
lon = np.linspace(0, 359, 360)
lat = np.linspace(-90, 89, 180)
fig = plt.figure(figsize=(10, 12))
cmap = dcmap("/RAID01/data/muqy/color/b2g2r.txt")
cmap.set_bad("gray")
cmap.set_over("#800000")
cmap.set_under("#191970")
ax1 = plt.subplot(
    311, projection=ccrs.PlateCarree(central_longitude=180)
)
ax1.coastlines(resolution="50m", lw=0.3)
ax1.set_global()
b = ax1.pcolormesh(
    lon,
    lat,
    np.nansum(Cld_2020[num1, :, num2, :, :], axis=0),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmax=150,
    vmin=0,
)
gl = ax1.gridlines(
    linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
)
gl.xlabels_top = False
gl.ylabels_left = False
ax1.set_title("1month cld", size=15)

ax2 = plt.subplot(
    312, projection=ccrs.PlateCarree(central_longitude=180)
)
ax2.coastlines(resolution="50m", lw=0.3)
ax2.set_global()
b = ax2.pcolormesh(
    lon,
    lat,
    np.nansum(Cld_2019[num1, :, num2, :, :], axis=0),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmax=150,
    vmin=0,
)
gl = ax2.gridlines(
    linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
)
gl.xlabels_top = False
gl.ylabels_left = False
ax2.set_title("1month cld", size=15)

ax3 = plt.subplot(
    313, projection=ccrs.PlateCarree(central_longitude=180)
)
ax3.coastlines(resolution="50m", lw=0.3)
ax3.set_global()
b = ax3.pcolormesh(
    lon,
    lat,
    np.nansum(Cld_2018[num1, :, num2, :, :], axis=0),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmax=150,
    vmin=0,
)
gl = ax3.gridlines(
    linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
)
gl.xlabels_top = False
gl.ylabels_left = False
ax3.set_title("1month cld", size=15)
fig.colorbar(b, ax=[ax1, ax2, ax3])
plt.savefig(
    "/RAID01/data/muqy/PYTHONFIG/test3.png",
    dpi=300,
    bbox_inches="tight",
)

