# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:46:59 2021

@author: Mu o(*￣▽￣*)ブ
"""

import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats import zscore
import calendar
from sklearn.decomposition import PCA
import pandas as pd
import os
import matplotlib.colors as colors
from scipy.stats import norm
import time
from itertools import product
import scipy
from scipy.signal import savgol_filter
import xarray as xr


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

data0 = xr.open_dataset("F:\\PYTHONDATA/2010_2020EOF_Cld.nc")
A_N1 = np.array(data0.EOF)

A_N1 = A_N1.reshape(119750400)
A_N1 = A_N1.reshape(66, 28, 180, 360)
A_N1e = A_N1[0:60, :, :, :]  # 2010-2019
A_NKK = A_N1[48:66, :, :, :]  # 2017-2020
A_N13 = A_N1[0:6, :, :, :]  # 2010
A_N14 = A_N1[6:12, :, :, :]  # 2011
A_N15 = A_N1[12:18, :, :, :]  # 2012
A_N16 = A_N1[18:24, :, :, :]  # 2013
A_N17 = A_N1[24:30, :, :, :]  # 2014
A_N18 = A_N1[30:36, :, :, :]  # 2015
A_N19 = A_N1[36:42, :, :, :]  # 2016
A_N110 = A_N1[42:48, :, :, :]  # 2017
A_N111 = A_N1[48:54, :, :, :]  # 2018
A_N112 = A_N1[54:60, :, :, :]  # 2019
A_N113 = A_N1[60:66, :, :, :]  # 2020

A_N1 = A_N1.reshape(119750400)
A_N1e = A_N1e.reshape(108864000)
A_N13 = A_N13.reshape(10886400)
A_N14 = A_N14.reshape(10886400)
A_N15 = A_N15.reshape(10886400)
A_N16 = A_N16.reshape(10886400)
A_N17 = A_N17.reshape(10886400)
A_N18 = A_N18.reshape(10886400)
A_N19 = A_N19.reshape(10886400)
A_N110 = A_N110.reshape(10886400)
A_N111 = A_N111.reshape(10886400)
A_N112 = A_N112.reshape(10886400)
A_N113 = A_N113.reshape(10886400)

############################### Deal CERES data ###################################

A_N20 = np.array(data0.Cldarea)
A_N21 = np.array(data0.Cldicerad)
A_N22 = np.array(data0.Cldtau)
A_N23 = np.array(data0.Cldtau_lin)
A_N24 = np.array(data0.IWP)
A_N26 = np.array(data0.Cldemissir)

A_N20t = A_N20.reshape(
    66, 28, 180, 360
)  # Choose the variable used in the plot
A_N20t[A_N20t == -999] = np.nan

A_NM = A_N20t[0:60, :, :, :]  # 2010-2019
A_NK = A_N20t[48:66, :, :, :]  # 2018-2020
A_N30 = A_N20t[0:6, :, :, :]  # 2010
A_N40 = A_N20t[6:12, :, :, :]  # 2011
A_N50 = A_N20t[12:18, :, :, :]  # 2012
A_N60 = A_N20t[18:24, :, :, :]  # 2013
A_N70 = A_N20t[24:30, :, :, :]  # 2014
A_N80 = A_N20t[30:36, :, :, :]  # 2015
A_N90 = A_N20t[36:42, :, :, :]  # 2016
A_N100 = A_N20t[42:48, :, :, :]  # 2017
A_N101 = A_N20t[48:54, :, :, :]  # 2018
A_N102 = A_N20t[54:60, :, :, :]  # 2019
A_N103 = A_N20t[60:66, :, :, :]  # 2020

A_N20t = A_N20t.reshape(119750400)
A_NM = A_NM.reshape(108864000)
A_N30 = A_N30.reshape(10886400)
A_N40 = A_N40.reshape(10886400)
A_N50 = A_N50.reshape(10886400)
A_N60 = A_N60.reshape(10886400)
A_N70 = A_N70.reshape(10886400)
A_N80 = A_N80.reshape(10886400)
A_N90 = A_N90.reshape(10886400)
A_N100 = A_N100.reshape(10886400)
A_N101 = A_N101.reshape(10886400)
A_N102 = A_N102.reshape(10886400)
A_N103 = A_N103.reshape(10886400)

########################## reshape the array ###############################################################

A_N30 = A_N30.reshape(168, 180, 360)
A_N13 = A_N13.reshape(168, 180, 360)
A_N40 = A_N40.reshape(168, 180, 360)
A_N14 = A_N14.reshape(168, 180, 360)
A_N50 = A_N50.reshape(168, 180, 360)
A_N15 = A_N15.reshape(168, 180, 360)
A_N60 = A_N60.reshape(168, 180, 360)
A_N16 = A_N16.reshape(168, 180, 360)
A_N70 = A_N70.reshape(168, 180, 360)
A_N17 = A_N17.reshape(168, 180, 360)
A_N80 = A_N80.reshape(168, 180, 360)
A_N18 = A_N18.reshape(168, 180, 360)
A_N90 = A_N90.reshape(168, 180, 360)
A_N19 = A_N19.reshape(168, 180, 360)
A_N100 = A_N100.reshape(168, 180, 360)
A_N110 = A_N110.reshape(168, 180, 360)
A_N101 = A_N101.reshape(168, 180, 360)
A_N111 = A_N111.reshape(168, 180, 360)
A_N103 = A_N103.reshape(168, 180, 360)
A_N113 = A_N113.reshape(168, 180, 360)
A_N102 = A_N102.reshape(168, 180, 360)
A_N112 = A_N112.reshape(168, 180, 360)

B_N2020 = np.zeros((180, 360))
B_N2019 = np.zeros((180, 360))
B_N2018 = np.zeros((180, 360))
B_N2017 = np.zeros((180, 360))
B_N2016 = np.zeros((180, 360))
B_N2015 = np.zeros((180, 360))
B_N2014 = np.zeros((180, 360))
B_N_all = np.zeros((180, 360))

A_N1 = A_N1.reshape(1848, 180, 360)
A_N20 = A_N20.reshape(
    1848, 180, 360
)  # Choose the variable used in the plot

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

A_P = np.zeros((11, 168, 16, 180, 360))
A_P1 = np.zeros((11, 168, 16, 180, 360))
A_Pk = np.zeros((11, 168, 16, 180, 360))
A_P1k = np.zeros((11, 168, 16, 180, 360))

for i in range(0, 16):
    A_P[:, :, i, :, :] = np.array(
        np.where(
            (
                np.nanpercentile(Box[:, i], 10)
                <= A_TEMP[:, :, i, :, :]
            ),
            -999,
            A_TEMP[:, :, i, :, :].astype("float64"),
        )
    )
    A_Pk[:, :, i, :, :] = np.array(
        np.where(
            (
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
            ),
            -999,
            A_TEMP1[:, :, i, :, :].astype("float64"),
        )
    )
    A_P1k[:, :, i, :, :] = np.array(
        np.where(
            (
                A_TEMP1[:, :, i, :, :]
                <= np.nanpercentile(Box[:, i], 90)
            ),
            -999,
            A_TEMP1[:, :, i, :, :].astype("float64"),
        )
    )

A_P[A_P == -999] = np.nan  # small error data(within 10%) for cld
A_P1[A_P1 == -999] = np.nan  # the same for eof
A_Pk[
    A_Pk == -999
] = np.nan  # large error data(out of 90%) for cld
A_P1k[A_P1k == -999] = np.nan

A_Z = np.zeros((11, 168, 16, 180, 360))
A_Z1 = np.zeros((11, 168, 16, 180, 360))

for i in range(0, 16):
    A_Z[:, :, i, :, :] = np.array(
        np.where(
            (
                np.nanpercentile(Box[:, i], 10)
                <= A_TEMP[:, :, i, :, :]
            )
            & (
                A_TEMP[:, :, i, :, :]
                <= np.nanpercentile(Box[:, i], 90)
            ),
            A_TEMP[:, :, i, :, :].astype("float64"),
            -999,
        )
    )
    A_Z1[:, :, i, :, :] = np.array(
        np.where(
            (
                np.nanpercentile(Box[:, i], 10)
                <= A_TEMP[:, :, i, :, :]
            )
            & (
                A_TEMP[:, :, i, :, :]
                <= np.nanpercentile(Box[:, i], 90)
            ),
            A_TEMP1[:, :, i, :, :].astype("float64"),
            -999,
        )
    )

A_Z[A_Z == -999] = np.nan  # cld data within(10%-90%)
A_Z1[A_Z1 == -999] = np.nan  # the same for eof

###################################### save important data  ###############################################################

ds = xr.Dataset(
    {
        "Filtered Cld": (
            ("YEAR", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_Z[:, :, :, :],
        ),
        "Filtered EOF": (
            ("YEAR", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_Z1[:, :, :, :],
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
ds.to_netcdf("/RAID01/data/muqy/PYTHONDATA/filtered_data" + ".nc")

###################################### Boxplot ###############################################################

Box = np.zeros((119750400, 16))
for i in range(0, 16):
    Box[:, i] = A_Z[:, :, i, :, :].reshape(119750400)

Box_P = pd.DataFrame(Box)
Box_P.columns = [np.arange(-2, 5.6, 0.5)]

fig, ax = plt.subplots(figsize=(12, 9))
Box_P.boxplot(
    sym="o", whis=[10, 90], meanline=None, showmeans=True
)
# plt.savefig('/RAID01/data/muqy/PYTHONFIG/boxplot'+'0.png',dpi=300,bbox_inches='tight')
plt.savefig(
    "/RAID01/data/huxy/muqy_plot/boxplot" + "0.png",
    dpi=300,
    bbox_inches="tight",
)

fig, ax = plt.subplots(figsize=(12, 9))
Box_P.boxplot(
    sym="o", whis=[5, 95], meanline=None, showmeans=True
)
# plt.savefig('/RAID01/data/muqy/PYTHONFIG/boxplot'+'1.png',dpi=300,bbox_inches='tight')
plt.savefig(
    "/RAID01/data/huxy/muqy_plot/boxplot" + "1.png",
    dpi=300,
    bbox_inches="tight",
)

######################## Complex-plot error data in each EOF gap ###############################################################

K = np.arange(-2.25, 5.8, 0.5)


def ploterror(i):
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
        np.nanmean(A_P[:, :, i, :, :], axis=(0, 1))
        - np.nanpercentile(Box[:, i], 10),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=0,
        vmin=-50,
    )
    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(a, ax=[ax1], shrink=0.9, extend="both")
    ax1.set_title(
        "Small error for EOF"
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
        np.nanmean(A_Pk[:, :, i, :, :], axis=(0, 1))
        - np.nanpercentile(Box[:, i], 90),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=50,
        vmin=0,
    )
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
    ax2.set_title(
        "Large error for EOF"
        + "("
        + str(K[i])
        + ","
        + str(K[i + 1])
        + ")"
        + "  Cldarea ",
        size=15,
    )
    # plt.savefig('/RAID01/data/muqy/PYTHONFIG/'+str(i)+'error.png',dpi=300,bbox_inches='tight')
    plt.savefig(
        "/RAID01/data/huxy/muqy_plot/" + str(i) + "error.png",
        dpi=300,
        bbox_inches="tight",
    )


for i in range(0, 16):
    ploterror(i)


def plotdata(i):
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
        np.nanmean(A_Z[:, :, i, :, :], axis=(0, 1)),
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
        "Cld for EOF"
        + "("
        + str(K[i])
        + ","
        + str(K[i + 1])
        + ")",
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
        np.nanmean(A_Z1[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-3,
    )
    # b = ax2.pcolormesh(lon,lat,np.nanmean(A_Z1[:,:,i,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),cmap=cmap,vmax=12,vmin=-7)
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
    ax2.set_title(
        "EOF" + "(" + str(K[i]) + "," + str(K[i + 1]) + ")",
        size=15,
    )
    # plt.savefig('/RAID01/data/muqy/PYTHONFIG/eachgap_allday_'+str(i)+'.png',dpi=300,bbox_inches='tight')
    plt.savefig(
        "/RAID01/data/huxy/muqy_plot/eachgap_allday_"
        + str(i).zfill(2)
        + "error.png",
        dpi=300,
        bbox_inches="tight",
    )


for i in range(0, 16):
    plotdata(i)


def plotdata1(i):
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
        np.nanmean(A_TEMP[:, :, i, :, :], axis=(0, 1)),
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
        "Cld for EOF"
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
        np.nanmean(A_TEMP1[:, :, i, :, :], axis=(0, 1)),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=6,
        vmin=-3,
    )
    # b = ax2.pcolormesh(lon,lat,np.nanmean(A_Z1[:,:,i,:,:],axis=(0,1)),transform=ccrs.PlateCarree(),cmap=cmap,vmax=12,vmin=-7)
    gl = ax2.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.xlabels_top = False
    gl.ylabels_left = False
    fig.colorbar(b, ax=[ax2], shrink=0.9, extend="both")
    ax2.set_title(
        "EOF" + "(" + str(K[i]) + "," + str(K[i + 1]) + ")",
        size=15,
    )
    # plt.savefig('/RAID01/data/muqy/PYTHONFIG/eachgap_allday1_'+str(i)+'.png',dpi=300,bbox_inches='tight')
    plt.savefig(
        "/RAID01/data/huxy/muqy_plot/eachgap_allday_"
        + str(i).zfill(2)
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )


for i in range(0, 16):
    plotdata1(i)

######################## compare the std of filtered and unfiltered data ###############################################################

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

A_Z_polar1 = A_Z[:, :, :, 150:180, :]
A_Z_polar2 = A_Z[:, :, :, 0:30, :]
A_Z_polar = np.concatenate((A_Z_polar1, A_Z_polar2), axis=3)
A_Z1_polar1 = A_Z1[:, :, :, 150:180, :]
A_Z1_polar2 = A_Z1[:, :, :, 0:30, :]
A_Z1_polar = np.concatenate((A_Z1_polar1, A_Z1_polar2), axis=3)

A_Z_mid1 = A_Z[:, :, :, 120:150, :]
A_Z_mid2 = A_Z[:, :, :, 30:60, :]
A_Z_mid = np.concatenate((A_Z_mid1, A_Z_mid2), axis=3)
A_Z1_mid1 = A_Z1[:, :, :, 120:150, :]
A_Z1_mid2 = A_Z1[:, :, :, 30:60, :]
A_Z1_mid = np.concatenate((A_Z1_mid1, A_Z1_mid2), axis=3)

A_Z_equator = A_Z[:, :, :, 60:120, :]
A_Z1_equator = A_Z1[:, :, :, 60:120, :]

# Std
C_N0_polar = np.zeros((16))
C_N0_mid = np.zeros((16))
C_N0_equator = np.zeros((16))
C_N01_polar = np.zeros((16))
C_N01_mid = np.zeros((16))
C_N01_equator = np.zeros((16))

for i in range(0, 16):
    C_N0_polar[i] = np.nanstd(
        A_Z_polar[:, :, i, :, :].reshape(39916800)
    )  # Standard deviation of Cldarea
    C_N0_mid[i] = np.nanstd(
        A_Z_mid[:, :, i, :, :].reshape(39916800)
    )
    C_N0_equator[i] = np.nanstd(
        A_Z_equator[:, :, i, :, :].reshape(39916800)
    )
    C_N01_polar[i] = np.nanstd(
        A_TEMP_polar[:, :, i, :, :].reshape(39916800)
    )
    C_N01_mid[i] = np.nanstd(
        A_TEMP_mid[:, :, i, :, :].reshape(39916800)
    )
    C_N01_equator[i] = np.nanstd(
        A_TEMP_equator[:, :, i, :, :].reshape(39916800)
    )

fig, ax = plt.subplots(figsize=(12, 9))
PCA = np.arange(-2, 5.6, 0.5)
ax.plot(
    PCA, C_N0_polar, "blue", label="Std of filtered Cld(Polar)"
)
ax.plot(PCA, C_N0_mid, "g", label="Std of filtered Cld(Mid)")
ax.plot(
    PCA, C_N0_equator, "r", label="Std of filtered Cld(Equator)"
)
ax.plot(
    PCA,
    C_N01_polar,
    "blue",
    label="Std of unfiltered Cld(Polar)",
    ls="-.",
)
ax.plot(
    PCA,
    C_N01_mid,
    "g",
    label="Std of unfiltered Cld(Mid)",
    ls="-.",
)
ax.plot(
    PCA,
    C_N01_equator,
    "r",
    label="Std of unfiltered Cld(Equator)",
    ls="-.",
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

plt.savefig(
    "/RAID01/data/muqy/PYTHONFIG/compare_filtered" + ".png",
    dpi=300,
    bbox_inches="tight",
)

######################## plot error data sum value and number account in each EOF gap ###############################################################

A_Pk_polar1 = A_Pk[:, :, :, 150:180, :]
A_Pk_polar2 = A_Pk[:, :, :, 0:30, :]
A_Pk_polar = np.concatenate((A_Pk_polar1, A_Pk_polar2), axis=3)
A_P1k_polar1 = A_P1k[:, :, :, 150:180, :]
A_P1k_polar2 = A_P1k[:, :, :, 0:30, :]
A_P1k_polar = np.concatenate((A_P1k_polar1, A_P1k_polar2), axis=3)

A_Pk_mid1 = A_Pk[:, :, :, 120:150, :]
A_Pk_mid2 = A_Pk[:, :, :, 30:60, :]
A_Pk_mid = np.concatenate((A_Pk_mid1, A_Pk_mid2), axis=3)
A_P1k_mid1 = A_P1k[:, :, :, 120:150, :]
A_P1k_mid2 = A_P1k[:, :, :, 30:60, :]
A_P1k_mid = np.concatenate((A_P1k_mid1, A_P1k_mid2), axis=3)

A_Pk_equator = A_Pk[:, :, :, 60:120, :]
A_P1k_equator = A_P1k[:, :, :, 60:120, :]

A_TEMP_polar1 = A_TEMP[:, :, :, 150:180, :]
A_TEMP_polar2 = A_TEMP[:, :, :, 0:30, :]
A_TEMP_polar = np.concatenate(
    (A_TEMP_polar1, A_TEMP_polar2), axis=3
)

A_TEMP_mid1 = A_TEMP[:, :, :, 120:150, :]
A_TEMP_mid2 = A_TEMP[:, :, :, 30:60, :]
A_TEMP_mid = np.concatenate((A_TEMP_mid1, A_TEMP_mid2), axis=3)

A_TEMP_equator = A_TEMP[:, :, :, 60:120, :]

# sum value
C_N0_polar = np.zeros((16))
C_N0_mid = np.zeros((16))
C_N0_equator = np.zeros((16))
C_N01_polar = np.zeros((16))
C_N01_mid = np.zeros((16))
C_N01_equator = np.zeros((16))
D_N0_polar = np.zeros((16))
D_N0_mid = np.zeros((16))
D_N0_equator = np.zeros((16))
E_N0_polar = np.zeros((16))
E_N0_mid = np.zeros((16))
E_N0_equator = np.zeros((16))

for i in range(0, 16):
    C_N0_polar[i] = np.nanmean(
        A_Pk_polar[:, :, i, :, :].reshape(39916800)
    ) - np.nanpercentile(Box[:, i], 90)
    C_N0_mid[i] = np.nanmean(
        A_Pk_mid[:, :, i, :, :].reshape(39916800)
    ) - np.nanpercentile(Box[:, i], 90)
    C_N0_equator[i] = np.nanmean(
        A_Pk_equator[:, :, i, :, :].reshape(39916800)
    ) - np.nanpercentile(Box[:, i], 90)
    C_N01_polar[i] = np.nanmean(
        A_P1k_polar[:, :, i, :, :].reshape(39916800)
    )
    C_N01_mid[i] = np.nanmean(
        A_P1k_mid[:, :, i, :, :].reshape(39916800)
    )
    C_N01_equator[i] = np.nanmean(
        A_P1k_equator[:, :, i, :, :].reshape(39916800)
    )
    D_N0_polar[i] = len(
        A_Pk_polar[:, :, i, :, :][
            ~np.isnan(A_Pk_polar[:, :, i, :, :])
        ]
    )
    D_N0_mid[i] = len(
        A_Pk_mid[:, :, i, :, :][
            ~np.isnan(A_Pk_mid[:, :, i, :, :])
        ]
    )
    D_N0_equator[i] = len(
        A_Pk_equator[:, :, i, :, :][
            ~np.isnan(A_Pk_equator[:, :, i, :, :])
        ]
    )


for i in range(0, 16):
    E_N0_polar[i] = len(
        A_TEMP_polar[:, :, i, :, :][
            ~np.isnan(A_TEMP_polar[:, :, i, :, :])
        ]
    )
    E_N0_mid[i] = len(
        A_TEMP_mid[:, :, i, :, :][
            ~np.isnan(A_TEMP_mid[:, :, i, :, :])
        ]
    )
    E_N0_equator[i] = len(
        A_TEMP_equator[:, :, i, :, :][
            ~np.isnan(A_TEMP_equator[:, :, i, :, :])
        ]
    )


# Plot cldarea data amount within each EOF gap
fig, ax = plt.subplots(figsize=(12, 9))
PCA = np.arange(-2, 5.6, 0.5)
ax.plot(
    PCA,
    E_N0_polar,
    "blue",
    label="Amount of data(Polar)",
    ls="-.",
)
ax.plot(PCA, E_N0_mid, "g", label="Amount of data(Mid)", ls="-.")
ax.plot(
    PCA,
    E_N0_equator,
    "r",
    label="Amount of data(Equator)",
    ls="-.",
)
plt.legend()
plt.grid(
    axis="y", color="grey", linestyle="--", lw=0.5, alpha=0.5
)
plt.title("Amount of data in each EOFgap", fontsize=18)
ax.set_xlabel("EOF", fontsize=14)
ax.set_ylabel("amount of Cldarea data", fontsize=14)
for ax in [ax]:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

plt.savefig(
    "/RAID01/data/muqy/PYTHONFIG/" + "lineplot2.png",
    dpi=300,
    bbox_inches="tight",
)

# Plot error cldarea data amount within each EOF gap and mean value
fig, ax1 = plt.subplots(figsize=(18, 9))
title = "Mean/Amount of error data in each EOFgap"
plt.title(title, fontsize=20)
plt.grid(color="grey", linestyle="--", lw=0.5, alpha=0.5)
plt.tick_params(axis="both", labelsize=14)
plot1 = ax1.plot(
    PCA, C_N0_polar, "blue", label="Mean of error Cldarea(Polar)"
)
plot2 = ax1.plot(
    PCA, C_N0_mid, "g", label="Mean of error Cldarea(Mid)"
)
plot3 = ax1.plot(
    PCA, C_N0_equator, "r", label="Mean of error Cldarea(Equator)"
)

ax1.set_ylabel("Mean", fontsize=18)
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
plot4 = ax2.plot(
    PCA,
    D_N0_polar,
    "blue",
    label="Amount of error data(Polar)",
    ls="-.",
)
plot5 = ax2.plot(
    PCA, D_N0_mid, "g", label="Amount of error data(Mid)", ls="-."
)
plot6 = ax2.plot(
    PCA,
    D_N0_equator,
    "r",
    label="Amount of error data(Equator)",
    ls="-.",
)

ax2.set_ylabel("amount of data", fontsize=18)
ax2.tick_params(axis="y", labelsize=14)
for tl in ax2.get_yticklabels():
    tl.set_color("g")

lines = plot1 + plot2 + plot3 + plot4 + plot5 + plot6
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

plt.savefig(
    "/RAID01/data/muqy/PYTHONFIG/" + "lineplot1.png",
    dpi=300,
    bbox_inches="tight",
)

# Plot error cldarea data amount within each LATITUDE and mean value
F_N0 = np.zeros((180))
G_N0 = np.zeros((180))
H_N0 = np.zeros((180))
I_N0 = np.zeros((180))

lat = np.linspace(-90, 89, 180)

for i in range(0, 180):
    F_N0[i] = len(
        A_TEMP[:, :, :, i, :][~np.isnan(A_TEMP[:, :, :, i, :])]
    )
    G_N0[i] = len(
        A_Pk[:, :, :, i, :][~np.isnan(A_Pk[:, :, :, i, :])]
    )
    H_N0[i] = np.nanmean(
        A_Pk_polar[:, :, :, i, :].reshape(39916800)
    ) - np.nanpercentile(A_Pk_polar[:, :, :, i, :], 90)
    I_N0[i] = np.nanmean(
        A_Pk_polar[:, :, :, i, :].reshape(39916800)
    ) - np.nanpercentile(A_Pk_polar[:, :, :, i, :], 90)

fig, ax = plt.subplots(figsize=(12, 9))
PCA = np.arange(-2, 5.6, 0.5)
ax.plot(lat, G_N0, "blue", label="Amount of error data", ls="-.")
# ax.plot(PCA, E_N0_mid,'g',label='Amount of data(Mid)',ls = '-.')
# ax.plot(PCA, E_N0_equator,'r',label='Amount of data(Equator)',ls = '-.')
plt.legend()
plt.grid(color="grey", linestyle="--", lw=0.5, alpha=0.5)
plt.title("Amount of data", fontsize=18)
ax.set_xlabel("Latitude", fontsize=14)
ax.set_ylabel("amount of error Cldarea data", fontsize=14)
for ax in [ax]:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

plt.savefig(
    "/RAID01/data/muqy/PYTHONFIG/" + "lineplot3.png",
    dpi=300,
    bbox_inches="tight",
)

# Plot cldarea within each EOF gap
K = np.arange(-2.25, 5.8, 0.5)


def ploterror(i):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("/RAID01/data/muqy/color/b2g2r.txt")
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
        np.nanmean(A_TEMP[:, :, i, :, :], axis=(0, 1)),
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
        " Cldarea for EOF"
        + "("
        + str(K[i])
        + ","
        + str(K[i + 1])
        + ")"
        + " ",
        size=15,
    )
    plt.savefig(
        "/RAID01/data/muqy/PYTHONFIG/" + str(i) + "DIAGNOSE.png",
        dpi=300,
        bbox_inches="tight",
    )


for i in range(0, 16):
    ploterror(i)

######################## the result of the defference ###############################################################

# CLD data
A_NM = A_Z[0:9, :, :, :, :]  # 2010-2019
A_NK = A_Z[7:10, :, :, :, :]  # 2018-2020
A_N30 = A_Z[0, :, :, :, :]  # 2010
A_N40 = A_Z[1, :, :, :, :]  # 2011
A_N50 = A_Z[2, :, :, :, :]  # 2012
A_N60 = A_Z[3, :, :, :, :]  # 2013
A_N70 = A_Z[4, :, :, :, :]  # 2014
A_N80 = A_Z[5, :, :, :, :]  # 2015
A_N90 = A_Z[6, :, :, :, :]  # 2016
A_N100 = A_Z[7, :, :, :, :]  # 2017
A_N101 = A_Z[8, :, :, :, :]  # 2018
A_N102 = A_Z[9, :, :, :, :]  # 2019
A_N103 = A_Z[10, :, :, :, :]  # 2020

# A_Z = A_Z.reshape(1916006400)
# A_NM = A_NM.reshape(1741824000)
# A_N30 = A_N30.reshape(174182400)
# A_N40 = A_N40.reshape(174182400)
# A_N50 = A_N50.reshape(174182400)
# A_N60 = A_N60.reshape(174182400)
# A_N70 = A_N70.reshape(174182400)
# A_N80 = A_N80.reshape(174182400)
# A_N90 = A_N90.reshape(174182400)
# A_N100 = A_N100.reshape(174182400)
# A_N101 = A_N101.reshape(174182400)
# A_N102 = A_N102.reshape(174182400)
# A_N103 = A_N103.reshape(174182400)

A_Z = A_Z.reshape(11, 6, 28, 16, 180, 360)
Cld_2010 = A_N30.reshape(6, 28, 16, 180, 360)
Cld_2011 = A_N40.reshape(6, 28, 16, 180, 360)
Cld_2012 = A_N50.reshape(6, 28, 16, 180, 360)
Cld_2013 = A_N60.reshape(6, 28, 16, 180, 360)
Cld_2014 = A_N70.reshape(6, 28, 16, 180, 360)
Cld_2015 = A_N80.reshape(6, 28, 16, 180, 360)
Cld_2016 = A_N90.reshape(6, 28, 16, 180, 360)
Cld_2017 = A_N100.reshape(6, 28, 16, 180, 360)
Cld_2018 = A_N101.reshape(6, 28, 16, 180, 360)
Cld_2019 = A_N102.reshape(6, 28, 16, 180, 360)
Cld_2020 = A_N103.reshape(6, 28, 16, 180, 360)

############################# Monthly mean errors version #############################

Cld_mean_filtered = np.zeros((6, 16, 180, 360))
Cld_mean_filtered = np.nanmean(A_Z, axis=(0, 2))

Cld_anormally_all = np.zeros((6, 16, 180, 360))
Cld_anormally_2020 = np.zeros((6, 16, 180, 360))
Cld_anormally_2019 = np.zeros((6, 16, 180, 360))
Cld_anormally_2018 = np.zeros((6, 16, 180, 360))
Cld_anormally_2017 = np.zeros((6, 16, 180, 360))
Cld_anormally_2016 = np.zeros((6, 16, 180, 360))
Cld_anormally_2015 = np.zeros((6, 16, 180, 360))
Cld_anormally_2014 = np.zeros((6, 16, 180, 360))
Cld_anormally_2013 = np.zeros((6, 16, 180, 360))
Cld_anormally_2012 = np.zeros((6, 16, 180, 360))
Cld_anormally_2011 = np.zeros((6, 16, 180, 360))
Cld_anormally_2010 = np.zeros((6, 16, 180, 360))

numlist1 = [i for i in range(0, 16)]
numlist2 = [i for i in range(0, 6)]
numlist3 = [i for i in range(0, 180)]
numlist4 = [i for i in range(0, 360)]

for num1, num2, num3, num4 in product(
    numlist1, numlist2, numlist3, numlist4
):
    Cld_anormally_2020[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2020[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2020[~np.isnan(Cld_2020)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2020[~np.isnan(Cld_2020)].shape))
    Cld_anormally_2019[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2019[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2019[~np.isnan(Cld_2019)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2019[~np.isnan(Cld_2019)].shape))
    Cld_anormally_2018[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2018[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2018[~np.isnan(Cld_2018)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2018[~np.isnan(Cld_2018)].shape))
    Cld_anormally_2017[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2017[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2017[~np.isnan(Cld_2017)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2017[~np.isnan(Cld_2017)].shape))
    Cld_anormally_2016[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2016[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2016[~np.isnan(Cld_2016)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2016[~np.isnan(Cld_2016)].shape))
    Cld_anormally_2015[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2015[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2015[~np.isnan(Cld_2015)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2015[~np.isnan(Cld_2015)].shape))
    Cld_anormally_2014[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2014[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2014[~np.isnan(Cld_2014)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2014[~np.isnan(Cld_2014)].shape))
    Cld_anormally_2013[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2013[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2013[~np.isnan(Cld_2013)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2013[~np.isnan(Cld_2013)].shape))
    Cld_anormally_2012[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2012[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2012[~np.isnan(Cld_2012)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2012[~np.isnan(Cld_2012)].shape))
    Cld_anormally_2011[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2011[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2011[~np.isnan(Cld_2011)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2011[~np.isnan(Cld_2011)].shape))
    Cld_anormally_2010[num2, num1, num3, num4] = (
        (
            np.nansum(Cld_2010[num2, :, num1, num3, num4], axis=0)
            - (np.array(Cld_2010[~np.isnan(Cld_2010)].shape))
            * Cld_mean_filtered[num2, num1, num3, num4]
        )
    ) / (np.array(Cld_2010[~np.isnan(Cld_2010)].shape))

A_NNNall2 = np.zeros((7, 16, 180, 360))
A_NNN20202 = np.zeros((7, 16, 180, 360))
A_NNN20192 = np.zeros((7, 16, 180, 360))
A_NNN20182 = np.zeros((7, 16, 180, 360))
A_NNN20172 = np.zeros((7, 16, 180, 360))
A_NNN20162 = np.zeros((7, 16, 180, 360))
A_NNN20152 = np.zeros((7, 16, 180, 360))
A_NNN20142 = np.zeros((7, 16, 180, 360))
A_NNN20132 = np.zeros((7, 16, 180, 360))
A_NNN20122 = np.zeros((7, 16, 180, 360))
A_NNN20112 = np.zeros((7, 16, 180, 360))
A_NNN20102 = np.zeros((7, 16, 180, 360))

numlist1 = [i for i in range(0, 16)]
numlist2 = [i for i in range(0, 7)]
numlist3 = [i for i in range(0, 180)]
numlist4 = [i for i in range(0, 360)]

A_NNNall2 = np.nanmean(A_Z, axis=(0, 2))

for num1, num2, num3, num4 in product(
    numlist1, numlist2, numlist3, numlist4
):
    A_NNN20202[num2, num1, num3, num4] = (
        np.nansum(A_N103[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N103[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20192[num2, num1, num3, num4] = (
        np.nansum(A_N102[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N102[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20182[num2, num1, num3, num4] = (
        np.nansum(A_N101[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N101[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20172[num2, num1, num3, num4] = (
        np.nansum(A_N100[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N100[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20162[num2, num1, num3, num4] = (
        np.nansum(A_N90[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N90[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20152[num2, num1, num3, num4] = (
        np.nansum(A_N80[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N80[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20142[num2, num1, num3, num4] = (
        np.nansum(A_N70[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N70[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20132[num2, num1, num3, num4] = (
        np.nansum(A_N60[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N60[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20122[num2, num1, num3, num4] = (
        np.nansum(A_N50[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N50[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20112[num2, num1, num3, num4] = (
        np.nansum(A_N40[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N40[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )
    A_NNN20102[num2, num1, num3, num4] = (
        np.nansum(A_N30[num2, :, num1, num3, num4], axis=0)
        - (np.array(A_N30[num2, :, num1, num3, num4].shape))
        * A_NNNall2[num2, num1, num3, num4]
    )


def C20182():
    for num1, num2, num3, num4 in product(
        numlist1, numlist2, numlist3, numlist4
    ):
        if (
            any(
                (
                    (np.array(num1) * 0.2)
                    <= A_N111[num2, :, num3, num4]
                )
                & (
                    A_N111[num2, :, num3, num4]
                    < ((np.array(num1) + 1) * 0.2)
                )
            )
            == True
        ):
            A_NNN20182[num1, num2, num3, num4] = (
                np.nansum(
                    A_N101[num2, :, num3, num4][
                        np.where(
                            (
                                A_N111[num2, :, num3, num4]
                                >= (np.array(num1) * 0.2)
                            )
                            & (
                                A_N111[num2, :, num3, num4]
                                < ((np.array(num1) + 1) * 0.2)
                            )
                        )
                    ]
                )
                - (
                    np.array(
                        A_N101[num2, :, num3, num4][
                            np.where(
                                (
                                    A_N111[num2, :, num3, num4]
                                    >= (np.array(num1) * 0.2)
                                )
                                & (
                                    A_N111[num2, :, num3, num4]
                                    < ((np.array(num1) + 1) * 0.2)
                                )
                            )
                        ].shape
                    )
                )
                * A_NNNall2[num1, num2, num3, num4]
            ) / A_NNNall2[num1, num2, num3, num4]
        else:
            A_NNN20182[num1, num2, num3, num4] = np.nan
    return A_NNN20182


def Aplot(i):
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
        np.nanmean(A_NNN20202[:, :, i, :, :], axis=(0, 1)),
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
        np.nanmean(A_NNN20192[:, :, i, :, :], axis=(0, 1)),
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
        np.nanmean(A_NNN20182[:, :, i, :, :], axis=(0, 1)),
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
        np.nanmean(A_NNN20172[:, :, i, :, :], axis=(0, 1)),
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
        np.nanmean(A_NNN20162[:, :, i, :, :], axis=(0, 1)),
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
        np.nanmean(A_NNN20152[:, :, i, :, :], axis=(0, 1)),
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
        np.nanmean(A_NNN20142[:, :, i, :, :], axis=(0, 1)),
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
        np.nanmean(A_NNN20132[:, :, i, :, :], axis=(0, 1)),
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
        np.nanmean(A_NNN20122[:, :, i, :, :], axis=(0, 1)),
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
        + str(i)
        + "all_result.png",
        dpi=300,
        bbox_inches="tight",
    )


for i in range(0, 16):
    Aplot(i)


def Diag(i):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(10, 12))
    cmap = dcmap("/RAID01/data/muqy/color/b2g2r.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("#191970")
    ax1 = plt.subplot(
        111, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax1.coastlines(resolution="50m", lw=0.3)
    ax1.set_global()
    b = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(A_NNN20202[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20192[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20182[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20172[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20162[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20152[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20142[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20132[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20122[:, i, :, :], axis=0)
        + np.nanmean(A_NNN20112[:, i, :, :], axis=0),
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
    fig.colorbar(b, ax=[ax1], shrink=0.9, extend="both")
    ax1.set_title("Diagnose", size=15)
    plt.savefig(
        "/RAID01/data/muqy/PYTHONFIG/" + str(i) + "diagnose.png",
        dpi=300,
        bbox_inches="tight",
    )


for i in range(0, 16):
    Diag(i)


np.nanmin(
    np.nanmean(A_NNN20202[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20192[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20182[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20172[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20162[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20152[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20142[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20132[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20122[:, i, :, :], axis=0)
    + np.nanmean(A_NNN20112[:, i, :, :], axis=0)
)

###################################### save important data  ###############################################################

ds = xr.Dataset(
    {
        "2020data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20202,
        ),
        "2019data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20192,
        ),
        "2018data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20182,
        ),
        "2018data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20182,
        ),
        "2018data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20182,
        ),
        "2018data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20182,
        ),
        "2018data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20182,
        ),
        "2018data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20182,
        ),
        "2018data": (
            ("Month", "DAY", "EOFGAP", "Latitude", "Longitude"),
            A_NNN20182,
        ),
    },
    coords={
        "lat": ("Latitude", np.linspace(-90, 89, 180)),
        "lon": ("Longitude", np.linspace(0, 359, 360)),
        "EOFGAP": ("EOFGAP", np.linspace(0, 15, 16)),
        "DAY": ("DAY", np.linspace(0, 23, 24)),
        "Month": ("Month", np.linspace(0, 6, 7)),
    },
)

# os.makedirs('F:\\PYTHONDATA\\',exist_ok=True)
ds.to_netcdf("/RAID01/data/muqy/PYTHONDATA/result_data" + ".nc")

###############################################################################################################################

data3 = xr.open_dataset("F://Cld_anormally_data.nc")
Cld_anormally_2020 = np.array(data3.Cld_anormally_2020)
Cld_anormally_2019 = np.array(data3.Cld_anormally_2019)
Cld_anormally_2018 = np.array(data3.Cld_anormally_2018)
Cld_anormally_2017 = np.array(data3.Cld_anormally_2017)
Cld_anormally_2016 = np.array(data3.Cld_anormally_2016)
Cld_anormally_2015 = np.array(data3.Cld_anormally_2015)
Cld_anormally_2014 = np.array(data3.Cld_anormally_2014)
Cld_anormally_2013 = np.array(data3.Cld_anormally_2013)
Cld_anormally_2012 = np.array(data3.Cld_anormally_2012)
Cld_anormally_2011 = np.array(data3.Cld_anormally_2011)
Cld_anormally_2010 = np.array(data3.Cld_anormally_2010)


def Alphaplot(i):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    fig = plt.figure(figsize=(6, 18))
    cmap = dcmap("F://color/test8.txt")
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
        np.nanmean(Cld_anormally_2020[:, i, :, :], axis=(0)),
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
        np.nanmean(Cld_anormally_2019[:, i, :, :], axis=(0)),
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
        np.nanmean(Cld_anormally_2018[:, i, :, :], axis=(0)),
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
        np.nanmean(Cld_anormally_2017[:, i, :, :], axis=(0)),
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
        np.nanmean(Cld_anormally_2016[:, i, :, :], axis=(0)),
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
        np.nanmean(Cld_anormally_2015[:, i, :, :], axis=(0)),
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
        np.nanmean(Cld_anormally_2014[:, i, :, :], axis=(0)),
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
        np.nanmean(Cld_anormally_2013[:, i, :, :], axis=(0)),
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
        np.nanmean(Cld_anormally_2012[:, i, :, :], axis=(0)),
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


#    plt.savefig('/RAID01/data/muqy/PYTHONFIG/'+str(i).zfill(2)+'Cld_anormally.png',dpi=300,bbox_inches='tight')

for i in range(6):
    Alphaplot(i)

##############################################################################


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


# import matplotlib
# matplotlib.use('ps')
# import matplotlib.pyplot as plt

# font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12, 'style':'italic'}
# plt.rcParams["font.family"] = ['sans-serif']


def plotbyEOFgap1():
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)
    lat1 = np.linspace(0, 69, 70)
    # lon,lat1 = np.meshgrid(lon,lat1)

    fig = plt.figure(figsize=(13, 10))
    plt.rc("font", size=10, weight="bold")

    cmap = dcmap("F://test_cld.txt")
    cmap.set_bad("gray")
    cmap.set_over("#800000")
    cmap.set_under("white")

    cmap1 = dcmap("F://test.txt")
    cmap1.set_bad("gray")
    cmap1.set_over("#800000")
    cmap1.set_under("#191970")

    ax1 = plt.subplot(
        2, 1, 1, projection=ccrs.Mollweide(central_longitude=180)
    )
    ax1.coastlines(resolution="50m", lw=0.4)
    ax1.set_global()
    a = ax1.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N20[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=40,
        vmin=0,
    )
    gl = ax1.gridlines(linestyle="-.", lw=0.1)
    gl.top_labels = False
    gl.left_labels = False
    ax1.set_title(" High Cloud Fraction (HCF) ", size=12)
    fig.colorbar(
        a,
        ax=[ax1],
        location="right",
        shrink=0.9,
        extend="both",
        label="HCF (%)",
    )

    ax2 = plt.subplot(
        2, 1, 2, projection=ccrs.Mollweide(central_longitude=180)
    )
    ax2.coastlines(resolution="50m", lw=0.4)
    ax2.set_global()
    b = ax2.pcolormesh(
        lon,
        lat,
        np.nanmean(A_N1[:, :, :], axis=0),
        transform=ccrs.PlateCarree(),
        norm=MidpointNormalize(midpoint=0),
        cmap=cmap1,
        vmax=2.5,
        vmin=-1,
    )
    gl = ax2.gridlines(linestyle="-.", lw=0.1)
    gl.top_labels = False
    gl.left_labels = False
    ax2.set_title(" Principle Component 1 (PC1) ", size=12)
    fig.colorbar(
        b,
        ax=[ax2],
        location="right",
        shrink=0.9,
        extend="both",
        label="PC1",
    )

    plt.savefig("EOF_CLDAREA.eps", dpi=800)
    # plt.tight_layout()
    # plt.show()


plotbyEOFgap1()
