# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 09:50:35 2021

@author: Mu o(*￣▽￣*)ブ
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA


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


data = xr.open_dataset("G://highcloud_era/ERA5_201002-202002.nc")
Q = data.r
T = data.t
Q = Q[:, 17, :, :]
T = T[:, 17, :, :]
# Q = np.array(Q)
# Q2 = Q[0,:,:].reshape(1038240)
# Q2 = ZscoreNormalization(Q2)
Q1 = np.zeros((33, 181, 360))
T1 = np.zeros((33, 181, 360))
Q1_N = np.zeros((33, 180, 360))
T1_N = np.zeros((33, 180, 360))
# decline spatial resolution from 0.25-1
T1[:, :, :] = T[:, ::4, ::4]
Q1[:, :, :] = Q[:, ::4, ::4]
T1 = np.delete(T1, -1, axis=1)
Q1 = np.delete(Q1, -1, axis=1)

# flipud the data, the data stored in era5 are reversed
for i in range(0, 11):
    Q1[i, :, :] = np.flipud(Q1[i, :, :])
    T1[i, :, :] = np.flipud(T1[i, :, :])
    Q1_N[i, :, :] = ZscoreNormalization(Q1[i, :, :])
    T1_N[i, :, :] = ZscoreNormalization(T1[i, :, :])

A_N1 = np.zeros((64800, 2))
Q1_N1 = Q1_N[0, :, :].reshape(64800)
T1_N1 = T1_N[0, :, :].reshape(64800)
A_N1[:, 0] = Q1_N1
A_N1[:, 1] = T1_N1
pca = PCA(n_components=1)
pca.fit(A_N1)
A_N1 = pca.fit_transform(A_N1)
A_N1 = A_N1.reshape(180, 360)
A_N1 = ZscoreNormalization(A_N1)

lon = np.linspace(0, 359.75, 360)
lat = np.linspace(-90, 90, 181)

# fig = plt.figure(figsize=(10, 10))

fig, ax = plt.subplots(
    figsize=(10, 5), constrained_layout=True, dpi=120
)
plt.rc("font", size=10, weight="bold")

# basemap设置部分
map = Basemap(
    projection="cyl",
    llcrnrlat=-90,
    urcrnrlat=90,
    llcrnrlon=0,
    urcrnrlon=359.75,
    resolution="l",
)
parallels = np.arange(-90, 90 + 30, 30)  # 纬线
map.drawparallels(
    parallels,
    labels=[True, False, False, False],
    linewidth=0.01,
    dashes=[1, 400],
)
ax.set_yticks(parallels, len(parallels) * [""])
meridians = np.arange(0, 360 + 30, 30)  # 经线
map.drawmeridians(
    meridians,
    labels=[False, False, False, True],
    linewidth=0.01,
    dashes=[1, 400],
)
ax.set_xticks(meridians, len(meridians) * [""])
# map.drawcountries(linewidth=1.5)
map.drawcoastlines()
lon, lat = np.meshgrid(lon, lat)
cmap = dcmap("F://color/test6.txt")
cmap.set_bad("gray")
# a=map.pcolormesh(lon, lat,ek1,cmap=cmap,vmin=-0.5,vmax=16.5)
a = map.pcolormesh(lon, lat, A_N1, cmap=cmap)
fig.colorbar(a)
ax.set_title("ERA5 300hPa Relative Humidy", size=16)

# sns.kdeplot(np.linspace(-3,3,100), shade = True)
# plt.plot(np.linspace(-3,3,100),stats.norm.pdf(np.linspace(-3,3,100)))
# plt.xlim(-4,4)
plt.show()
