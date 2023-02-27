#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#                       神兽保佑
#                      代码无BUG!

"""

    Code for PCA-HCF analyze
    
    Owner: Mu Qingyu
    version 2.0
        changes from v1.0-v2.0: remove functions to util_PCA_CLD_analysis_function
            to save some space
            
    Created: 2022-04-13
    
    Including the following parts:
        
        1) Read the Pre-calculated PC1 and HCF data 
        
        2) Filter the data to fit the PC1 gap like -1.5 ~ 3.5
        
        3) Plot boxplot to show the distribution of HCF data
        
        4) Compare HCF of different years in the same PC1 condition
                
        5) All sort of test code
        
"""

# ------------ PCA analysis ------------
# ------------ Start import ------------
import calendar
import glob
import os
import time
import tkinter
from itertools import product

import cartopy.crs as ccrs
import matplotlib
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from metpy.units import units
from numba import jit
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import norm, zscore
from sklearn.decomposition import PCA

# ----------  importing dcmap from my util ----------#
from muqy_20220413_util_useful_functions import dcmap as dcmap

# ----------  done importing  ----------#


def read_PC1_CERES_from_netcdf():
    # Read data from netcdf file
    print("Reading data from netcdf file...")
    data0 = xr.open_dataset(
        "/RAID01/data/All_data/2010_2020_PC1_and_CLD.nc"
    )
    data1 = xr.open_dataset(
        "/RAID01/data/PCA_data/2010_2020_5_parameters_300hPa_PC1.nc"
    )
    print("Done loading netcdf file.")

    PC_all = np.array(data1.PC1)

    # Arrange data from all years
    PC_all = PC_all.reshape(239500800)
    PC_all = PC_all.reshape(132, 28, 180, 360)

    # --------- Original data ---------#
    # region
    PC_2010_2019 = PC_all[0:120, :, :, :]  # 2010-2019
    PC_2017_2020 = PC_all[84:132, :, :, :]  # 2017-2020
    PC_2010 = PC_all[0:12, :, :, :]  # 2010
    PC_2011 = PC_all[12:24, :, :, :]  # 2011
    PC_2012 = PC_all[24:36, :, :, :]  # 2012
    PC_2013 = PC_all[36:48, :, :, :]  # 2013
    PC_2014 = PC_all[48:60, :, :, :]  # 2014
    PC_2015 = PC_all[60:72, :, :, :]  # 2015
    PC_2016 = PC_all[72:84, :, :, :]  # 2016
    PC_2017 = PC_all[84:96, :, :, :]  # 2017
    PC_2018 = PC_all[96:108, :, :, :]  # 2018
    PC_2019 = PC_all[108:120, :, :, :]  # 2019
    PC_2020 = PC_all[120:132, :, :, :]  # 2020
    # endregion

    # --------- Get 4-6 month data of each years ---------#
    PC_2010_2019_4_month = PC_2010_2019[3::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    PC_2010_2019_5_month = PC_2010_2019[4::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    PC_2010_2019_6_month = PC_2010_2019[5::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    # put them together
    PC_2010_2019_4_6_month = np.concatenate(
        [
            PC_2010_2019_4_month,
            PC_2010_2019_5_month,
            PC_2010_2019_6_month,
        ],
        axis=0,
    ).reshape(-1, 180, 360)
    PC_2017_2019_4_6_month = PC_2010_2019_4_6_month[588:, :, :]
    # --------- Get 4-6 month data of each years ---------#
    PC_2020_4_6_month = PC_all[123:126, :, :, :].reshape(
        -1, 180, 360
    )
    # PC_2019_4_6_month = PC_2010_2019_4_6_month[756:, :, :]
    # PC_2018_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]
    # PC_2017_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]
    # PC_2018_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]
    # PC_2018_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]
    # PC_2018_4_6_month = PC_2010_2019_4_6_month[672:756, :, :]

    # region
    PC_all = PC_all.reshape(3696, 180, 360)
    PC_2010_2019 = PC_2010_2019.reshape(3360, 180, 360)
    PC_2010 = PC_2010.reshape(336, 180, 360)
    PC_2011 = PC_2011.reshape(336, 180, 360)
    PC_2012 = PC_2012.reshape(336, 180, 360)
    PC_2013 = PC_2013.reshape(336, 180, 360)
    PC_2014 = PC_2014.reshape(336, 180, 360)
    PC_2015 = PC_2015.reshape(336, 180, 360)
    PC_2016 = PC_2016.reshape(336, 180, 360)
    PC_2017 = PC_2017.reshape(336, 180, 360)
    PC_2018 = PC_2018.reshape(336, 180, 360)
    PC_2019 = PC_2019.reshape(336, 180, 360)
    PC_2020 = PC_2020.reshape(336, 180, 360)
    # endregion

    Cldarea_data = np.array(data0.Cldarea)
    Cldicerad_data = np.array(data0.Cldicerad)
    Cldtau_data = np.array(data0.Cldtau)
    Cldtau_lin_data = np.array(data0.Cldtau_lin)
    IWP_data = np.array(data0.IWP)
    Cldemissir_data = np.array(data0.Cldemissir)

    # --------- Original data ---------#
    # region
    # Cld_all = Cldarea_data.reshape(
    #     132, 28, 180, 360
    # )  # Choose the variable used in the plot
    Cld_all = Cldarea_data.reshape(
        132, 28, 180, 360
    )  # Choose the variable used in the plot
    Cld_all[Cld_all == -999] = np.nan

    Cld_2010_2019 = Cld_all[0:120, :, :, :]  # 2010-2019
    Cld_2018_2020 = Cld_all[84:132, :, :, :]  # 2018-2020
    Cld_2010 = Cld_all[0:12, :, :, :]  # 2010
    Cld_2011 = Cld_all[12:24, :, :, :]  # 2011
    Cld_2012 = Cld_all[24:36, :, :, :]  # 2012
    Cld_2013 = Cld_all[36:48, :, :, :]  # 2013
    Cld_2014 = Cld_all[48:60, :, :, :]  # 2014
    Cld_2015 = Cld_all[60:72, :, :, :]  # 2015
    Cld_2016 = Cld_all[72:84, :, :, :]  # 2016
    Cld_2017 = Cld_all[84:96, :, :, :]  # 2017
    Cld_2018 = Cld_all[96:108, :, :, :]  # 2018
    Cld_2019 = Cld_all[108:120, :, :, :]  # 2019
    Cld_2020 = Cld_all[120:132, :, :, :]  # 2020
    # endregion

    # --------- Get 4-6 month data of each years ---------#
    Cld_2010_2019_4_month = Cld_2010_2019[3::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    Cld_2010_2019_5_month = Cld_2010_2019[4::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    Cld_2010_2019_6_month = Cld_2010_2019[5::12, :, :, :].reshape(
        (-1, 180, 360)
    )
    # put them together
    Cld_2010_2019_4_6_month = np.concatenate(
        [
            Cld_2010_2019_4_month,
            Cld_2010_2019_5_month,
            Cld_2010_2019_6_month,
        ],
        axis=0,
    ).reshape(-1, 180, 360)
    Cld_2017_2019_4_6_month = Cld_2010_2019_4_6_month[588:, :, :]
    # --------- Get 4-6 month data of each years ---------#
    Cld_2020_4_6_month = Cld_all[123:126, :, :, :].reshape(
        -1, 180, 360
    )

    # region
    Cld_all = Cld_all.reshape(3696, 180, 360)
    Cld_2010_2019 = Cld_2010_2019.reshape(3360, 180, 360)
    Cld_2010 = Cld_2010.reshape(336, 180, 360)
    Cld_2011 = Cld_2011.reshape(336, 180, 360)
    Cld_2012 = Cld_2012.reshape(336, 180, 360)
    Cld_2013 = Cld_2013.reshape(336, 180, 360)
    Cld_2014 = Cld_2014.reshape(336, 180, 360)
    Cld_2015 = Cld_2015.reshape(336, 180, 360)
    Cld_2016 = Cld_2016.reshape(336, 180, 360)
    Cld_2017 = Cld_2017.reshape(336, 180, 360)
    Cld_2018 = Cld_2018.reshape(336, 180, 360)
    Cld_2019 = Cld_2019.reshape(336, 180, 360)
    Cld_2020 = Cld_2020.reshape(336, 180, 360)
    # endregion

    return (
        PC_all,
        PC_2010_2019_4_6_month,
        PC_2017_2019_4_6_month,
        PC_2020_4_6_month,
        PC_2010_2019,
        PC_2010,
        PC_2011,
        PC_2012,
        PC_2013,
        PC_2014,
        PC_2015,
        PC_2016,
        PC_2017,
        PC_2018,
        PC_2019,
        PC_2020,
        Cld_all,
        Cld_2010_2019_4_6_month,
        Cld_2017_2019_4_6_month,
        Cld_2020_4_6_month,
        Cld_2010_2019,
        Cld_2018_2020,
        Cld_2010,
        Cld_2011,
        Cld_2012,
        Cld_2013,
        Cld_2014,
        Cld_2015,
        Cld_2016,
        Cld_2017,
        Cld_2018,
        Cld_2019,
        Cld_2020,
    )


(
    PC_all,
    PC_2010_2019_4_6_month,
    PC_2017_2019_4_6_month,
    PC_2020_4_6_month,
    PC_2010_2019,
    PC_2010,
    PC_2011,
    PC_2012,
    PC_2013,
    PC_2014,
    PC_2015,
    PC_2016,
    PC_2017,
    PC_2018,
    PC_2019,
    PC_2020,
    ######### Read in CERES data #############
    Cld_all,
    Cld_2010_2019_4_6_month,
    Cld_2017_2019_4_6_month,
    Cld_2020_4_6_month,
    Cld_2010_2019,
    Cld_2018_2020,
    Cld_2010,
    Cld_2011,
    Cld_2012,
    Cld_2013,
    Cld_2014,
    Cld_2015,
    Cld_2016,
    Cld_2017,
    Cld_2018,
    Cld_2019,
    Cld_2020,
) = read_PC1_CERES_from_netcdf()

# --------- Divide 2017-2019 data into 2017,2018,2019 data ---------#
PC_2017_4_6_month = PC_2017_2019_4_6_month[:84, :, :]
PC_2018_4_6_month = PC_2017_2019_4_6_month[84:168, :, :]
PC_2019_4_6_month = PC_2017_2019_4_6_month[168:252, :, :]

Cld_2017_4_6_month = Cld_2017_2019_4_6_month[:84, :, :]
Cld_2018_4_6_month = Cld_2017_2019_4_6_month[84:168, :, :]
Cld_2019_4_6_month = Cld_2017_2019_4_6_month[168:252, :, :]

#################################################################################
####### Filter_data_fit_PC1_gap #################################################
#################################################################################


import matplotlib.pyplot as plt

# Use the font and apply the matplotlib style
plt.rc("font", family="Times New Roman")
plt.style.use("seaborn")
# Reuse the same font to ensure that font set properly
plt.rc("font", family="Times New Roman")


@jit(nopython=True, parallel=True)
def NUMBA_FILTER_DATA_FIT_PC1_GAP(
    var1,
    var2,
    coef,
    gap_num,
    PC_gap_len,
    latitude_len,
    longitude_len,
    Cld_data,
    PC_data,
):

    Cld_match_PC_gap = np.zeros(
        (PC_gap_len, latitude_len, longitude_len)
    )
    PC_match_PC_gap = np.zeros(
        (PC_gap_len, latitude_len, longitude_len)
    )
    print("Start filtering data")
    for lat in range(latitude_len):
        for lon in range(longitude_len):
            for gap_num in range(PC_gap_len):
                # Filter Cld data with gap, start and end with giving gap
                Cld_match_PC_gap[gap_num, lat, lon] = np.nanmean(
                    Cld_data[:, lat, lon][
                        np.where(
                            (
                                PC_data[:, lat, lon]
                                >= (
                                    np.array(gap_num + var1)
                                    * coef
                                )
                            )
                            & (
                                PC_data[:, lat, lon]
                                < (
                                    np.array(gap_num + var2)
                                    * coef
                                )
                            )
                        )
                    ]
                )
                # generate PC match PC gap as well to insure
                PC_match_PC_gap[gap_num, lat, lon] = np.nanmean(
                    PC_data[:, lat, lon][
                        np.where(
                            (
                                PC_data[:, lat, lon]
                                >= (
                                    np.array(gap_num + var1)
                                    * coef
                                )
                            )
                            & (
                                PC_data[:, lat, lon]
                                < (
                                    np.array(gap_num + var2)
                                    * coef
                                )
                            )
                        )
                    ]
                )

    return Cld_match_PC_gap, PC_match_PC_gap


class Filter_data_fit_PC1_gap_plot(object):
    def __init__(self, start, end, gap):
        self.start = start
        self.end = end
        self.gap = gap
        self.latitude = [i for i in range(0, 180, 1)]
        self.longitude = [i for i in range(0, 360, 1)]

    def Filter_data_fit_PC1_gap(self, Cld_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data
        PC_data : numpy.array
            PC data
        start : int
            Start PC value, like -1
        end : int
            End PC value, like 2
        gap : int
            Giving gap, like 0.2

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num, " ******")
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
            " ******",
        )
        PC_gap = [i for i in range(0, int(gap_num), 1)]

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        for lat in self.latitude:
            for lon in self.longitude:
                for gap_num in PC_gap:
                    # Filter Cld data with gap, start and end with giving gap
                    Cld_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        Cld_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )
                    # generate PC match PC gap as well to insure
                    PC_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        PC_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )

        return Cld_match_PC_gap, PC_match_PC_gap

    def Filter_data_fit_PC1_gap_each_day(self, Cld_data, PC_data):
        """
        Filter data with gap, start and end with giving gap

        Parameters
        ----------
        Cld_data : numpy.array
            Cloud data
        PC_data : numpy.array
            PC data
        start : int
            Start PC value, like -1
        end : int
            End PC value, like 2
        gap : int
            Giving gap, like 0.2

        Returns
        -------
        Cld_data_fit : numpy.array
            Filtered data, Cld data for each PC gap
            array(PC_gap, lat, lon)
        """
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num)
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
        )
        PC_gap = [i for i in range(0, int(gap_num), 1)]

        Cld_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        PC_match_PC_gap = np.zeros(
            (len(PC_gap), len(self.latitude), len(self.longitude))
        )
        for lat in self.latitude:
            for lon in self.longitude:
                for gap_num in PC_gap:
                    # Filter Cld data with gap, start and end with giving gap
                    Cld_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        Cld_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )
                    # generate PC match PC gap as well to insure
                    PC_match_PC_gap[
                        gap_num, lat, lon
                    ] = np.nanmean(
                        PC_data[:, lat, lon][
                            np.where(
                                (
                                    PC_data[:, lat, lon]
                                    >= (
                                        np.array(gap_num + var1)
                                        * coef
                                    )
                                )
                                & (
                                    PC_data[:, lat, lon]
                                    < (
                                        np.array(gap_num + var2)
                                        * coef
                                    )
                                )
                            )
                        ]
                    )

        return Cld_match_PC_gap, PC_match_PC_gap

    def numba_Filter_data_fit_PC1_gap(self, Cld_data, PC_data):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        PC_gap = [i for i in range(0, int(gap_num), 1)]

        print("****** Start filter data with PC gap ******")
        print("****** Current gap number is:", gap_num)
        print(
            "****** Current PC1 range is:",
            self.start,
            " to ",
            self.end,
        )

        return NUMBA_FILTER_DATA_FIT_PC1_GAP(
            var1,
            var2,
            coef,
            gap_num,
            len(PC_gap),
            len(self.latitude),
            len(self.longitude),
            Cld_data,
            PC_data,
        )

    def give_loop_list_for_giving_gap(self):
        """
        Give the loop list for giving gap

        Parameters
        ----------
        start : int
            start of the loop
        end : int
            end of the loop
        gap : int
            gap

        Returns
        -------
        loop_list : list
            loop list
        """
        range = self.end - self.start
        loop_num = range / self.gap

        var1 = (
            self.start
            * (loop_num - 1)
            / ((self.end - self.gap) - self.start)
        )
        coefficient = self.start / var1
        var2 = self.gap / coefficient + var1

        return var1, var2, coefficient, loop_num

    def calc_correlation_PC1_Cld(self, PC_data, Cld_data):
        Correlation = np.zeros((180, 360))

        for i in range(180):
            for j in range(360):
                Correlation[i, j] = pd.Series(
                    PC_data[:, i, j]
                ).corr(
                    pd.Series(Cld_data[:, i, j]),
                    method="pearson",
                )

        return Correlation

    def plot_PC1_Cld(
        self, start, end, PC_match_PC_gap, Cld_match_PC_gap
    ):

        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        print("****** Start plot PC1 ******")
        fig, (ax1, ax2) = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(20, 20),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            211,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray", alpha=0)
        cmap.set_over("#800000", alpha=0)
        cmap.set_under("#191970", alpha=0)

        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # vmax=1.5,
            # vmin=-1.5,
            cmap=cmap,
        )
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cbar = fig.colorbar(
            a,
            ax=[ax1],
            location="right",
            shrink=0.9,
            extend="both",
            label="PC 1",
        )
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        cbar.ax.tick_params(labelsize=24)

        ax2 = plt.subplot(
            212,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        ax2.coastlines(resolution="50m", lw=0.3)
        ax2.set_global()
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # vmax=30,
            # vmin=0,
            cmap=cmap,
        )
        gl = ax2.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cbar = fig.colorbar(
            b,
            ax=[ax2],
            location="right",
            shrink=0.9,
            extend="both",
            label="HCF (%)",
        )
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        cbar.ax.tick_params(labelsize=24)
        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(
                np.round((np.array((end - 1) + var2)) * coef, 2)
            ),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_PC1_Cld_Difference(
        self,
        start,
        end,
        PC_match_PC_gap,
        Cld_match_PC_gap,
        pc_max,
        cld_max,
    ):

        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        print("****** Start plot PC1 ******")
        fig, (ax1, ax2) = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(15, 15),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            211,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        # norm1 = colors.CenteredNorm(halfrange=pc_max)
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            # norm=norm1,
            vmax=pc_max,
            vmin=-pc_max,
            cmap=cmap,
        )
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb1 = fig.colorbar(
            a,
            ax=ax1,
            extend="both",
            location="right",
            shrink=0.8,
        )
        # adjust the colorbar label size
        cb1.set_label(label="PC 1", size=24)
        cb1.ax.tick_params(labelsize=24)

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}
        # set serial number for this subplot
        # ax1.text(
        #     0.05,
        #     0.95,
        #     "PC",
        #     transform=ax1.transAxes,
        #     fontsize=24,
        #     verticalalignment="top",
        # )

        ax2 = plt.subplot(
            212,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        ax2.coastlines(resolution="50m", lw=0.7)
        ax2.set_global()
        norm2 = colors.CenteredNorm(halfrange=cld_max)
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_match_PC_gap[start:end, :, :], axis=0),
            transform=ccrs.PlateCarree(),
            norm=norm2,
            cmap=cmap,
        )
        gl = ax2.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = fig.colorbar(
            b,
            ax=ax2,
            location="right",
            shrink=0.8,
            extend="both",
        )
        cb2.set_label(label="HCF (%)", size=24)
        cb2.ax.tick_params(labelsize=24)

        # cbar.ax.tick_params(labelsize=24)
        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(
                np.round((np.array((end - 1) + var2)) * coef, 2)
            ),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_PC1_Cld_test(
        self, start, end, Cld_data, cld_max,
    ):
        (
            var1,
            var2,
            coef,
            gap_num,
        ) = self.give_loop_list_for_giving_gap()

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)

        print("****** Start plot PC1 ******")
        fig, ax1 = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(15, 8),
            constrained_layout=True,
        )
        plt.rcParams.update({"font.family": "Times New Roman"})

        ax1 = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        # norm1 = colors.CenteredNorm(halfrange=pc_max)
        a = ax1.pcolormesh(
            lon,
            lat,
            Cld_data,
            transform=ccrs.PlateCarree(),
            # norm=norm1,
            vmax=cld_max,
            vmin=-cld_max,
            cmap=cmap,
        )
        gl = ax1.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb1 = fig.colorbar(
            a,
            ax=ax1,
            extend="both",
            location="right",
            shrink=0.7,
        )
        # adjust the colorbar label size
        cb1.set_label(label="PC 1", size=24)
        cb1.ax.tick_params(labelsize=24)

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        plt.suptitle(
            str(np.round((np.array(start + var1)) * coef, 2))
            + "<=PC1<"
            + str(
                np.round((np.array((end - 1) + var2)) * coef, 2)
            ),
            x=0.45,
            y=0.99,
            size=42,
            fontweight="bold",
        )

    def plot_All_year_mean_PC1_Cld(self, PC_data, Cld_data):
        # plot all year mean PC1 and Cld
        # ! Input PC_data and Cld_data must be the same shape
        # ! [time, lat, lon]
        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)
        # lon,lat1 = np.meshgrid(lon,lat1)

        print(
            "****** Start plot all year mean PC1 and Cld ******"
        )
        fig = plt.figure(figsize=(18, 15))
        plt.rc("font", size=10, weight="bold")

        cmap = dcmap("/RAID01/data/muqy/color/test_cld.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("white")

        cmap1 = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap1.set_bad("gray")
        cmap1.set_over("#800000")
        cmap1.set_under("#191970")

        ax1 = plt.subplot(
            2,
            1,
            1,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax1.coastlines(resolution="50m", lw=0.9)
        ax1.set_global()
        a = ax1.pcolormesh(
            lon,
            lat,
            np.nanmean(Cld_data[:, :, :], axis=0),
            linewidth=0,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmax=40,
            vmin=0,
        )
        ax1.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
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
            2,
            1,
            2,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax2.coastlines(resolution="50m", lw=0.7)
        ax2.set_global()
        b = ax2.pcolormesh(
            lon,
            lat,
            np.nanmean(PC_data[:, :, :], axis=0),
            linewidth=0,
            transform=ccrs.PlateCarree(),
            # norm=MidpointNormalize(midpoint=0),
            cmap=cmap1,
            vmax=2,
            vmin=-1,
        )
        ax2.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        ax2.set_title(" Principle Component 1 (PC1) ", size=12)
        fig.colorbar(
            b,
            ax=[ax2],
            location="right",
            shrink=0.9,
            extend="both",
            label="PC1",
        )
        # plt.savefig('PC1_CLDAREA1.pdf')
        # plt.tight_layout()
        plt.show()

    def plot_correlation_PC1_Cld(self, Corr_data):

        lon = np.linspace(0, 359, 360)
        lat = np.linspace(-90, 89, 180)
        lat1 = np.linspace(0, 69, 70)
        # lon,lat1 = np.meshgrid(lon,lat1)

        fig = plt.figure(figsize=(10, 6))
        plt.rc("font", size=10, weight="bold")

        cmap = dcmap("/RAID01/data/muqy/color/test.txt")
        cmap.set_bad("gray")
        cmap.set_over("#800000")
        cmap.set_under("#191970")

        ax1 = plt.subplot(
            1,
            1,
            1,
            projection=ccrs.Mollweide(central_longitude=180),
        )
        ax1.coastlines(resolution="50m", lw=0.3)
        ax1.set_global()
        a = ax1.pcolor(
            lon,
            lat,
            Corr_data,
            # Corr_all,
            # Corr_d,
            linewidth=0,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmax=1,
            vmin=-1,
        )
        ax1.gridlines(linestyle="-.", lw=0, alpha=0.5)
        # gl.xlabels_top = False
        # gl.ylabels_left = False
        ax1.set_title(" PC1-HCF Correlation (Corr) ", size=12)
        fig.colorbar(
            a,
            ax=[ax1],
            location="right",
            shrink=0.9,
            extend="both",
            label="Corr",
        )

        plt.show()


filter_data_fit_PC1_gap_plot = Filter_data_fit_PC1_gap_plot(
    start=-1.5, end=4.5, gap=0.05
)

# ------ Segmentation of cloud data within each PC interval ---------------------------------

# region
(
    Cld_2017_match_PC_gap,
    PC_2017_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2017, PC_data=PC_2017,
)

(
    Cld_2018_match_PC_gap,
    PC_2018_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2018, PC_data=PC_2018,
)

(
    Cld_2019_match_PC_gap,
    PC_2019_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2019, PC_data=PC_2019,
)

(
    Cld_2020_match_PC_gap,
    PC_2020_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2020, PC_data=PC_2020,
)

(
    Cld_2010_2019_match_PC_gap,
    PC_2010_2019_match_PC_gap,
) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
    Cld_data=Cld_2010_2019, PC_data=PC_2010_2019,
)
# -------------- 2010-2019 4-6 month version ------------------------------------------------------------
# (
#     Cld_2020_match_PC_gap,
#     PC_2020_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2020_4_6_month, PC_data=PC_2020_4_6_month,
# )

# (
#     Cld_2010_2019_match_PC_gap,
#     PC_2010_2019_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2010_2019_4_6_month, PC_data=PC_2010_2019_4_6_month,
# )
# -------------- 2017-2019 4-6 month version ------------------------------------------------------------
# (
#     Cld_2017_match_PC_gap,
#     PC_2017_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2017_4_6_month, PC_data=PC_2017_4_6_month,
# )

# (
#     Cld_2018_match_PC_gap,
#     PC_2018_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2018_4_6_month, PC_data=PC_2018_4_6_month,
# )

# (
#     Cld_2019_match_PC_gap,
#     PC_2019_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2019_4_6_month, PC_data=PC_2019_4_6_month,
# )

# (
#     Cld_2020_match_PC_gap,
#     PC_2020_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2020_4_6_month, PC_data=PC_2020_4_6_month,
# )

# (
#     Cld_2010_2019_match_PC_gap,
#     PC_2010_2019_match_PC_gap,
# ) = filter_data_fit_PC1_gap_plot.Filter_data_fit_PC1_gap(
#     Cld_data=Cld_2017_2019_4_6_month,
#     PC_data=PC_2017_2019_4_6_month,
# )

# endregion

# -----  Plot filtered data to verify ---------------------------------

filter_data_fit_PC1_gap_plot.plot_PC1_Cld(
    90, 110, PC_2010_2019_match_PC_gap, Cld_2010_2019_match_PC_gap
)

# ----- Plot global mean data to verify ---------------------------------

filter_data_fit_PC1_gap_plot.plot_All_year_mean_PC1_Cld(
    PC_all, Cld_all
)

# ----- Plot correlation between PC1 and Cld ---------------------------------

Correlation = filter_data_fit_PC1_gap_plot.calc_correlation_PC1_Cld(
    PC_all, Cld_all
)

filter_data_fit_PC1_gap_plot.plot_correlation_PC1_Cld(Correlation)

# ----- Plot filtered data with boxplot ---------------------------------


class Box_plot(object):
    """
    Plot boxplot of Cld data match each PC1 interval
    
    """

    def __init__(self, Cld_match_PC_gap, time_str):
        """
        Initialize the class

        Parameters
        ----------
        Cld_match_PC_gap : Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
        time_str : string
            time string like '2010to2019' or "2010to2019_4_6_month" or "2018only"
        """
        # Input array must be in shape of (PC1_gap, lat, lon)
        self.Cld_match_PC_gap = Cld_match_PC_gap
        self.time_str = time_str

    def Convert_pandas(self):

        gap_num = self.Cld_match_PC_gap.shape[0]
        Box = np.zeros(
            (
                self.Cld_match_PC_gap.shape[1]
                * self.Cld_match_PC_gap.shape[2],
                gap_num,
            )
        )
        # Box = np.zeros((gap_num, 64800, ))

        for i in range(gap_num):
            Box[:, i] = self.Cld_match_PC_gap[i, :, :].reshape(
                64800
            )

        Box = pd.DataFrame(Box)
        # Number of PC1 interval is set in np.arange(start, end, interval gap)
        Box.columns = np.round(np.arange(-1.5, 4.5, 0.05), 3)

        return Box

    def plot_box_plot(self):
        """
        Plot boxplot of Cld data match each PC1 interval
        Main plot function
        """
        Box = self.Convert_pandas()

        fig, ax = plt.subplots(figsize=(18, 10))
        flierprops = dict(
            marker="o", markersize=7, markeredgecolor="grey",
        )
        Box.boxplot(
            # sym="o",
            flierprops=flierprops,
            whis=[10, 90],
            meanline=None,
            showmeans=True,
            notch=True,
        )
        plt.xlabel("PC1", size=26, weight="bold")
        plt.ylabel("HCF (%)", size=26, weight="bold")
        # plt.xticks((np.round(np.arange(-1.5, 3.5, 0.5), 2)),fontsize=26, weight="bold", )
        plt.yticks(
            fontsize=26, weight="bold",
        )
        plt.savefig(
            "Box_plot_PC1_Cld_" + self.time_str + ".png",
            dpi=500,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.show()


# Box plot of cld data from 2010 to 2019
box_plot = Box_plot(Cld_2010_2019_match_PC_gap,"2010to2019")
box_plot.plot_box_plot()
# Box plot of cld data 2020 only
box_plot = Box_plot(Cld_2020_match_PC_gap,"2020only")
box_plot.plot_box_plot()
# Box plot of cld data 2019 only
box_plot = Box_plot(Cld_2019_match_PC_gap,"2019only")
box_plot.plot_box_plot()
# Box plot of cld data 2018 only
box_plot = Box_plot(Cld_2018_match_PC_gap,"2018only")
box_plot.plot_box_plot()
# Box plot of cld data 2017 only
box_plot = Box_plot(Cld_2017_match_PC_gap,"2017only")
box_plot.plot_box_plot()


# region

# ----- Plot 2018-2020 Cld data& PC data ---------------------------------
# Bad atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    14,
    22,
    PC_2018_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2018_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=5,
)

# Good atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    52,
    60,
    PC_2018_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2018_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=20,
)

# ----- Plot 2019-2020 Cld data& PC data ---------------------------------
# Bad atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    14,
    22,
    PC_2019_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2019_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=5,
)

# Good atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    52,
    60,
    PC_2019_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2019_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=20,
)

# endregion

# ----- Plot mean-2020 Cld data& PC data ---------------------------------
# Bad atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    # 18,
    # 28,
    14,
    18,
    PC_2010_2019_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2010_2019_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=3,
)

# Good atmospheric conditions
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_Difference(
    # 46,
    # 56,
    56,
    60,
    PC_2010_2019_match_PC_gap - PC_2020_match_PC_gap,
    Cld_2010_2019_match_PC_gap - Cld_2020_match_PC_gap,
    pc_max=0.02,
    cld_max=18,
)

# ---------- Comparing the difference between other years and 2020 -------------------
# ---------- In the giving PC1 gap, and the giving atmospheric gap -------------------


def compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020,
    Cld_all_match_PC_gap_others,
    start,
    end,
):
    """
    Loop over the given array to See if the data at each location is nan, if so, 
    assign it to nan, if not, subtract the data at that location within two years
    """
    Cld_all_match_PC_gap_others_sub_2020 = np.zeros(
        (
            Cld_all_match_PC_gap_2020.shape[1],
            Cld_all_match_PC_gap_2020.shape[2],
        )
    )

    Cld_all_match_PC_gap_others_sub_2020[:, :] = np.nan

    for lat in range((Cld_all_match_PC_gap_2020.shape[1])):
        for lon in range((Cld_all_match_PC_gap_2020.shape[2])):
            for gap in range(start, end):
                if (
                    np.isnan(
                        Cld_all_match_PC_gap_2020[gap, lat, lon]
                    )
                    == False
                ) and (
                    np.isnan(
                        Cld_all_match_PC_gap_others[gap, lat, lon]
                    )
                    == False
                ):
                    Cld_all_match_PC_gap_others_sub_2020[
                        lat, lon
                    ] = (
                        Cld_all_match_PC_gap_others[gap, lat, lon]
                        - Cld_all_match_PC_gap_2020[gap, lat, lon]
                    )
                elif (
                    np.isnan(
                        Cld_all_match_PC_gap_2020[gap, lat, lon]
                    )
                    == True
                ) or (
                    np.isnan(
                        Cld_all_match_PC_gap_others[gap, lat, lon]
                    )
                    == True
                ):
                    if (
                        np.isnan(
                            Cld_all_match_PC_gap_others_sub_2020[
                                lat, lon
                            ]
                        )
                        == True
                    ):
                        Cld_all_match_PC_gap_others_sub_2020[
                            lat, lon
                        ] = np.nan
                    else:
                        pass

    return Cld_all_match_PC_gap_others_sub_2020


# Calculate cld anomaly in fixed bad atmospheric conditions
# region
Cld_all_match_PC_gap_mean_sub_2020_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=14,
    end=18,
    # start=18,
    # end=28,
)
Cld_all_match_PC_gap_mean_sub_2017_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2017_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=14,
    end=18,
    # start=18,
    # end=28,
)
Cld_all_match_PC_gap_mean_sub_2018_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2018_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=14,
    end=18,
    # start=18,
    # end=28,
)
Cld_all_match_PC_gap_mean_sub_2019_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2019_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=14,
    end=18,
    # start=18,
    # end=28,
)
# endregion

# Calculate cld anomaly in fixed good atmospheric conditions
# region
Cld_all_match_PC_gap_mean_sub_2020_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=56,
    end=60,
)
Cld_all_match_PC_gap_mean_sub_2017_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2017_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=56,
    end=60,
)
Cld_all_match_PC_gap_mean_sub_2018_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2018_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=56,
    end=60,
)
Cld_all_match_PC_gap_mean_sub_2019_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2019_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    start=56,
    end=60,
)
# endregion

PC_all_match_PC_gap_mean_sub_2020_bad = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=PC_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=PC_2010_2019_match_PC_gap,
    # start=14,
    # end=18,
    start=18,
    end=28,
)

# Good atmospheric conditions
Cld_all_match_PC_gap_mean_sub_2020_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=Cld_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=Cld_2010_2019_match_PC_gap,
    # start=56,
    # end=60,
    start=46,
    end=56,
)
PC_all_match_PC_gap_mean_sub_2020_good = compare_cld_between_2020_others(
    Cld_all_match_PC_gap_2020=PC_2020_match_PC_gap,
    Cld_all_match_PC_gap_others=PC_2010_2019_match_PC_gap,
    # start=56,
    # end=60,
    start=46,
    end=56,
)

# Plot the difference between the two years
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    # 56,
    # 60,
    46,
    56,
    PC_all_match_PC_gap_mean_sub_2020_good,
    Cld_all_match_PC_gap_mean_sub_2020_good,
    pc_max=0.02,
    cld_max=25,
)

# Plot 2017-2020 cld anormally in bad atmospheric conditions
# region
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2020_bad,
    cld_max=4,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2019_bad,
    cld_max=4,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2018_bad,
    cld_max=4,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2017_bad,
    cld_max=4,
)
# endregion

# Plot 2017-2020 cld anormally in good atmospheric conditions
# region
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    46,
    56,
    # 18,
    # 28,
    Cld_all_match_PC_gap_mean_sub_2020_good,
    cld_max=23,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    46, 56, Cld_all_match_PC_gap_mean_sub_2019_good, cld_max=23,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    46, 56, Cld_all_match_PC_gap_mean_sub_2018_good, cld_max=23,
)
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    46, 56, Cld_all_match_PC_gap_mean_sub_2017_good, cld_max=23,
)
# endregion

# Plot 2020 Cld anormaly without atmospheric control
filter_data_fit_PC1_gap_plot.plot_PC1_Cld_test(
    14,
    18,
    np.nanmean(PC_2017_2019_4_6_month, axis=0)
    - np.nanmean(PC_2020_4_6_month, axis=0),
    np.nanmean(Cld_2017_2019_4_6_month, axis=0)
    - np.nanmean(Cld_2020_4_6_month, axis=0),
    pc_max=1,
    cld_max=20,
)

# ---------- Compare the Cld for giving atmospheric condiction by latitude -------------------

plt.style.use("seaborn-whitegrid")
plt.rc("font", family="Times New Roman")


def compare_cld_between_mean_and_other_year_by_Lat(
    Cld_data_2020, Cld_data_2017, Cld_data_2018, Cld_data_2019
):
    """
    Plot mean cld anormaly from lat 30 to 60

    Parameters
    ----------
    Cld_data : numpy array
        Cld data from different years, shape in (lat, lon)
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_2017[120:150, :], axis=0),
        linewidth=3,
        label="2017",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_2018[120:150, :], axis=0),
        linewidth=3,
        label="2018",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_2019[120:150, :], axis=0),
        linewidth=3,
        label="2019",
        alpha=0.7,
    )
    ax.plot(
        np.linspace(-180, 179, 360),
        np.nanmean(Cld_data_2020[120:150, :], axis=0),
        color="red",
        label="2020",
        linewidth=3,
    )

    ax.set_facecolor("white")
    ax.legend()
    # adjust the legend font size
    for text in ax.get_legend().get_texts():
        plt.setp(text, color="black", fontsize=20)

    x_ticks_mark = [
        "60$^\circ$E",
        "120$^\circ$E",
        "180$^\circ$",
        "120$^\circ$W",
        "60$^\circ$W",
    ]

    x_ticks = [-120, -60, 0, 60, 120]
    plt.xlim([-180, 180])
    plt.xlabel("Longitude", size=23, weight="bold")
    plt.xticks(x_ticks, x_ticks_mark, fontsize=20, weight="bold")
    plt.yticks(fontsize=20, weight="bold")
    plt.ylabel("HCF difference (%)", size=20, weight="bold")
    # plt.ylim(0, 100)
    plt.show()


compare_cld_between_mean_and_other_year_by_Lat(
    # np.nanmean(Cld_all_match_PC_gap_mean_sub_2020_bad, axis=0),
    # np.nanmean(Cld_all_match_PC_gap_mean_sub_2019_bad, axis=0),
    # np.nanmean(Cld_all_match_PC_gap_mean_sub_2018_bad, axis=0),
    # np.nanmean(Cld_all_match_PC_gap_mean_sub_2017_bad, axis=0),
    # Cld_data_2020=Cld_all_match_PC_gap_mean_sub_2020_bad,
    # Cld_data_2017=Cld_all_match_PC_gap_mean_sub_2017_bad,
    # Cld_data_2018=Cld_all_match_PC_gap_mean_sub_2018_bad,
    # Cld_data_2019=Cld_all_match_PC_gap_mean_sub_2019_bad,
    Cld_data_2020=Cld_all_match_PC_gap_mean_sub_2020_good,
    Cld_data_2017=Cld_all_match_PC_gap_mean_sub_2017_good,
    Cld_data_2018=Cld_all_match_PC_gap_mean_sub_2018_good,
    Cld_data_2019=Cld_all_match_PC_gap_mean_sub_2019_good,
)

