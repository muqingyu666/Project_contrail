# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 18:52:11 2021

@author: Mu o(*￣▽￣*)ブ
"""

import numpy as np
import glob
import os
import calendar
import xarray as xr

A_N20 = np.zeros((1))
A_N21 = np.zeros((1))
A_N22 = np.zeros((1))
A_N23 = np.zeros((1))
A_N24 = np.zeros((1))
A_N25 = np.zeros((1))
A_N26 = np.zeros((1))
A_N27 = np.zeros((1))

CERES_file = glob.glob("G:\\CERES_TOA_RAD\\CERES_rad_" + "*.nc")

for i in range(0, 11):

    FILE_NAME = CERES_file[i]
    id_name = int(os.path.basename(CERES_file[i])[10:14])

    if calendar.isleap(id_name) == False:

        file_obj = xr.open_dataset(FILE_NAME)
        lat = file_obj.lat
        lon = file_obj.lon
        # t = file_obj.time
        toa_alb = file_obj.toa_alb_all_daily
        toa_lw = file_obj.toa_lw_all_daily
        toa_net = file_obj.toa_net_all_daily
        toa_sw = file_obj.toa_sw_all_daily

        toa_alb = np.array(toa_alb)
        toa_lw = np.array(toa_lw)
        toa_net = np.array(toa_net)
        toa_sw = np.array(toa_sw)

        for j in range(31, 59):  # FEB

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(59, 87):  # MAR

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(90, 118):  # APR

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(120, 148):  # MAY

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(151, 179):  # JUN

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(181, 209):  # JUL

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

    elif calendar.isleap(id_name) == True:

        file_obj = xr.open_dataset(FILE_NAME)
        lat = file_obj.lat
        lon = file_obj.lon
        # t = file_obj.time
        toa_alb = file_obj.toa_alb_all_daily
        toa_lw = file_obj.toa_lw_all_daily
        toa_net = file_obj.toa_net_all_daily
        toa_sw = file_obj.toa_sw_all_daily

        toa_alb = np.array(toa_alb)
        toa_lw = np.array(toa_lw)
        toa_net = np.array(toa_net)
        toa_sw = np.array(toa_sw)

        for j in range(31, 59):  # FEB

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(60, 88):  # MAR

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(91, 119):  # APR

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(121, 149):  # MAY

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(152, 180):  # JUN

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

        for j in range(182, 210):  # JUL

            toa_alb1 = toa_alb[j, :, :].reshape(64800)
            toa_lw1 = toa_lw[j, :, :].reshape(64800)
            toa_net1 = toa_net[j, :, :].reshape(64800)
            toa_sw1 = toa_sw[j, :, :].reshape(64800)

            A_N20 = np.concatenate((A_N20, toa_alb1), axis=0)
            A_N21 = np.concatenate((A_N21, toa_lw1), axis=0)
            A_N22 = np.concatenate((A_N22, toa_net1), axis=0)
            A_N23 = np.concatenate((A_N23, toa_sw1), axis=0)

A_N20 = np.delete(A_N20, 0, axis=0)  # toa_alb
A_N21 = np.delete(A_N21, 0, axis=0)  # toa_lw
A_N22 = np.delete(A_N22, 0, axis=0)  # toa_net
A_N23 = np.delete(A_N23, 0, axis=0)  # toa_sw

A_N20t = A_N21.reshape(
    66, 28, 180, 360
)  # Choose the variable used in the plot
A_N20t[A_N20t == -999] = np.nan

A_NM = A_N20t[0:60, :, :, :]  # 2010-2019
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
