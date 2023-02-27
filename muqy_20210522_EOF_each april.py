# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:37:08 2021

@author: Mu o(*￣▽￣*)ブ
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from scipy import stats
from scipy.stats import zscore
import calendar
from sklearn.decomposition import PCA
import glob
import pandas as pd
import os
from sklearn.preprocessing import scale

def dcmap(file_path):
      fid=open(file_path)
      data=fid.readlines()
      n=len(data);
      rgb=np.zeros((n,3))
      for i in np.arange(n):
            rgb[i][0]=data[i].split(',')[0]
            rgb[i][1]=data[i].split(',')[1]
            rgb[i][2]=data[i].split(',')[2]
            rgb[i]=rgb[i]/255.0           
            icmap=mpl.colors.ListedColormap(rgb,name='my_color')                  
      return icmap

########################### ERA5 PCA ######################################

A_N10 = np.zeros((1))
A_N11 = np.zeros((1))
A_N12 = np.zeros((1))

for i in range(0,11):
    
    # year_str=str(2010).zfill(4)
    # month_str=str(1).zfill(2)
    # time_str0=year_str+month_str
    
    ERA_file=glob.glob('G:\\ERA5_daily_stored per month_global_3\\ERA5_daily_' +'*.nc')
    FILE_NAME_ERA = ERA_file[i]
    
    for i in range(0,28):
        
        file_obj = xr.open_dataset(FILE_NAME_ERA)
        lat = file_obj.lat
        lon = file_obj.lon
        P = file_obj.level
        z = file_obj.Geo
        RH = file_obj.RH
        T = file_obj.T
        W = file_obj.W
        T = T[:,7,:,:]
        RH = RH[:,7,:,:]
        W = W[:,7,:,:]
        
        T = np.delete(T, 0, axis=1)
        RH = np.delete(RH, 0, axis=1)
        W = np.delete(W, 0, axis=1)
        
        T[i,:,:] = np.flipud(T[i,:,:])
        RH[i,:,:] = np.flipud(RH[i,:,:])
        W[i,:,:] = np.flipud(W[i,:,:])
        
        # RH = RH[:,:,:]
        # T = T[:,:,:]
        # W = W[:,:,:]
        
        RH_1 = np.array(RH[i,:,:]).reshape(64800)
        T_1 = np.array(T[i,:,:]).reshape(64800)
        W_1 = np.array(W[i,:,:]).reshape(64800)
        
        # RH_N = stats.zscore(RH_1)
        # T_N = stats.zscore(T_1)
        # W_N = stats.zscore(W_1)
        
        A_N10 = np.concatenate((A_N10,RH_1),axis=0)
        A_N11 = np.concatenate((A_N11,T_1),axis=0)
        A_N12 = np.concatenate((A_N12,W_1),axis=0)
        
A_N10 = np.delete(A_N10, 0, axis=0)
A_N11 = np.delete(A_N11, 0, axis=0)
A_N12 = np.delete(A_N12, 0, axis=0)

A_N10 = scale(A_N10)
A_N11 = scale(A_N11)
A_N12 = scale(A_N12)

A_N1 = np.zeros((19958400,3))
A_N1[:,0] = A_N10
A_N1[:,1] = A_N11
A_N1[:,2] = A_N12

pca = PCA(n_components=1,whiten=False,copy=False)
# pca.fit(A_N1)
A_N1 = pca.fit_transform(A_N1)
# A_N1 = A_N1.reshape(11,28,180,360)

# A_N1 = A_N1[1:11,:,:,:]
# A_N1 = np.mean(A_N1[:,:,:,:], axis=0) 
A_N1 = A_N1.reshape(19958400)
A_N1 = A_N1.reshape(11,28,180,360)
A_N13 = A_N1[0,:,:,:] #2010
A_N14 = A_N1[1,:,:,:] #2011
A_N15 = A_N1[2,:,:,:] #2012
A_N16 = A_N1[3,:,:,:] #2013
A_N17 = A_N1[4,:,:,:] #2014
A_N18 = A_N1[5,:,:,:] #2015
A_N19 = A_N1[6,:,:,:] #2016
A_N110 = A_N1[7,:,:,:] #2017
A_N111 = A_N1[8,:,:,:] #2018
A_N112 = A_N1[9,:,:,:] #2019
A_N113 = A_N1[10,:,:,:] #2020


A_N1 = A_N1.reshape(19958400)
A_N13 = A_N13.reshape(1814400)
A_N14 = A_N14.reshape(1814400)
A_N15 = A_N15.reshape(1814400)
A_N16 = A_N16.reshape(1814400)
A_N17 = A_N17.reshape(1814400)
A_N18 = A_N18.reshape(1814400)
A_N19 = A_N19.reshape(1814400)
A_N110 = A_N110.reshape(1814400)
A_N111 = A_N111.reshape(1814400)
A_N112 = A_N112.reshape(1814400)
A_N113 = A_N113.reshape(1814400)

################################# rename ERA5 #######################################

# CERES_file=glob.glob('G:\\CERES_highcloud\\02-05\\CERES_highcloud_' +'*.nc')
# path = 'G:'
# n = 2013

# for i in range(0,10):
#     oldname = CERES_file[i]
#     newname = path+ os.sep +'CERES_highcloud_'+str(n)+'0201-'+str(n)+'0531.nc'
#     n = n+1
#     os.rename(oldname,newname)

############################### Deal CERES data ###################################

A_N20 = np.zeros((1))
A_N21 = np.zeros((1))
A_N22 = np.zeros((1))
A_N23 = np.zeros((1))
A_N24 = np.zeros((1))
A_N25 = np.zeros((1))
A_N26 = np.zeros((1))
A_N27 = np.zeros((1))

CERES_file=glob.glob('G:\\CERES_highcloud\\02-05\\CERES_highcloud_' +'*.nc')

for i in range(0,11):
    
    FILE_NAME_ERA = CERES_file[i]
    id_name = int(os.path.basename(CERES_file[i])[17:21])
    
    if calendar.isleap(id_name)==False:
        
        file_obj = xr.open_dataset(FILE_NAME_ERA)
        lat = file_obj.lat
        lon = file_obj.lon
        # t = file_obj.time
        cldarea = file_obj.cldarea_high_daily
        cldicerad = file_obj.cldicerad_high_daily
        cldtau = file_obj.cldtau_high_daily
        iwp = file_obj.iwp_high_daily
        cldpress = file_obj.cldpress_top_high_daily
        cldphase = file_obj.cldphase_high_daily
        cldemissir = file_obj.cldemissir_high_daily
        cldarea = cldarea[:,:,:]
        cldicerad = cldicerad[:,:,:]
        cldtau = cldtau[:,:,:]
        cldpress = cldpress[:,:,:]
        iwp = iwp[:,:,:]
        cldphase = cldphase[:,:,:]
        cldemissir = cldemissir[:,:,:]
        cldarea = np.array(cldarea)
        cldicerad = np.array(cldicerad)
        cldtau = np.array(cldtau)
        iwp = np.array(iwp)
        cldpress = np.array(cldpress)
        cldphase = np.array(cldphase)
        cldemissir = np.array(cldemissir)
            
        for i in range(59,87):
            
            cldarea1 = cldarea[i,:,:].reshape(64800)
            cldicerad1 = cldicerad[i,:,:].reshape(64800)
            cldtau1 = cldtau[i,:,:].reshape(64800)
            iwp1 = iwp[i,:,:].reshape(64800)
            cldpress1 = cldpress[i,:,:].reshape(64800)
            cldphase1 = cldphase[i,:,:].reshape(64800)
            cldemissir1 = cldemissir[i,:,:].reshape(64800)
            A_N20 = np.concatenate((A_N20,cldarea1),axis=0)
            A_N21 = np.concatenate((A_N21,cldicerad1),axis=0)
            A_N22 = np.concatenate((A_N22,cldtau1),axis=0)
            A_N23 = np.concatenate((A_N23,iwp1),axis=0)
            A_N24 = np.concatenate((A_N24,cldpress1),axis=0)
            A_N25 = np.concatenate((A_N25,cldphase1),axis=0)
            A_N26 = np.concatenate((A_N26,cldemissir1),axis=0)

    else:
        
        FILE_NAME_ERA = CERES_file[10]
        
        file_obj = xr.open_dataset(FILE_NAME_ERA)
        lat = file_obj.lat
        lon = file_obj.lon
        # t = file_obj.time
        cldarea = file_obj.cldarea_high_daily
        cldicerad = file_obj.cldicerad_high_daily
        cldtau = file_obj.cldtau_high_daily
        iwp = file_obj.iwp_high_daily
        cldpress = file_obj.cldpress_top_high_daily
        cldphase = file_obj.cldphase_high_daily
        cldemissir = file_obj.cldemissir_high_daily
        cldarea = cldarea[:,:,:]
        cldicerad = cldicerad[:,:,:]
        cldtau = cldtau[:,:,:]
        cldpress = cldpress[:,:,:]
        iwp = iwp[:,:,:]
        cldphase = cldphase[:,:,:]
        cldemissir = cldemissir[:,:,:]
        cldarea = np.array(cldarea)
        cldicerad = np.array(cldicerad)
        cldtau = np.array(cldtau)
        iwp = np.array(iwp)
        cldpress = np.array(cldpress)
        cldphase = np.array(cldphase)
        cldemissir = np.array(cldemissir)
            
        for i in range(60,88):
            
            cldarea1 = cldarea[i,:,:].reshape(64800)
            cldicerad1 = cldicerad[i,:,:].reshape(64800)
            cldtau1 = cldtau[i,:,:].reshape(64800)
            iwp1 = iwp[i,:,:].reshape(64800)
            cldpress1 = cldpress[i,:,:].reshape(64800)
            cldphase1 = cldphase[i,:,:].reshape(64800)
            cldemissir1 = cldemissir[i,:,:].reshape(64800)
            A_N20 = np.concatenate((A_N20,cldarea1),axis=0)
            A_N21 = np.concatenate((A_N21,cldicerad1),axis=0)
            A_N22 = np.concatenate((A_N22,cldtau1),axis=0)
            A_N23 = np.concatenate((A_N23,iwp1),axis=0)
            A_N24 = np.concatenate((A_N24,cldpress1),axis=0)
            A_N25 = np.concatenate((A_N25,cldphase1),axis=0)
            A_N26 = np.concatenate((A_N26,cldemissir1),axis=0)
    
    
A_N20 = np.delete(A_N20, 0, axis=0) #cldarea
A_N21 = np.delete(A_N21, 0, axis=0) #cldicerad
A_N22 = np.delete(A_N22, 0, axis=0) #cldtau
A_N23 = np.delete(A_N23, 0, axis=0) #iwp
A_N24 = np.delete(A_N24, 0, axis=0) #cldpress
A_N25 = np.delete(A_N25, 0, axis=0) #cldphase
A_N26 = np.delete(A_N26, 0, axis=0) #cldemissir

A_N20 = A_N20.reshape(11,28,180,360) #Choose the variable used in the plot
A_NM = A_N20
A_N30 = A_N20[0,:,:,:] #2010
A_N40 = A_N20[1,:,:,:] #2011
A_N50 = A_N20[2,:,:,:] #2012
A_N60 = A_N20[3,:,:,:] #2013
A_N70 = A_N20[4,:,:,:] #2014
A_N80 = A_N20[5,:,:,:] #2015
A_N90 = A_N20[6,:,:,:] #2016
A_N100 = A_N20[7,:,:,:] #2017
A_N101 = A_N20[8,:,:,:] #2018
A_N102 = A_N20[9,:,:,:] #2019
A_N103 = A_N20[10,:,:,:] #2020

A_N20 = A_N20.reshape(19958400)
A_NM = A_NM.reshape(19958400)
A_N30 = A_N30.reshape(1814400)
A_N40 = A_N40.reshape(1814400)
A_N50 = A_N50.reshape(1814400)
A_N60 = A_N60.reshape(1814400)
A_N70 = A_N70.reshape(1814400)
A_N80 = A_N80.reshape(1814400)
A_N90 = A_N90.reshape(1814400)
A_N100 = A_N100.reshape(1814400)
A_N101 = A_N101.reshape(1814400)
A_N102 = A_N102.reshape(1814400)
A_N103 = A_N103.reshape(1814400)


# A_N20 = np.nanmean(A_N20.reshape(10,1814400), axis=0) 
# A_N21 = np.nanmean(A_N21.reshape(10,1814400), axis=0) 
# A_N22 = np.nanmean(A_N22.reshape(10,1814400), axis=0) 
# A_N23 = np.nanmean(A_N23.reshape(10,1814400), axis=0) 

# A_N21 = A_N21.reshape(28,180,360)
# plt.plot(A_N1, A_N20)

#########################################################################################

# A_NNN = np.zeros((180,360))
# A_NNN = np.random.rand(180,360)
# A_NNN1 = np.random.rand(180,360)
# A_N103 = A_N103.reshape(84,180,360)
# A_N113 = A_N113.reshape(84,180,360)
# A_N102 = A_N102.reshape(84,180,360)
# A_N112 = A_N112.reshape(84,180,360)
# A_nnm = np.nanmean(A_N103,axis = 0)

# for i in range(0,180):
#     for j in range(0,360):
#         if A_N103[:,i,j][np.where((A_N113[:,i,j]>=0))].shape == 0:
#             A_NNN[i,j] = np.nan
#         else :
#             A_NNN[i,j] = np.nanmean(A_N103[:,i,j][np.where((A_N113[:,i,j]>=0))])

# for i in range(0,180):
#     for j in range(0,360):
#         if A_N102[:,i,j][np.where((A_N112[:,i,j]>=0))].shape == 0:
#             A_NNN1[i,j] = np.nan
#         else :
#             A_NNN1[i,j] = np.nanmean(A_N102[:,i,j][np.where((A_N112[:,i,j]>=0))])

# DDD = A_NNN1-A_NNN

############################## plot PCA-cldarea #########################################

B_N0 = np.zeros((40))
B_N0[0] = np.nanmean(A_NM[np.where((A_N1>-5)&(A_N1<=-4.75))])
B_N0[1] = np.nanmean(A_NM[np.where((A_N1>-4.75)&(A_N1<=-4.5))])
B_N0[2] = np.nanmean(A_NM[np.where((A_N1>-4.5)&(A_N1<=-4.25))])
B_N0[3] = np.nanmean(A_NM[np.where((A_N1>-4.25)&(A_N1<=-4))])
B_N0[4] = np.nanmean(A_NM[np.where((A_N1>-4)&(A_N1<=-3.75))])
B_N0[5] = np.nanmean(A_NM[np.where((A_N1>-3.75)&(A_N1<=-3.5))])
B_N0[6] = np.nanmean(A_NM[np.where((A_N1>-3.5)&(A_N1<=-3.25))])
B_N0[7] = np.nanmean(A_NM[np.where((A_N1>-3.25)&(A_N1<=-3))])
B_N0[8] = np.nanmean(A_NM[np.where((A_N1>-3)&(A_N1<=-2.75))])
B_N0[9] = np.nanmean(A_NM[np.where((A_N1>-2.75)&(A_N1<=-2.5))])
B_N0[10] = np.nanmean(A_NM[np.where((A_N1>-2.5)&(A_N1<=-2.25))])
B_N0[11] = np.nanmean(A_NM[np.where((A_N1>-2.25)&(A_N1<=-2))])
B_N0[12] = np.nanmean(A_NM[np.where((A_N1>-2)&(A_N1<=-1.75))])
B_N0[13] = np.nanmean(A_NM[np.where((A_N1>-1.75)&(A_N1<=-1.5))])
B_N0[14] = np.nanmean(A_NM[np.where((A_N1>-1.5)&(A_N1<=-1.25))])
B_N0[15] = np.nanmean(A_NM[np.where((A_N1>-1.25)&(A_N1<=-1))])
B_N0[16] = np.nanmean(A_NM[np.where((A_N1>-1)&(A_N1<=-0.75))])
B_N0[17] = np.nanmean(A_NM[np.where((A_N1>-0.75)&(A_N1<=-0.5))])
B_N0[18] = np.nanmean(A_NM[np.where((A_N1>-0.5)&(A_N1<=-0.25))])
B_N0[19] = np.nanmean(A_NM[np.where((A_N1>-0.25)&(A_N1<=0))])
B_N0[20] = np.nanmean(A_NM[np.where((A_N1>0)&(A_N1<=0.25))])
B_N0[21] = np.nanmean(A_NM[np.where((A_N1>0.25)&(A_N1<=0.5))])
B_N0[22] = np.nanmean(A_NM[np.where((A_N1>0.5)&(A_N1<=0.75))])
B_N0[23] = np.nanmean(A_NM[np.where((A_N1>0.75)&(A_N1<=1))])
B_N0[24] = np.nanmean(A_NM[np.where((A_N1>1)&(A_N1<=1.25))])
B_N0[25] = np.nanmean(A_NM[np.where((A_N1>1.25)&(A_N1<=1.5))])
B_N0[26] = np.nanmean(A_NM[np.where((A_N1>1.5)&(A_N1<=1.75))])
B_N0[27] = np.nanmean(A_NM[np.where((A_N1>1.75)&(A_N1<=2))])
B_N0[28] = np.nanmean(A_NM[np.where((A_N1>2)&(A_N1<=2.25))])
B_N0[29] = np.nanmean(A_NM[np.where((A_N1>2.25)&(A_N1<=2.5))])
B_N0[30] = np.nanmean(A_NM[np.where((A_N1>2.5)&(A_N1<=2.75))])
B_N0[31] = np.nanmean(A_NM[np.where((A_N1>2.75)&(A_N1<=3))])
B_N0[32] = np.nanmean(A_NM[np.where((A_N1>3)&(A_N1<=3.25))])
B_N0[33] = np.nanmean(A_NM[np.where((A_N1>3.25)&(A_N1<=3.5))])
B_N0[34] = np.nanmean(A_NM[np.where((A_N1>3.5)&(A_N1<=3.75))])
B_N0[35] = np.nanmean(A_NM[np.where((A_N1>3.75)&(A_N1<=4))])
B_N0[36] = np.nanmean(A_NM[np.where((A_N1>4)&(A_N1<=4.25))])
B_N0[37] = np.nanmean(A_NM[np.where((A_N1>4.25)&(A_N1<=4.5))])
B_N0[38] = np.nanmean(A_NM[np.where((A_N1>4.5)&(A_N1<=4.75))])
B_N0[39] = np.nanmean(A_NM[np.where((A_N1>4.75)&(A_N1<=5))])

# 2010
# B_N = np.zeros((28))
# B_N[0] = np.nanmean(A_N30[np.where((A_N13>-1.55)&(A_N13<=-1.45))])
# B_N[1] = np.nanmean(A_N30[np.where((A_N13>-1.45)&(A_N13<=-1.35))])
# B_N[2] = np.nanmean(A_N30[np.where((A_N13>-1.35)&(A_N13<=-1.25))])
# B_N[3] = np.nanmean(A_N30[np.where((A_N13>-1.25)&(A_N13<=-1.15))])
# B_N[4] = np.nanmean(A_N30[np.where((A_N13>-1.15)&(A_N13<=-1.05))])
# B_N[5] = np.nanmean(A_N30[np.where((A_N13>-1.05)&(A_N13<=-0.95))])
# B_N[6] = np.nanmean(A_N30[np.where((A_N13>-0.95)&(A_N13<=-0.85))])
# B_N[7] = np.nanmean(A_N30[np.where((A_N13>-0.85)&(A_N13<=-0.75))])
# B_N[8] = np.nanmean(A_N30[np.where((A_N13>-0.75)&(A_N13<=-0.65))])
# B_N[9] = np.nanmean(A_N30[np.where((A_N13>-0.65)&(A_N13<=-0.55))])
# B_N[10] = np.nanmean(A_N30[np.where((A_N13>-0.55)&(A_N13<=-0.45))])
# B_N[11] = np.nanmean(A_N30[np.where((A_N13>-0.45)&(A_N13<=-0.35))])
# B_N[12] = np.nanmean(A_N30[np.where((A_N13>-0.35)&(A_N13<=-0.25))])
# B_N[13] = np.nanmean(A_N30[np.where((A_N13>-0.25)&(A_N13<=-0.15))])
# B_N[14] = np.nanmean(A_N30[np.where((A_N13>-0.15)&(A_N13<=-0.05))])
# B_N[15] = np.nanmean(A_N30[np.where((A_N13>-0.05)&(A_N13<=0.05))])
# B_N[16] = np.nanmean(A_N30[np.where((A_N13>0.05)&(A_N13<=0.15))])
# B_N[17] = np.nanmean(A_N30[np.where((A_N13>0.15)&(A_N13<=0.25))])
# B_N[18] = np.nanmean(A_N30[np.where((A_N13>0.25)&(A_N13<=0.35))])
# B_N[19] = np.nanmean(A_N30[np.where((A_N13>0.35)&(A_N13<=0.45))])
# B_N[20] = np.nanmean(A_N30[np.where((A_N13>0.45)&(A_N13<=0.55))])
# B_N[21] = np.nanmean(A_N30[np.where((A_N13>0.55)&(A_N13<=0.65))])
# B_N[22] = np.nanmean(A_N30[np.where((A_N13>0.65)&(A_N13<=0.75))])
# B_N[23] = np.nanmean(A_N30[np.where((A_N13>0.75)&(A_N13<=0.85))])
# B_N[24] = np.nanmean(A_N30[np.where((A_N13>0.85)&(A_N13<=0.95))])
# B_N[25] = np.nanmean(A_N30[np.where((A_N13>0.95)&(A_N13<=1.05))])
# B_N[26] = np.nanmean(A_N30[np.where((A_N13>1.05)&(A_N13<=1.15))])
# B_N[27] = np.nanmean(A_N30[np.where((A_N13>1.15)&(A_N13<=1.25))])

# #2011
# B_N0 = np.zeros((40))
# B_N0[0] = np.nanmean(A_NM[np.where((A_N1>-5)&(A_N1<=-4.75))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1>-4.75)&(A_N1<=-4.5))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1>-4.5)&(A_N1<=-4.25))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1>-4.25)&(A_N1<=-4))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1>-4)&(A_N1<=-3.75))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1>-3.75)&(A_N1<=-3.5))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1>-3.5)&(A_N1<=-3.25))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1>-3.25)&(A_N1<=-3))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1>-3)&(A_N1<=-2.75))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1>-2.75)&(A_N1<=-2.5))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1>-2.5)&(A_N1<=-2.25))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1>-2.25)&(A_N1<=-2))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1>-2)&(A_N1<=-1.75))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1>-1.75)&(A_N1<=-1.5))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1>-1.5)&(A_N1<=-1.25))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1>-1.25)&(A_N1<=-1))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1>-1)&(A_N1<=-0.75))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1>-0.75)&(A_N1<=-0.5))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1>-0.5)&(A_N1<=-0.25))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1>-0.25)&(A_N1<=0))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1>0)&(A_N1<=0.25))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1>0.25)&(A_N1<=0.5))])
# B_N0[22] = np.nanmean(A_NM[np.where((A_N1>0.5)&(A_N1<=0.75))])
# B_N0[23] = np.nanmean(A_NM[np.where((A_N1>0.75)&(A_N1<=1))])
# B_N0[24] = np.nanmean(A_NM[np.where((A_N1>1)&(A_N1<=1.25))])
# B_N0[25] = np.nanmean(A_NM[np.where((A_N1>1.25)&(A_N1<=1.5))])
# B_N0[26] = np.nanmean(A_NM[np.where((A_N1>1.5)&(A_N1<=1.75))])
# B_N0[27] = np.nanmean(A_NM[np.where((A_N1>1.75)&(A_N1<=2))])
# B_N0[28] = np.nanmean(A_NM[np.where((A_N1>2)&(A_N1<=2.25))])
# B_N0[29] = np.nanmean(A_NM[np.where((A_N1>2.25)&(A_N1<=2.5))])
# B_N0[30] = np.nanmean(A_NM[np.where((A_N1>2.5)&(A_N1<=2.75))])
# B_N0[31] = np.nanmean(A_NM[np.where((A_N1>2.75)&(A_N1<=3))])
# B_N0[32] = np.nanmean(A_NM[np.where((A_N1>3)&(A_N1<=3.25))])
# B_N0[33] = np.nanmean(A_NM[np.where((A_N1>3.25)&(A_N1<=3.5))])
# B_N0[34] = np.nanmean(A_NM[np.where((A_N1>3.5)&(A_N1<=3.75))])
# B_N0[35] = np.nanmean(A_NM[np.where((A_N1>3.75)&(A_N1<=4))])
# B_N0[36] = np.nanmean(A_NM[np.where((A_N1>4)&(A_N1<=4.25))])
# B_N0[37] = np.nanmean(A_NM[np.where((A_N1>4.25)&(A_N1<=4.5))])
# B_N0[38] = np.nanmean(A_NM[np.where((A_N1>4.5)&(A_N1<=4.75))])
# B_N0[39] = np.nanmean(A_NM[np.where((A_N1>4.75)&(A_N1<=5))])


# #2012
# B_N0 = np.zeros((40))
# B_N0[0] = np.nanmean(A_NM[np.where((A_N1>-5)&(A_N1<=-4.75))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1>-4.75)&(A_N1<=-4.5))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1>-4.5)&(A_N1<=-4.25))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1>-4.25)&(A_N1<=-4))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1>-4)&(A_N1<=-3.75))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1>-3.75)&(A_N1<=-3.5))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1>-3.5)&(A_N1<=-3.25))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1>-3.25)&(A_N1<=-3))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1>-3)&(A_N1<=-2.75))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1>-2.75)&(A_N1<=-2.5))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1>-2.5)&(A_N1<=-2.25))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1>-2.25)&(A_N1<=-2))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1>-2)&(A_N1<=-1.75))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1>-1.75)&(A_N1<=-1.5))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1>-1.5)&(A_N1<=-1.25))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1>-1.25)&(A_N1<=-1))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1>-1)&(A_N1<=-0.75))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1>-0.75)&(A_N1<=-0.5))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1>-0.5)&(A_N1<=-0.25))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1>-0.25)&(A_N1<=0))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1>0)&(A_N1<=0.25))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1>0.25)&(A_N1<=0.5))])
# B_N0[22] = np.nanmean(A_NM[np.where((A_N1>0.5)&(A_N1<=0.75))])
# B_N0[23] = np.nanmean(A_NM[np.where((A_N1>0.75)&(A_N1<=1))])
# B_N0[24] = np.nanmean(A_NM[np.where((A_N1>1)&(A_N1<=1.25))])
# B_N0[25] = np.nanmean(A_NM[np.where((A_N1>1.25)&(A_N1<=1.5))])
# B_N0[26] = np.nanmean(A_NM[np.where((A_N1>1.5)&(A_N1<=1.75))])
# B_N0[27] = np.nanmean(A_NM[np.where((A_N1>1.75)&(A_N1<=2))])
# B_N0[28] = np.nanmean(A_NM[np.where((A_N1>2)&(A_N1<=2.25))])
# B_N0[29] = np.nanmean(A_NM[np.where((A_N1>2.25)&(A_N1<=2.5))])
# B_N0[30] = np.nanmean(A_NM[np.where((A_N1>2.5)&(A_N1<=2.75))])
# B_N0[31] = np.nanmean(A_NM[np.where((A_N1>2.75)&(A_N1<=3))])
# B_N0[32] = np.nanmean(A_NM[np.where((A_N1>3)&(A_N1<=3.25))])
# B_N0[33] = np.nanmean(A_NM[np.where((A_N1>3.25)&(A_N1<=3.5))])
# B_N0[34] = np.nanmean(A_NM[np.where((A_N1>3.5)&(A_N1<=3.75))])
# B_N0[35] = np.nanmean(A_NM[np.where((A_N1>3.75)&(A_N1<=4))])
# B_N0[36] = np.nanmean(A_NM[np.where((A_N1>4)&(A_N1<=4.25))])
# B_N0[37] = np.nanmean(A_NM[np.where((A_N1>4.25)&(A_N1<=4.5))])
# B_N0[38] = np.nanmean(A_NM[np.where((A_N1>4.5)&(A_N1<=4.75))])
# B_N0[39] = np.nanmean(A_NM[np.where((A_N1>4.75)&(A_N1<=5))])


# #2013
# B_N0 = np.zeros((40))
# B_N0[0] = np.nanmean(A_NM[np.where((A_N1>-5)&(A_N1<=-4.75))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1>-4.75)&(A_N1<=-4.5))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1>-4.5)&(A_N1<=-4.25))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1>-4.25)&(A_N1<=-4))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1>-4)&(A_N1<=-3.75))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1>-3.75)&(A_N1<=-3.5))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1>-3.5)&(A_N1<=-3.25))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1>-3.25)&(A_N1<=-3))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1>-3)&(A_N1<=-2.75))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1>-2.75)&(A_N1<=-2.5))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1>-2.5)&(A_N1<=-2.25))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1>-2.25)&(A_N1<=-2))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1>-2)&(A_N1<=-1.75))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1>-1.75)&(A_N1<=-1.5))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1>-1.5)&(A_N1<=-1.25))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1>-1.25)&(A_N1<=-1))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1>-1)&(A_N1<=-0.75))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1>-0.75)&(A_N1<=-0.5))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1>-0.5)&(A_N1<=-0.25))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1>-0.25)&(A_N1<=0))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1>0)&(A_N1<=0.25))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1>0.25)&(A_N1<=0.5))])
# B_N0[22] = np.nanmean(A_NM[np.where((A_N1>0.5)&(A_N1<=0.75))])
# B_N0[23] = np.nanmean(A_NM[np.where((A_N1>0.75)&(A_N1<=1))])
# B_N0[24] = np.nanmean(A_NM[np.where((A_N1>1)&(A_N1<=1.25))])
# B_N0[25] = np.nanmean(A_NM[np.where((A_N1>1.25)&(A_N1<=1.5))])
# B_N0[26] = np.nanmean(A_NM[np.where((A_N1>1.5)&(A_N1<=1.75))])
# B_N0[27] = np.nanmean(A_NM[np.where((A_N1>1.75)&(A_N1<=2))])
# B_N0[28] = np.nanmean(A_NM[np.where((A_N1>2)&(A_N1<=2.25))])
# B_N0[29] = np.nanmean(A_NM[np.where((A_N1>2.25)&(A_N1<=2.5))])
# B_N0[30] = np.nanmean(A_NM[np.where((A_N1>2.5)&(A_N1<=2.75))])
# B_N0[31] = np.nanmean(A_NM[np.where((A_N1>2.75)&(A_N1<=3))])
# B_N0[32] = np.nanmean(A_NM[np.where((A_N1>3)&(A_N1<=3.25))])
# B_N0[33] = np.nanmean(A_NM[np.where((A_N1>3.25)&(A_N1<=3.5))])
# B_N0[34] = np.nanmean(A_NM[np.where((A_N1>3.5)&(A_N1<=3.75))])
# B_N0[35] = np.nanmean(A_NM[np.where((A_N1>3.75)&(A_N1<=4))])
# B_N0[36] = np.nanmean(A_NM[np.where((A_N1>4)&(A_N1<=4.25))])
# B_N0[37] = np.nanmean(A_NM[np.where((A_N1>4.25)&(A_N1<=4.5))])
# B_N0[38] = np.nanmean(A_NM[np.where((A_N1>4.5)&(A_N1<=4.75))])
# B_N0[39] = np.nanmean(A_NM[np.where((A_N1>4.75)&(A_N1<=5))])


# #2014
# B_N0 = np.zeros((40))
# B_N0[0] = np.nanmean(A_NM[np.where((A_N1>-5)&(A_N1<=-4.75))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1>-4.75)&(A_N1<=-4.5))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1>-4.5)&(A_N1<=-4.25))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1>-4.25)&(A_N1<=-4))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1>-4)&(A_N1<=-3.75))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1>-3.75)&(A_N1<=-3.5))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1>-3.5)&(A_N1<=-3.25))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1>-3.25)&(A_N1<=-3))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1>-3)&(A_N1<=-2.75))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1>-2.75)&(A_N1<=-2.5))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1>-2.5)&(A_N1<=-2.25))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1>-2.25)&(A_N1<=-2))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1>-2)&(A_N1<=-1.75))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1>-1.75)&(A_N1<=-1.5))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1>-1.5)&(A_N1<=-1.25))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1>-1.25)&(A_N1<=-1))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1>-1)&(A_N1<=-0.75))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1>-0.75)&(A_N1<=-0.5))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1>-0.5)&(A_N1<=-0.25))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1>-0.25)&(A_N1<=0))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1>0)&(A_N1<=0.25))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1>0.25)&(A_N1<=0.5))])
# B_N0[22] = np.nanmean(A_NM[np.where((A_N1>0.5)&(A_N1<=0.75))])
# B_N0[23] = np.nanmean(A_NM[np.where((A_N1>0.75)&(A_N1<=1))])
# B_N0[24] = np.nanmean(A_NM[np.where((A_N1>1)&(A_N1<=1.25))])
# B_N0[25] = np.nanmean(A_NM[np.where((A_N1>1.25)&(A_N1<=1.5))])
# B_N0[26] = np.nanmean(A_NM[np.where((A_N1>1.5)&(A_N1<=1.75))])
# B_N0[27] = np.nanmean(A_NM[np.where((A_N1>1.75)&(A_N1<=2))])
# B_N0[28] = np.nanmean(A_NM[np.where((A_N1>2)&(A_N1<=2.25))])
# B_N0[29] = np.nanmean(A_NM[np.where((A_N1>2.25)&(A_N1<=2.5))])
# B_N0[30] = np.nanmean(A_NM[np.where((A_N1>2.5)&(A_N1<=2.75))])
# B_N0[31] = np.nanmean(A_NM[np.where((A_N1>2.75)&(A_N1<=3))])
# B_N0[32] = np.nanmean(A_NM[np.where((A_N1>3)&(A_N1<=3.25))])
# B_N0[33] = np.nanmean(A_NM[np.where((A_N1>3.25)&(A_N1<=3.5))])
# B_N0[34] = np.nanmean(A_NM[np.where((A_N1>3.5)&(A_N1<=3.75))])
# B_N0[35] = np.nanmean(A_NM[np.where((A_N1>3.75)&(A_N1<=4))])
# B_N0[36] = np.nanmean(A_NM[np.where((A_N1>4)&(A_N1<=4.25))])
# B_N0[37] = np.nanmean(A_NM[np.where((A_N1>4.25)&(A_N1<=4.5))])
# B_N0[38] = np.nanmean(A_NM[np.where((A_N1>4.5)&(A_N1<=4.75))])
# B_N0[39] = np.nanmean(A_NM[np.where((A_N1>4.75)&(A_N1<=5))])


# #2015
# B_N0 = np.zeros((40))
# B_N0[0] = np.nanmean(A_NM[np.where((A_N1>-5)&(A_N1<=-4.75))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1>-4.75)&(A_N1<=-4.5))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1>-4.5)&(A_N1<=-4.25))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1>-4.25)&(A_N1<=-4))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1>-4)&(A_N1<=-3.75))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1>-3.75)&(A_N1<=-3.5))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1>-3.5)&(A_N1<=-3.25))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1>-3.25)&(A_N1<=-3))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1>-3)&(A_N1<=-2.75))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1>-2.75)&(A_N1<=-2.5))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1>-2.5)&(A_N1<=-2.25))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1>-2.25)&(A_N1<=-2))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1>-2)&(A_N1<=-1.75))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1>-1.75)&(A_N1<=-1.5))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1>-1.5)&(A_N1<=-1.25))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1>-1.25)&(A_N1<=-1))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1>-1)&(A_N1<=-0.75))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1>-0.75)&(A_N1<=-0.5))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1>-0.5)&(A_N1<=-0.25))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1>-0.25)&(A_N1<=0))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1>0)&(A_N1<=0.25))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1>0.25)&(A_N1<=0.5))])
# B_N0[22] = np.nanmean(A_NM[np.where((A_N1>0.5)&(A_N1<=0.75))])
# B_N0[23] = np.nanmean(A_NM[np.where((A_N1>0.75)&(A_N1<=1))])
# B_N0[24] = np.nanmean(A_NM[np.where((A_N1>1)&(A_N1<=1.25))])
# B_N0[25] = np.nanmean(A_NM[np.where((A_N1>1.25)&(A_N1<=1.5))])
# B_N0[26] = np.nanmean(A_NM[np.where((A_N1>1.5)&(A_N1<=1.75))])
# B_N0[27] = np.nanmean(A_NM[np.where((A_N1>1.75)&(A_N1<=2))])
# B_N0[28] = np.nanmean(A_NM[np.where((A_N1>2)&(A_N1<=2.25))])
# B_N0[29] = np.nanmean(A_NM[np.where((A_N1>2.25)&(A_N1<=2.5))])
# B_N0[30] = np.nanmean(A_NM[np.where((A_N1>2.5)&(A_N1<=2.75))])
# B_N0[31] = np.nanmean(A_NM[np.where((A_N1>2.75)&(A_N1<=3))])
# B_N0[32] = np.nanmean(A_NM[np.where((A_N1>3)&(A_N1<=3.25))])
# B_N0[33] = np.nanmean(A_NM[np.where((A_N1>3.25)&(A_N1<=3.5))])
# B_N0[34] = np.nanmean(A_NM[np.where((A_N1>3.5)&(A_N1<=3.75))])
# B_N0[35] = np.nanmean(A_NM[np.where((A_N1>3.75)&(A_N1<=4))])
# B_N0[36] = np.nanmean(A_NM[np.where((A_N1>4)&(A_N1<=4.25))])
# B_N0[37] = np.nanmean(A_NM[np.where((A_N1>4.25)&(A_N1<=4.5))])
# B_N0[38] = np.nanmean(A_NM[np.where((A_N1>4.5)&(A_N1<=4.75))])
# B_N0[39] = np.nanmean(A_NM[np.where((A_N1>4.75)&(A_N1<=5))])


# #2016
# B_N0 = np.zeros((40))
# B_N0[0] = np.nanmean(A_NM[np.where((A_N1>-5)&(A_N1<=-4.75))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1>-4.75)&(A_N1<=-4.5))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1>-4.5)&(A_N1<=-4.25))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1>-4.25)&(A_N1<=-4))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1>-4)&(A_N1<=-3.75))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1>-3.75)&(A_N1<=-3.5))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1>-3.5)&(A_N1<=-3.25))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1>-3.25)&(A_N1<=-3))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1>-3)&(A_N1<=-2.75))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1>-2.75)&(A_N1<=-2.5))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1>-2.5)&(A_N1<=-2.25))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1>-2.25)&(A_N1<=-2))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1>-2)&(A_N1<=-1.75))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1>-1.75)&(A_N1<=-1.5))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1>-1.5)&(A_N1<=-1.25))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1>-1.25)&(A_N1<=-1))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1>-1)&(A_N1<=-0.75))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1>-0.75)&(A_N1<=-0.5))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1>-0.5)&(A_N1<=-0.25))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1>-0.25)&(A_N1<=0))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1>0)&(A_N1<=0.25))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1>0.25)&(A_N1<=0.5))])
# B_N0[22] = np.nanmean(A_NM[np.where((A_N1>0.5)&(A_N1<=0.75))])
# B_N0[23] = np.nanmean(A_NM[np.where((A_N1>0.75)&(A_N1<=1))])
# B_N0[24] = np.nanmean(A_NM[np.where((A_N1>1)&(A_N1<=1.25))])
# B_N0[25] = np.nanmean(A_NM[np.where((A_N1>1.25)&(A_N1<=1.5))])
# B_N0[26] = np.nanmean(A_NM[np.where((A_N1>1.5)&(A_N1<=1.75))])
# B_N0[27] = np.nanmean(A_NM[np.where((A_N1>1.75)&(A_N1<=2))])
# B_N0[28] = np.nanmean(A_NM[np.where((A_N1>2)&(A_N1<=2.25))])
# B_N0[29] = np.nanmean(A_NM[np.where((A_N1>2.25)&(A_N1<=2.5))])
# B_N0[30] = np.nanmean(A_NM[np.where((A_N1>2.5)&(A_N1<=2.75))])
# B_N0[31] = np.nanmean(A_NM[np.where((A_N1>2.75)&(A_N1<=3))])
# B_N0[32] = np.nanmean(A_NM[np.where((A_N1>3)&(A_N1<=3.25))])
# B_N0[33] = np.nanmean(A_NM[np.where((A_N1>3.25)&(A_N1<=3.5))])
# B_N0[34] = np.nanmean(A_NM[np.where((A_N1>3.5)&(A_N1<=3.75))])
# B_N0[35] = np.nanmean(A_NM[np.where((A_N1>3.75)&(A_N1<=4))])
# B_N0[36] = np.nanmean(A_NM[np.where((A_N1>4)&(A_N1<=4.25))])
# B_N0[37] = np.nanmean(A_NM[np.where((A_N1>4.25)&(A_N1<=4.5))])
# B_N0[38] = np.nanmean(A_NM[np.where((A_N1>4.5)&(A_N1<=4.75))])
# B_N0[39] = np.nanmean(A_NM[np.where((A_N1>4.75)&(A_N1<=5))])

# #2017
# B_N0 = np.zeros((40))
# B_N0[0] = np.nanmean(A_NM[np.where((A_N1>-5)&(A_N1<=-4.75))])
# B_N0[1] = np.nanmean(A_NM[np.where((A_N1>-4.75)&(A_N1<=-4.5))])
# B_N0[2] = np.nanmean(A_NM[np.where((A_N1>-4.5)&(A_N1<=-4.25))])
# B_N0[3] = np.nanmean(A_NM[np.where((A_N1>-4.25)&(A_N1<=-4))])
# B_N0[4] = np.nanmean(A_NM[np.where((A_N1>-4)&(A_N1<=-3.75))])
# B_N0[5] = np.nanmean(A_NM[np.where((A_N1>-3.75)&(A_N1<=-3.5))])
# B_N0[6] = np.nanmean(A_NM[np.where((A_N1>-3.5)&(A_N1<=-3.25))])
# B_N0[7] = np.nanmean(A_NM[np.where((A_N1>-3.25)&(A_N1<=-3))])
# B_N0[8] = np.nanmean(A_NM[np.where((A_N1>-3)&(A_N1<=-2.75))])
# B_N0[9] = np.nanmean(A_NM[np.where((A_N1>-2.75)&(A_N1<=-2.5))])
# B_N0[10] = np.nanmean(A_NM[np.where((A_N1>-2.5)&(A_N1<=-2.25))])
# B_N0[11] = np.nanmean(A_NM[np.where((A_N1>-2.25)&(A_N1<=-2))])
# B_N0[12] = np.nanmean(A_NM[np.where((A_N1>-2)&(A_N1<=-1.75))])
# B_N0[13] = np.nanmean(A_NM[np.where((A_N1>-1.75)&(A_N1<=-1.5))])
# B_N0[14] = np.nanmean(A_NM[np.where((A_N1>-1.5)&(A_N1<=-1.25))])
# B_N0[15] = np.nanmean(A_NM[np.where((A_N1>-1.25)&(A_N1<=-1))])
# B_N0[16] = np.nanmean(A_NM[np.where((A_N1>-1)&(A_N1<=-0.75))])
# B_N0[17] = np.nanmean(A_NM[np.where((A_N1>-0.75)&(A_N1<=-0.5))])
# B_N0[18] = np.nanmean(A_NM[np.where((A_N1>-0.5)&(A_N1<=-0.25))])
# B_N0[19] = np.nanmean(A_NM[np.where((A_N1>-0.25)&(A_N1<=0))])
# B_N0[20] = np.nanmean(A_NM[np.where((A_N1>0)&(A_N1<=0.25))])
# B_N0[21] = np.nanmean(A_NM[np.where((A_N1>0.25)&(A_N1<=0.5))])
# B_N0[22] = np.nanmean(A_NM[np.where((A_N1>0.5)&(A_N1<=0.75))])
# B_N0[23] = np.nanmean(A_NM[np.where((A_N1>0.75)&(A_N1<=1))])
# B_N0[24] = np.nanmean(A_NM[np.where((A_N1>1)&(A_N1<=1.25))])
# B_N0[25] = np.nanmean(A_NM[np.where((A_N1>1.25)&(A_N1<=1.5))])
# B_N0[26] = np.nanmean(A_NM[np.where((A_N1>1.5)&(A_N1<=1.75))])
# B_N0[27] = np.nanmean(A_NM[np.where((A_N1>1.75)&(A_N1<=2))])
# B_N0[28] = np.nanmean(A_NM[np.where((A_N1>2)&(A_N1<=2.25))])
# B_N0[29] = np.nanmean(A_NM[np.where((A_N1>2.25)&(A_N1<=2.5))])
# B_N0[30] = np.nanmean(A_NM[np.where((A_N1>2.5)&(A_N1<=2.75))])
# B_N0[31] = np.nanmean(A_NM[np.where((A_N1>2.75)&(A_N1<=3))])
# B_N0[32] = np.nanmean(A_NM[np.where((A_N1>3)&(A_N1<=3.25))])
# B_N0[33] = np.nanmean(A_NM[np.where((A_N1>3.25)&(A_N1<=3.5))])
# B_N0[34] = np.nanmean(A_NM[np.where((A_N1>3.5)&(A_N1<=3.75))])
# B_N0[35] = np.nanmean(A_NM[np.where((A_N1>3.75)&(A_N1<=4))])
# B_N0[36] = np.nanmean(A_NM[np.where((A_N1>4)&(A_N1<=4.25))])
# B_N0[37] = np.nanmean(A_NM[np.where((A_N1>4.25)&(A_N1<=4.5))])
# B_N0[38] = np.nanmean(A_NM[np.where((A_N1>4.5)&(A_N1<=4.75))])
# B_N0[39] = np.nanmean(A_NM[np.where((A_N1>4.75)&(A_N1<=5))])

#2018
B_N8 = np.zeros((40))
B_N8[0] = np.nanmean(A_N101[np.where((A_N111>-5)&(A_N111<=-4.75))])
B_N8[1] = np.nanmean(A_N101[np.where((A_N111>-4.75)&(A_N111<=-4.5))])
B_N8[2] = np.nanmean(A_N101[np.where((A_N111>-4.5)&(A_N111<=-4.25))])
B_N8[3] = np.nanmean(A_N101[np.where((A_N111>-4.25)&(A_N111<=-4))])
B_N8[4] = np.nanmean(A_N101[np.where((A_N111>-4)&(A_N111<=-3.75))])
B_N8[5] = np.nanmean(A_N101[np.where((A_N111>-3.75)&(A_N111<=-3.5))])
B_N8[6] = np.nanmean(A_N101[np.where((A_N111>-3.5)&(A_N111<=-3.25))])
B_N8[7] = np.nanmean(A_N101[np.where((A_N111>-3.25)&(A_N111<=-3))])
B_N8[8] = np.nanmean(A_N101[np.where((A_N111>-3)&(A_N111<=-2.75))])
B_N8[9] = np.nanmean(A_N101[np.where((A_N111>-2.75)&(A_N111<=-2.5))])
B_N8[10] = np.nanmean(A_N101[np.where((A_N111>-2.5)&(A_N111<=-2.25))])
B_N8[11] = np.nanmean(A_N101[np.where((A_N111>-2.25)&(A_N111<=-2))])
B_N8[12] = np.nanmean(A_N101[np.where((A_N111>-2)&(A_N111<=-1.75))])
B_N8[13] = np.nanmean(A_N101[np.where((A_N111>-1.75)&(A_N111<=-1.5))])
B_N8[14] = np.nanmean(A_N101[np.where((A_N111>-1.5)&(A_N111<=-1.25))])
B_N8[15] = np.nanmean(A_N101[np.where((A_N111>-1.25)&(A_N111<=-1))])
B_N8[16] = np.nanmean(A_N101[np.where((A_N111>-1)&(A_N111<=-0.75))])
B_N8[17] = np.nanmean(A_N101[np.where((A_N111>-0.75)&(A_N111<=-0.5))])
B_N8[18] = np.nanmean(A_N101[np.where((A_N111>-0.5)&(A_N111<=-0.25))])
B_N8[19] = np.nanmean(A_N101[np.where((A_N111>-0.25)&(A_N111<=0))])
B_N8[20] = np.nanmean(A_N101[np.where((A_N111>0)&(A_N111<=0.25))])
B_N8[21] = np.nanmean(A_N101[np.where((A_N111>0.25)&(A_N111<=0.5))])
B_N8[22] = np.nanmean(A_N101[np.where((A_N111>0.5)&(A_N111<=0.75))])
B_N8[23] = np.nanmean(A_N101[np.where((A_N111>0.75)&(A_N111<=1))])
B_N8[24] = np.nanmean(A_N101[np.where((A_N111>1)&(A_N111<=1.25))])
B_N8[25] = np.nanmean(A_N101[np.where((A_N111>1.25)&(A_N111<=1.5))])
B_N8[26] = np.nanmean(A_N101[np.where((A_N111>1.5)&(A_N111<=1.75))])
B_N8[27] = np.nanmean(A_N101[np.where((A_N111>1.75)&(A_N111<=2))])
B_N8[28] = np.nanmean(A_N101[np.where((A_N111>2)&(A_N111<=2.25))])
B_N8[29] = np.nanmean(A_N101[np.where((A_N111>2.25)&(A_N111<=2.5))])
B_N8[30] = np.nanmean(A_N101[np.where((A_N111>2.5)&(A_N111<=2.75))])
B_N8[31] = np.nanmean(A_N101[np.where((A_N111>2.75)&(A_N111<=3))])
B_N8[32] = np.nanmean(A_N101[np.where((A_N111>3)&(A_N111<=3.25))])
B_N8[33] = np.nanmean(A_N101[np.where((A_N111>3.25)&(A_N111<=3.5))])
B_N8[34] = np.nanmean(A_N101[np.where((A_N111>3.5)&(A_N111<=3.75))])
B_N8[35] = np.nanmean(A_N101[np.where((A_N111>3.75)&(A_N111<=4))])
B_N8[36] = np.nanmean(A_N101[np.where((A_N111>4)&(A_N111<=4.25))])
B_N8[37] = np.nanmean(A_N101[np.where((A_N111>4.25)&(A_N111<=4.5))])
B_N8[38] = np.nanmean(A_N101[np.where((A_N111>4.5)&(A_N111<=4.75))])
B_N8[39] = np.nanmean(A_N101[np.where((A_N111>4.75)&(A_N111<=5))])

#2019
B_N9 = np.zeros((40))
B_N8[0] = np.nanmean(A_N102[np.where((A_N112>-5)&(A_N112<=-4.75))])
B_N9[1] = np.nanmean(A_N102[np.where((A_N112>-4.75)&(A_N112<=-4.5))])
B_N9[2] = np.nanmean(A_N102[np.where((A_N112>-4.5)&(A_N112<=-4.25))])
B_N9[3] = np.nanmean(A_N102[np.where((A_N112>-4.25)&(A_N112<=-4))])
B_N9[4] = np.nanmean(A_N102[np.where((A_N112>-4)&(A_N112<=-3.75))])
B_N9[5] = np.nanmean(A_N102[np.where((A_N112>-3.75)&(A_N112<=-3.5))])
B_N9[6] = np.nanmean(A_N102[np.where((A_N112>-3.5)&(A_N112<=-3.25))])
B_N9[7] = np.nanmean(A_N102[np.where((A_N112>-3.25)&(A_N112<=-3))])
B_N9[8] = np.nanmean(A_N102[np.where((A_N112>-3)&(A_N112<=-2.75))])
B_N9[9] = np.nanmean(A_N102[np.where((A_N112>-2.75)&(A_N112<=-2.5))])
B_N9[10] = np.nanmean(A_N102[np.where((A_N112>-2.5)&(A_N112<=-2.25))])
B_N9[11] = np.nanmean(A_N102[np.where((A_N112>-2.25)&(A_N112<=-2))])
B_N9[12] = np.nanmean(A_N102[np.where((A_N112>-2)&(A_N112<=-1.75))])
B_N9[13] = np.nanmean(A_N102[np.where((A_N112>-1.75)&(A_N112<=-1.5))])
B_N9[14] = np.nanmean(A_N102[np.where((A_N112>-1.5)&(A_N112<=-1.25))])
B_N9[15] = np.nanmean(A_N102[np.where((A_N112>-1.25)&(A_N112<=-1))])
B_N9[16] = np.nanmean(A_N102[np.where((A_N112>-1)&(A_N112<=-0.75))])
B_N9[17] = np.nanmean(A_N102[np.where((A_N112>-0.75)&(A_N112<=-0.5))])
B_N9[18] = np.nanmean(A_N102[np.where((A_N112>-0.5)&(A_N112<=-0.25))])
B_N9[19] = np.nanmean(A_N102[np.where((A_N112>-0.25)&(A_N112<=0))])
B_N9[20] = np.nanmean(A_N102[np.where((A_N112>0)&(A_N112<=0.25))])
B_N9[21] = np.nanmean(A_N102[np.where((A_N112>0.25)&(A_N112<=0.5))])
B_N9[22] = np.nanmean(A_N102[np.where((A_N112>0.5)&(A_N112<=0.75))])
B_N9[23] = np.nanmean(A_N102[np.where((A_N112>0.75)&(A_N112<=1))])
B_N9[24] = np.nanmean(A_N102[np.where((A_N112>1)&(A_N112<=1.25))])
B_N9[25] = np.nanmean(A_N102[np.where((A_N112>1.25)&(A_N112<=1.5))])
B_N9[26] = np.nanmean(A_N102[np.where((A_N112>1.5)&(A_N112<=1.75))])
B_N9[27] = np.nanmean(A_N102[np.where((A_N112>1.75)&(A_N112<=2))])
B_N9[28] = np.nanmean(A_N102[np.where((A_N112>2)&(A_N112<=2.25))])
B_N9[29] = np.nanmean(A_N102[np.where((A_N112>2.25)&(A_N112<=2.5))])
B_N9[30] = np.nanmean(A_N102[np.where((A_N112>2.5)&(A_N112<=2.75))])
B_N9[31] = np.nanmean(A_N102[np.where((A_N112>2.75)&(A_N112<=3))])
B_N9[32] = np.nanmean(A_N102[np.where((A_N112>3)&(A_N112<=3.25))])
B_N9[33] = np.nanmean(A_N102[np.where((A_N112>3.25)&(A_N112<=3.5))])
B_N9[34] = np.nanmean(A_N102[np.where((A_N112>3.5)&(A_N112<=3.75))])
B_N9[35] = np.nanmean(A_N102[np.where((A_N112>3.75)&(A_N112<=4))])
B_N9[36] = np.nanmean(A_N102[np.where((A_N112>4)&(A_N112<=4.25))])
B_N9[37] = np.nanmean(A_N102[np.where((A_N112>4.25)&(A_N112<=4.5))])
B_N9[38] = np.nanmean(A_N102[np.where((A_N112>4.5)&(A_N112<=4.75))])
B_N9[39] = np.nanmean(A_N102[np.where((A_N112>4.75)&(A_N112<=5))])

#2020
B_N10 = np.zeros((40))
B_N10[0] = np.nanmean(A_N103[np.where((A_N113>-5)&(A_N113<=-4.75))])
B_N10[1] = np.nanmean(A_N103[np.where((A_N113>-4.75)&(A_N113<=-4.5))])
B_N10[2] = np.nanmean(A_N103[np.where((A_N113>-4.5)&(A_N113<=-4.25))])
B_N10[3] = np.nanmean(A_N103[np.where((A_N113>-4.25)&(A_N113<=-4))])
B_N10[4] = np.nanmean(A_N103[np.where((A_N113>-4)&(A_N113<=-3.75))])
B_N10[5] = np.nanmean(A_N103[np.where((A_N113>-3.75)&(A_N113<=-3.5))])
B_N10[6] = np.nanmean(A_N103[np.where((A_N113>-3.5)&(A_N113<=-3.25))])
B_N10[7] = np.nanmean(A_N103[np.where((A_N113>-3.25)&(A_N113<=-3))])
B_N10[8] = np.nanmean(A_N103[np.where((A_N113>-3)&(A_N113<=-2.75))])
B_N10[9] = np.nanmean(A_N103[np.where((A_N113>-2.75)&(A_N113<=-2.5))])
B_N10[10] = np.nanmean(A_N103[np.where((A_N113>-2.5)&(A_N113<=-2.25))])
B_N10[11] = np.nanmean(A_N103[np.where((A_N113>-2.25)&(A_N113<=-2))])
B_N10[12] = np.nanmean(A_N103[np.where((A_N113>-2)&(A_N113<=-1.75))])
B_N10[13] = np.nanmean(A_N103[np.where((A_N113>-1.75)&(A_N113<=-1.5))])
B_N10[14] = np.nanmean(A_N103[np.where((A_N113>-1.5)&(A_N113<=-1.25))])
B_N10[15] = np.nanmean(A_N103[np.where((A_N113>-1.25)&(A_N113<=-1))])
B_N10[16] = np.nanmean(A_N103[np.where((A_N113>-1)&(A_N113<=-0.75))])
B_N10[17] = np.nanmean(A_N103[np.where((A_N113>-0.75)&(A_N113<=-0.5))])
B_N10[18] = np.nanmean(A_N103[np.where((A_N113>-0.5)&(A_N113<=-0.25))])
B_N10[19] = np.nanmean(A_N103[np.where((A_N113>-0.25)&(A_N113<=0))])
B_N10[20] = np.nanmean(A_N103[np.where((A_N113>0)&(A_N113<=0.25))])
B_N10[21] = np.nanmean(A_N103[np.where((A_N113>0.25)&(A_N113<=0.5))])
B_N10[22] = np.nanmean(A_N103[np.where((A_N113>0.5)&(A_N113<=0.75))])
B_N10[23] = np.nanmean(A_N103[np.where((A_N113>0.75)&(A_N113<=1))])
B_N10[24] = np.nanmean(A_N103[np.where((A_N113>1)&(A_N113<=1.25))])
B_N10[25] = np.nanmean(A_N103[np.where((A_N113>1.25)&(A_N113<=1.5))])
B_N10[26] = np.nanmean(A_N103[np.where((A_N113>1.5)&(A_N113<=1.75))])
B_N10[27] = np.nanmean(A_N103[np.where((A_N113>1.75)&(A_N113<=2))])
B_N10[28] = np.nanmean(A_N103[np.where((A_N113>2)&(A_N113<=2.25))])
B_N10[29] = np.nanmean(A_N103[np.where((A_N113>2.25)&(A_N113<=2.5))])
B_N10[30] = np.nanmean(A_N103[np.where((A_N113>2.5)&(A_N113<=2.75))])
B_N10[31] = np.nanmean(A_N103[np.where((A_N113>2.75)&(A_N113<=3))])
B_N10[32] = np.nanmean(A_N103[np.where((A_N113>3)&(A_N113<=3.25))])
B_N10[33] = np.nanmean(A_N103[np.where((A_N113>3.25)&(A_N113<=3.5))])
B_N10[34] = np.nanmean(A_N103[np.where((A_N113>3.5)&(A_N113<=3.75))])
B_N10[35] = np.nanmean(A_N103[np.where((A_N113>3.75)&(A_N113<=4))])
B_N10[36] = np.nanmean(A_N103[np.where((A_N113>4)&(A_N113<=4.25))])
B_N10[37] = np.nanmean(A_N103[np.where((A_N113>4.25)&(A_N113<=4.5))])
B_N10[38] = np.nanmean(A_N103[np.where((A_N113>4.5)&(A_N113<=4.75))])
B_N10[39] = np.nanmean(A_N103[np.where((A_N113>4.75)&(A_N113<=5))])


PCA = np.arange(-5,5,0.25)
# plt.plot(PCA, B_N,label='2010')
plt.plot(PCA, B_N0,color='blue', label='2011-2019')
# plt.plot(PCA, B_N1,label='2011')
# plt.plot(PCA, B_N2,label='2012')
# plt.plot(PCA, B_N3,label='2013')
# plt.plot(PCA, B_N4,label='2014')
# plt.plot(PCA, B_N5,label='2015')
# plt.plot(PCA, B_N6,label='2016')
# plt.plot(PCA, B_N7,label='2017')
# plt.plot(PCA, B_N8,color='black',label='2018')
plt.plot(PCA, B_N9,color='green', label='2019')
plt.plot(PCA, B_N10,color='red',label='2020')

plt.legend() 
plt.title("EOF-CERES_Cldarea", fontsize=18) 
plt.xlabel("EOF", fontsize=14)
plt.ylabel("CERES_Cldarea(%)", fontsize=14) 

################################################################################

# D_N = np.zeros((180,360))
# C_N = np.zeros((2,308,180,360))
# A_N1 = A_N1.reshape(308,180,360)
# A_N10 = A_N10.reshape(308,180,360)
# A_N11 = A_N11.reshape(308,180,360)
# A_N12 = A_N12.reshape(308,180,360)
# A_N20 = A_N20.reshape(308,180,360)

# C_N[0,:,:,:] = A_N1[:,:,:]
# C_N[1,:,:,:] = A_N20[:,:,:]

# # ds = xr.Dataset({'PCA': (('Time','Latitude','Longitude'), A_N1[:,:,:]),
# #                   'cldarea':(('Time','Latitude','Longitude'),A_N20[:,:,:]),
# #                   },
# #                 coords={'lat': ('Latitude', np.arange(0,81,1)),
# #                         'lon': ('Longitude', np.arange(40,181,1)),
# #                         'time': ('Time', np.arange(0,308,1)),
# #                         })
# # ds.to_netcdf('G:\\PCA_cldarea_daily.nc')

# for i in range(0,81):
#     for j in range(0,141):
#         D_N[i,j] = pd.Series(C_N[0,:,i,j]).corr(pd.Series(C_N[1,:,i,j]), method='pearson')

# A_N20 = A_N20.reshape(11,28,180,360)

# lon = np.linspace(0,360,360)
# lat = np.linspace(-90,90,180)

# fig, ax = plt.subplots(figsize=(10,10),constrained_layout=True,dpi=200)
# plt.rc('font', size=10, weight='bold')

# #basemap设置部分
# map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=00,urcrnrlon=360,resolution='l')  
# parallels = np.arange(-90,90+30,30) #纬线
# map.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.01,dashes=[1,400])
# ax.set_yticks(parallels,len(parallels)*[''])
# meridians = np.arange(0,360+60,60) #经线
# map.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.01,dashes=[1,400])
# ax.set_xticks(meridians,len(meridians)*[''])

# map.drawcountries(linewidth=1.5)
# map.drawcoastlines()
# lon, lat = np.meshgrid(lon, lat)
# cmap=dcmap('F://color/test6.txt')
# cmap.set_bad('gray')
# a=map.pcolormesh(lon, lat,DDD,cmap=cmap)
# fig.colorbar(a)
# ax.set_title('2019 minus 2020 cldarea(EOF>0)',size=16)

# plt.show()


