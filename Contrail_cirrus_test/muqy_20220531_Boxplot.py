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


class Box_plot(object):
    """
    Plot boxplot of Cld data match each PC1 interval

    """

    def __init__(self, Cld_match_PC_gap):
        """
        Initialize the class

        Parameters
        ----------
        Cld_match_PC_gap : Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
        """
        # Input array must be in shape of (PC1_gap, lat, lon)
        self.Cld_match_PC_gap = Cld_match_PC_gap

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
                self.Cld_match_PC_gap.shape[1]
                * self.Cld_match_PC_gap.shape[2]
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
            marker="o",
            markersize=7,
            markeredgecolor="grey",
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
            fontsize=26,
            weight="bold",
        )
        # plt.savefig(
        #     "Box_plot_PC1_Cld.png",
        #     dpi=500,
        #     facecolor=fig.get_facecolor(),
        #     edgecolor="none",
        # )
        plt.show()


# Cld_2020_match_PC_gap is the example data
box_plot = Box_plot(Cld_2020_match_PC_gap)
box_plot.plot_box_plot()
