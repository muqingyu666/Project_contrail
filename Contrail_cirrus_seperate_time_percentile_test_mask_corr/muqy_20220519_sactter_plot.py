import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def scatter_plot_simulated_observed(
    data0, data1, test_xlabel, test_ylabel
):
    """
    draw a scatter plot with linear fit straight line and mark the slope and error of the line

    Parameters
    ----------
    data0 : numpy.ndarray
        data 0
    data1 : numpy.ndarray
        data 1
    """

    from scipy import stats

    x = np.array(data0.reshape(-1))
    y = np.array(data1.reshape(-1))

    x[np.isnan(x)] = 0
    y[np.isnan(y)] = 0

    fig, ax = plt.subplots(figsize=(12, 12))
    (
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
    ) = stats.linregress(x, y)
    x_fit = np.linspace(0, x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, "r-", lw=1, label="slope = %.3f" % slope)
    plt.scatter(x, y, alpha=0.5)
    ax.set(
        xlim=(data0.min(), 40),
        # xticks=np.arange(1, 8),
        ylim=(data1.min(), data1.max()),
        # yticks=np.arange(1, 8),
    )
    # plt.xlabel(r"$\mathrm{CERES\ TOA\ UP\ (W/m^2)}$", fontsize=24)
    # plt.ylabel(r"$\mathrm{SBDART\ TOA\ UP\ (W/m^2)}$", fontsize=24)
    plt.xlabel(test_xlabel, fontsize=24)
    plt.ylabel(test_ylabel, fontsize=24)
    # mark the slope and error of the line
    plt.annotate(
        "R = %.3f" % r_value,
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        fontsize=24,
    )
    plt.annotate(
        "slope = %.3f" % slope,
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        fontsize=24,
    )
    ax.tick_params(labelsize=24)
    # plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    pass
