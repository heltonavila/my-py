import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from math import sqrt
from pandas.tools.plotting import autocorrelation_plot

def gibbs_normal_bi(chute=[1,1], N=1000, RA=1298259):

    ro = int(str(RA)[-2:]) / 100
    valores = {"x":[chute[1]],"y":[chute[1]]}

    for i in range(N):
        valores["x"].append( np.random.normal(loc=valores["y"][-1]*ro , scale=sqrt(1-(ro**2)) ) )
        valores["y"].append( np.random.normal(loc=valores["x"][-1]*ro , scale=sqrt(1-(ro**2)) ) )

    return valores

def faz_plots(valores):

    nullfmt = NullFormatter()         # no labels
    plt.style.use('bmh')

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    x = valores["x"]
    y = valores["y"]
    axScatter.scatter(x,y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins, histtype="stepfilled", alpha=0.8, normed=True)
    axHisty.hist(y, bins=bins, orientation='horizontal', histtype="stepfilled", alpha=0.8, normed=True)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    plt.show()

    ####  GRÁFICO TEMPORAL ####
    ax1 = plt.subplot(211)
    plt.plot(x)
    ax1.set_title("Gráfico de X")
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    ax2 = plt.subplot(212)
    plt.plot(y)
    ax2.set_title("Gráfico de Y")
    plt.setp(ax2.get_xticklabels(), fontsize=6)
    plt.show()

    #### GRÁFICO DE AUTOCORRELAÇÃO
    autocorrelation_plot(x)
    plt.title("Autocorrelação de X")
    plt.show()

    autocorrelation_plot(y)
    plt.title("Autocorrelação de Y")
    plt.show()

    return None
