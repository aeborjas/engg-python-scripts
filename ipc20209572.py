import numpy as np, pandas as pd, os, sys
from typing import NewType
from scipy.stats import norm
from scipy.stats import lognorm
import matplotlib.pyplot as plt

import win32clipboard as clipboard

def toClipboardForExcel(array):
    """
    Copies an array into a string format acceptable by Excel.
    Columns separated by \t, rows separated by \n
    """
    # Create string from array
##    line_strings = []
##    for line in array:
##        line_strings.append("\t".join(line.astype(str)).replace("\n",""))
    array_string = "\r\n".join(array.astype(str))

    # Put string into clipboard (open, clear, set, close)
    clipboard.OpenClipboard()
    clipboard.EmptyClipboard()
    clipboard.SetClipboardText(array_string, clipboard.CF_TEXT)
    clipboard.CloseClipboard()

##mmpyr = NewType('mmpyr', float)

mpy_to_mmpyr = (1.0/1000.0)*25.4

def cgr_weibull(p,shp=1.6,scl=0.1):
    '''
    Returns the imperfection growth rate based on a Weibull distribution inverse cumulative density function, using shape and scale parameters.
    Requires Numpy module
    :param p:   random number between 0 and 1
    :param shp: shape parameter, default of 1.6
    :param scl: scale parameter, default of 0.1
    :return:    imperfection growth rate in mm/yr
    '''
    return  scl*np.power( -np.log(1.0-p) ,1.0/shp)

def cgr_lognorm(p, mu= 2.1, median=1.7):
    scale, shape = median, np.sqrt(2.0*np.log(mu))
    dist = lognorm(shape, loc=0, scale=scale)
    return dist.ppf(p)*mpy_to_mmpyr

def pod(d, q=np.log(1.0 - 0.9)/-15):
    return 1-np.exp(-q*d)

def monte_carlo(n, cgr_func=cgr_weibull):

    wt = 6.02
    md2 = 24.0
    sigma2 = 7.80
    sigma1 = 7.80
    reportable_t = 10.0
    tinsp = 5.0

    #random number generators
    e2_n_1 = np.random.rand(n)
    g_n_2 = np.random.rand(n)
    u_n_3 = np.random.rand(n)
    e1_n_4 = np.random.rand(n)

    #measurement error
    e2 = norm.ppf(e2_n_1, loc=0, scale=sigma2)
    e1 = norm.ppf(e1_n_4, loc=0, scale=sigma1)

    #true depth at time 2
    td2 = md2 - e2

    #stochastic growth rate
##    g = cgr_weibull(g_n_2)
##    g = cgr_lognorm(g_n_2)
    g = cgr_func(g_n_2)

    td1 = td2 - (g*100./wt)*tinsp

    tinit = td2*wt/(100.*g)

    md1 = td1 + e1

    #limit state: td1 is negative, defect wouldn't have existed in previous inspection
    ls_1_td1_neg = (td1 < 0.0).astype(int)

    #limit state: defect is not new since the time of previous inspection
    ls_2_detection = (u_n_3 >= pod(td1)).astype(int)

    #limit state: measured defect at previous ILI smaller than reportable threshold
    ls_3_reportable = (md1 < reportable_t).astype(int)

    #comb limit state: new feature
    ls_new = ls_1_td1_neg.copy()

    #comb limit state: unreported feature
    ls_unreported = np.maximum.reduce([ls_1_td1_neg, ls_2_detection, ls_3_reportable])

    p_new = np.sum(ls_new, axis=0)/n
    p_unreported = np.sum(ls_unreported, axis=0)/n

    g_unreported = g[ls_unreported == 1]

    g_unreported_fit = lognorm.fit(g_unreported, floc=0.)

    g_unreported_dist = lognorm(g_unreported_fit[0], scale=g_unreported_fit[2])

    calc = result(locals())

    return calc

class result:
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
        setattr(self, '__name__', my_dict['cgr_func'].__name__)

    def plot_dist(self, n_points:int=1000, nbins:int=10, xlim:tuple=(0.0, 0.6)):
        fig = plt.figure()
        fig.suptitle(self.cgr_func.__name__)
        x = np.linspace(xlim[0], xlim[1], n_points)
        ax = plt.axes()
        ax.plot(x, lognorm.pdf(x, self.g_unreported_fit[0], scale=self.g_unreported_fit[2]), c='darkgreen')
        plt.grid(which='both')
        plt.legend([f'Lognormal\nshape={self.g_unreported_fit[0]:.4f}\nscale={self.g_unreported_fit[2]:.4f}'])
        ax.set_xlim(*xlim)
        ax2 = plt.gca().twinx()
        ax2.hist(self.g_unreported, bins=nbins, alpha=0.5)

    def report(self, output=False):
        df = pd.DataFrame(index=['wt (mm)',
                                 'PDP2 (%WT)',
                                 'Tool2 Std (%WT)',
                                 'Tool1 Std (%WT)',
                                 'Report. Thr (%WT)',
                                 'Insp. Interval (yr)',
                                 'p_new',
                                 'p_unreported'],
                   data=[self.wt,
                         self.md2,
                         self.sigma2,
                         self.sigma1,
                         self.reportable_t,
                         self.tinsp,
                         self.p_new,
                         self.p_unreported],
                   columns=[self.__name__])

        if output:
            print(df,sep='\n')
            
        return df
    
def cgr_mpy(q):
    return np.tile((4.),q.shape)*mpy_to_mmpyr

result_weibull = monte_carlo(1000000)
result_lognorm = monte_carlo(1000000, cgr_func=cgr_lognorm)
result_mpy = monte_carlo(1000000, cgr_func=cgr_mpy)

totals = result_weibull.report().join(result_lognorm.report()).join(result_mpy.report())

print(totals)

result_weibull.plot_dist(nbins=50)
result_lognorm.plot_dist(nbins=500)
plt.show()
