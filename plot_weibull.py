from scipy.stats import norm, gamma
import time, datetime
import numpy as np, pandas as pd

from simpoe.corrosion import failureStress, corrLimitStates
from simpoe import unpacker, model_constants, distributer, cgr

pd.set_option('display.max_columns', 500)

import matplotlib.pyplot as plt
import mpld3
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams["figure.figsize"] = (10,10)

shape, scale = 2.253161414, 0.011928564
bincount = 75

weibullCGR = pd.DataFrame(cgr.cgr_weibull(np.random.rand(10000),shp=shape,scl=scale), columns=['Weibull Distribution'])


fig, axs = plt.subplots(1, sharex='all', figsize=(10,5), gridspec_kw={'hspace':0.5})

weibullCGR.mask(lambda x: x.loc[:,'Weibull Distribution'] > 0.40, 0.40000).plot.kde(grid=True, ax=axs)
weibullCGR.mask(lambda x: x.loc[:,'Weibull Distribution'] > 0.40, 0.40000).plot.hist(bins=bincount, grid=True, ax=axs, color='blue', alpha=0.5, xlim=[0,0.12], label='Weibull')

##for i, ax in enumerate(axs.reshape(-1)):
##    weibullCGR.mask(lambda x: x.loc[:,'Weibull Distribution'] > 0.40, 0.40000).plot.kde(grid=True, ax=ax)
##    weibullCGR.mask(lambda x: x.loc[:,'Weibull Distribution'] > 0.40, 0.40000).plot.hist(bins=bincount, grid=True, ax=ax, color='blue', alpha=0.5, xlim=[0,0.12], label='Weibull')

##weibullCGR.plot.kde(grid=True)
axs.annotate(f"shp: {shape}\nscl: {scale}", xy=(weibullCGR.max().values[0],
                                                axs.get_ylim()[1]*0.75))
plt.show()
