import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider

# fig, ax = plt.subplots(1,1)

def plot_weibull(**kwargs):
    fig, ax = plt.subplots(1,1, sharex='all', sharey='all')

    for i, (lbl, params) in enumerate(kwargs.items()):
        shape, scale = params[0], params[1]

        x = np.linspace(stats.weibull_min.ppf(0.0, shape, scale=scale),
                        stats.weibull_min.ppf(0.9999, shape, scale=scale), 1000)
        
        HSV = (np.random.random(),1,1)
        rgb = hsv_to_rgb(HSV)
        if len(kwargs) == 1:
            ax.plot(x, stats.weibull_min.pdf(x, shape, scale=scale), 
                    c=rgb, 
                    lw=3, 
                    alpha=0.6)
            ax.set_title(f"{lbl} - SHAPE:{shape}, SCALE:{scale}")
            ax.set_xlim(0.00, x[-1])
        else:
            ax.plot(x, stats.weibull_min.pdf(x, shape, scale=scale), 
                    c=rgb, 
                    lw=3, 
                    alpha=0.6,
                    label=lbl)
            # ax.set_title(f"{lbl} - SHAPE:{shape}, SCALE:{scale}")
            ax.set_xlim(0.00, 0.4)
            ax.set_ylim(0.00, 15)
            ax.legend()

    fig.subplots_adjust(hspace=0.5)
    plt.show()


def plot_rayleigh(**kwargs):
    fig, ax = plt.subplots(len(kwargs),1, sharex='all', sharey='all')

    for i, (lbl, params) in enumerate(kwargs.items()):
        mean, stdev = params[0], params[1]

        x = np.linspace(stats.rayleigh.ppf(0.0, loc=mean, scale=stdev),
                        stats.rayleigh.ppf(0.9999, loc=mean, scale=stdev), 1000)
        if len(kwargs) == 1:
            ax.plot(x, stats.rayleigh.pdf(x, loc=mean, scale=stdev), 
                    'r-', 
                    lw=3, 
                    alpha=0.6)
            ax.set_title(f"{lbl} - MEAN:{a}, STDEV:{stdev}")
            ax.set_xlim(0.00, x[-1])
        else:
            ax[i].plot(x, stats.rayleigh.pdf(x, loc=mean, scale=stdev), 
                    'r-', 
                    lw=3, 
                    alpha=0.6,
                    label=lbl)
            ax[i].set_title(f"{lbl} - MEAN:{mean}, STDEV:{stdev}")
            ax[i].set_xlim(0.00, x[-1])

    fig.subplots_adjust(hspace=0.5)
    plt.show()

# #instantiate an object X using the above four parameters,
# X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

# #generate 1000 sample data
# samples = X.rvs(1000)

# #compute the PDF of the sample data
# pdf_probs = stats.truncnorm.pdf(samples, (lower-mu)/sigma, (upper-mu)/sigma, mu, sigma)

# #plot the PDF curves 
# plt.plot(samples[samples.argsort()],pdf_probs[samples.argsort()],linewidth=2.3,label='PDF curve')

# plot_weibull(CSA_10pct=[1.24959, 0.107359*1.1 + 0.3*(0.10-0.2)],
#             CSA_20pct=[1.24959, 0.107359*1.1 + 0.3*(0.20-0.2)],
#             CSA_50pct=[1.24959, 0.107359*1.1 + 0.3*(0.50-0.2)])

# plot_weibull(CSA=[1.24959, 0.107359 + 0.30 * ((0.3-0.2))],
#             SHP_1_2=[1.2, 0.107359 + 0.30 * ((0.30-0.2))],
#             SHP_1_1=[1.1, 0.107359 + 0.30 * ((0.30-0.2))],
#             SHP_1_0=[1.0, 0.107359 + 0.30 * ((0.30-0.2))],
#             SHP_0_8=[0.8, 0.107359 + 0.30 * ((0.30-0.2))])

# plot_weibull(CSA=[1.24959, 0.107359],
#              WEIB=[1.439, 0.1])

fig, ax = plt.subplots(1,1, sharex='all', sharey='all')
plt.subplots_adjust(left=0.25, bottom=0.3)
ax.set_xlim([0.0,12.0])
ax.set_ylim([-0, 0.50])
ax.grid(True)
HSV = (np.random.random(),1,1)
rgb = hsv_to_rgb(HSV)

# shape, scale = 1.6073, 0.1
# x = np.linspace(stats.weibull_min.ppf(0.0, shape, scale=scale),
#                         stats.weibull_min.ppf(0.9999, shape, scale=scale), 1000)
# l, = plt.plot(x, stats.weibull_min.pdf(x, shape, scale=scale), 
#                     c=rgb, 
#                     lw=3, 
#                     alpha=0.6)
# ax_shape = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# ax_scale = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# s_shape = Slider(ax_shape, 'shape', 1.0, 5.0, valinit=1.215, valstep=0.1)
# s_scale = Slider(ax_scale, 'scale', 0.01, 1.0, valinit=0.1, valstep=0.01)

#lognormal plot
mu, median = 4.6, 3.7#2.1, 1.7
scale, shape = median, np.sqrt(2.0*np.log(mu)) 
dist = stats.lognorm(shape, scale=scale)
# x = np.linspace(dist.ppf(0.0000),
#                 dist.ppf(0.9999), 10000)

x = np.linspace(0.00, 100.0, 1000)

l, = plt.plot(x, dist.pdf(x), 
                    c=rgb, 
                    lw=3, 
                    alpha=0.6)

l2, = plt.plot([dist.mean(),
                dist.mean()],
                [0.00, 100.00], 'r-')

mode = x[l.get_ydata().argmax()]

l3, = plt.plot([mode,
                mode],
                [0.00, 100.00], 'b-')

annotation = plt.annotate(f"Mean Y: {dist.mean():.4f}",
                        (0.00,0.75),
                        xycoords='figure fraction')
annotation2 = plt.annotate(f"Mode Y: {mode:.4f}",
                        (0.00,0.65),
                        xycoords='figure fraction')

ax_mu = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_median = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

s_mu = Slider(ax_mu, 'mu', 0.0, 20.0, valinit=mu, valstep=0.1)
s_median = Slider(ax_median, 'median', 0.01, 10.0, valinit=median, valstep=0.01)

#sigmoidal plot
def sigmoid(x, scale=1.0, shape=1.0, x0=0, y0=0, xr=False, yr=False):
    if xr:
        x_reflect=-1.0
    else:
        x_reflect=1.0

    if yr:
        y_reflect=-1.0
    else:
        y_reflect=1.0

    return y_reflect*((scale*1.)/(1. + np.exp(-(shape*x_reflect*(x-x0))))) + y0

def exponential(x, scale=1.0, shape=1.0, x0=0, y0=0, xr=False, yr=False):
    if xr:
        x_reflect=-1.0
    else:
        x_reflect=1.0

    if yr:
        y_reflect=-1.0
    else:
        y_reflect=1.0

    return scale*y_reflect*np.exp(shape*x_reflect*(x-x0)) + y0   

# scale, shape, x0, y0 = 0.9, 1.0, 0, 0.1
# x = np.linspace(0.0,50,100)
# l, = plt.plot(x, exponential(x, scale=scale, shape=shape, x0=x0, y0=y0, xr=True),
#                     c = rgb,
#                     lw=3,
#                     alpha=0.6)

# ax_shape = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# ax_scale = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# ax_x0 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# ax_y0 = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# s_shape = Slider(ax_shape, 'shape', -1.0, 1.0, valinit=shape, valstep=0.01)
# s_scale = Slider(ax_scale, 'scale', 0.01, 5.0, valinit=scale, valstep=0.01)
# s_x0 = Slider(ax_x0, 'x0', -30, 30, valinit=x0, valstep=1)
# s_y0 = Slider(ax_y0, 'y0', -5.0, 5.0, valinit=y0, valstep=0.01)

def update(val):
#     # shape = s_shape.val
#     # scale = s_scale.val
#     # l.set_ydata(stats.weibull_min.pdf(x, shape, scale=scale))

    mu = s_mu.val
    median = s_median.val
    scale, shape = median, np.sqrt(2.0*np.log(mu))
    dist = stats.lognorm(shape, scale=scale)
    l.set_ydata(dist.pdf(x))
    mode = x[l.get_ydata().argmax()]
    annotation.set_text(f"Mean Y: {dist.mean():.4f}")
    annotation2.set_text(f"Mode Y: {mode:.4f}")
    l2.set_xdata([dist.mean(),
                dist.mean()])
    l3.set_xdata([mode,mode])

    # shape = s_shape.val
    # scale = s_scale.val
    # x0 = s_x0.val
    # y0 = s_y0.val
    # l.set_ydata(exponential(x, scale=scale, shape=shape, x0=x0, y0=y0, xr=True))
    # fig.canvas.draw_idle()
    
# # s_shape.on_changed(update)
# # s_scale.on_changed(update)

s_mu.on_changed(update)
s_median.on_changed(update)

# s_shape.on_changed(update)
# s_scale.on_changed(update)
# s_x0.on_changed(update)
# s_y0.on_changed(update)

plt.show()
