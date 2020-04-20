import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

pd.set_option('display.max_columns',700)
pd.set_option('display.max_rows',700)
pd.set_option('display.expand_frame_repr',False)

def best_fit_m_b(xs,ys):
    m = ( ((mean(xs)*mean(ys)) - mean(xs*ys)) /
          ((mean(xs)**2) - mean(xs**2)))

    b = mean(ys) - m*mean(xs)

    return m, b


def sqerror(ys_orig,ys_line):
    return sum((ys_line-ys_orig)**2)


def rsquared(ys_orig,ys_line):
    y_mean_line = np.array([mean(ys_orig) for y in ys_orig])
    sq_error_regr = sqerror(ys_orig,ys_line)
    sq_error_y_mean = sqerror(ys_orig,y_mean_line)
    return 1 - (sq_error_regr/sq_error_y_mean)


start = time.time()

print("Accessing workbook...")
book_path = r"Z:\Plains Midstream\2018_01_IRAS_Implementation\3_Engineering\Quantitative Risk Run (V6)\Results & QC\Crack GR Analysis\Circumferencial Crack Growth Rate Analysis\circumferential_crack_data.xlsx"

df = pd.read_excel(book_path)

end1 = time.time()
print("Workbook loaded! Loading Time: {}".format(end1-start))

#Select variable to fit
var = pd.DataFrame(df['Half-life Growth Rate (mm/yr)'])

var.columns = ['data']
var = var.sort_values(by='data')

var = var.drop( var[ var['data']<=0 ].index )
var = var.reset_index(drop=True)

var['rank'] = var.index+1
var['median_rank'] = (var['rank']-0.3)/(var['rank'].max()+0.4)
var['inv_median_rank'] = 1/(1-var['median_rank'])
var['ln_data']=np.log(var['data'])
var['ln_ln_inv_median_rank'] = np.log(np.log(var['inv_median_rank']))

m, b = best_fit_m_b(var['ln_data'],var['ln_ln_inv_median_rank'])
print("Slope (m)= {}".format(m))
print("Intercept (b)= {}".format(b), end='\n\n')

shape = m
scale = np.exp((-b)/shape)

print("shape parameter= {}".format(shape))
print("scale parameter= {}".format(scale), end='\n\n')

regression_line = np.array(m*var['ln_data']+b)

print("Starting calculation of r-square...")
r_squared = rsquared(var['ln_ln_inv_median_rank'],regression_line)
print("Coefficient of determination (R-squared)= {}".format(r_squared), end='\n\n')

end2= time.time()
print("Total Processing Time: {}".format(end2-end1), end='\n\n')

print("Plotting...")
plt.scatter(var['ln_data'],var['ln_ln_inv_median_rank'], s=1, edgecolors='k')
plt.plot(var['ln_data'],regression_line, color='g')
plt.title('Weibull Distribution Fitting for Data')
plt.xlabel('ln(data)')
plt.ylabel('ln(ln(inv_median_rank))')
plt.text(-6,6,r'shape={:.2f}, scale={:.4f}'.format(shape,scale))
plt.show()

