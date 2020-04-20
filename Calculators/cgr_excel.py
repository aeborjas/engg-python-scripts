import xlrd
import time
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


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

print("Accessing workbooks...")
book_ext = xlrd.open_workbook(r"Z:\Plains Midstream\2018_01_IRAS_Implementation\3_Engineering\Quantitative Risk Run (V6)\Results & QC\Corrosion CGR Analysis\Half Life Analysis\ext_1962879_cgr_analysis_halflife.xlsx")
book_int = xlrd.open_workbook(r"Z:\Plains Midstream\2018_01_IRAS_Implementation\3_Engineering\Quantitative Risk Run (V6)\Results & QC\Corrosion CGR Analysis\Half Life Analysis\int_799296_cgr_analysis_halflife.xlsx")
end1 = time.time()
print("Workbooks opened! Loading Time: {}".format(end1-start))

sheet1_ext = book_ext.sheet_by_index(0)
sheet2_ext = book_ext.sheet_by_index(1)
sheet3_ext = book_ext.sheet_by_index(2)

sheet_int = book_int.sheet_by_index(0)

data = []

print("Accessing EXT 0 - 900K...")
for row in range(1,sheet1_ext.nrows ): #range(1,899653):
    data.append(sheet1_ext.cell(row,0))
print("Data loaded.")

print("Accessing EXT 900K - 1,800K...")
for row in range(1,sheet2_ext.nrows): #range(1,894166):
    data.append(sheet2_ext.cell(row,0))
print("Data loaded.")

print("Accessing EXT 1,800K - END...")
for row in range(1,sheet3_ext.nrows): #range(1,162878):
    data.append(sheet3_ext.cell(row,0))
print("Data loaded.",end='\n\n')

print("Accessing INT column...")
for row in range(1,sheet_int.nrows): #range(1,799296):
    data.append(sheet_int.cell(row,0))
print("Data loaded.",end='\n\n')

print("Converting data to values...")
data = [data[x].value for x in range(len(data))]
print("Conversion complete!")
print("Size of list = {:,}".format(len(data)),end='\n\n')

data = np.array(data, dtype=float)
data = np.sort(data)

rank = np.array([x for x in range(1,len(data)+1)])
median_rank = (rank-0.3)/(max(rank)+0.4)
inv_median_rank = 1/(1-median_rank)
ln_data = np.log(data)
ln_ln_inv_median_rank = np.log(np.log(inv_median_rank))

m, b = best_fit_m_b(ln_data,ln_ln_inv_median_rank)
print("Slope (m)= {}".format(m))
print("Intercept (b)= {}".format(b), end='\n\n')

shape = m
scale = np.exp((-b)/shape)

print("shape parameter= {}".format(shape))
print("scale parameter= {}".format(scale), end='\n\n')

regression_line = np.array([(m*x)+b for x in ln_data])

##print("Starting calculation of r-square...")
##r_squared = rsquared(ln_ln_inv_median_rank,regression_line)
##print("Coefficient of determination (R-squared)= {}".format(r_squared), end='\n\n')

end2= time.time()
print("Total Processing Time: {}".format(end2-end1), end='\n\n')

print("Plotting...")
plt.scatter(ln_data,ln_ln_inv_median_rank, s=1, edgecolors='k')
plt.plot(ln_data,regression_line, color='g')
plt.title('Weibull Distribution Fitting for CGR')
plt.xlabel('ln(CGR)')
plt.ylabel('ln(ln(inv_median_rank))')
plt.text(-6,6,r'shape={:.2f}, scale={:.4f}'.format(shape,scale))
plt.show()

