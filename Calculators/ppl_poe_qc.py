#the norm and gamma are functions from the scipy.stats module. norm is the normal distribution, and gamma is the gamma function.
#python module for accessing and working with time.
import datetime
import time
from os.path import abspath, dirname, join

#numpy is a python module to work with arrays. pandas is a python module to work with tabular data.
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from simpoe import cgr, distributer, model_constants, unpacker
#simpoe is a custom module I created to facilitate some of the functions required for POE. Please find the files for each of these modules in the "poe dependencies" folder.
from simpoe.corrosion import corrLimitStates, failureStress


def weibull_fitter(var):
    def best_fit_m_b(xs,ys):
        m = ( ((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
              ((np.mean(xs)**2) - np.mean(xs**2)))

        b = np.mean(ys) - m*np.mean(xs)

        return m, b

#     def sqerror(ys_orig,ys_line):
#         return sum((ys_line-ys_orig)**2)


#     def rsquared(ys_orig,ys_line):
#         y_mean_line = np.array([np.mean(ys_orig) for y in ys_orig])
#         sq_error_regr = sqerror(ys_orig,ys_line)
#         sq_error_y_mean = sqerror(ys_orig,y_mean_line)
#         return 1 - (sq_error_regr/sq_error_y_mean)

    var = pd.DataFrame(var)
    
    #rename column to 'data', sort ascending, drop values that are negative, and reset index
    var.columns = ['data']
    var = var.sort_values(by='data')
    var = var.drop( var[ var['data']<=0 ].index )
    var = var.reset_index(drop=True)

    #create rank, median rank variables
    var['rank'] = var.index+1
    var['median_rank'] = (var['rank']-0.3)/(var['rank'].max()+0.4)
    
    var['inv_median_rank'] = 1/(1-var['median_rank'])
    var['ln_data']=np.log(var['data'])
    var['ln_ln_inv_median_rank'] = np.log(np.log(var['inv_median_rank']))

    #find the best fit of the ln(data) and the ln(ln(inverse median rank))
    m, b = best_fit_m_b(var['ln_data'],var['ln_ln_inv_median_rank'])
    shape = m
    scale = np.exp((-b)/shape)

#     regression_line = np.array(m*var['ln_data']+b)
#     r_squared = rsquared(var['ln_ln_inv_median_rank'],regression_line)

#     print(f"shape parameter= {shape:0.4f}")
#     print(f"scale parameter= {scale:0.4f}")
#     print(f"Coefficient of determination (R-squared)= {r_squared:0.2%}")

#     plt.scatter(var['ln_data'],var['ln_ln_inv_median_rank'], s=10, edgecolors='k')
#     plt.plot(var['ln_data'],regression_line, color='g')
#     plt.title('Weibull Distribution Fitting for Data')
#     plt.xlabel('ln(data)')
#     plt.ylabel('ln(ln(inv_median_rank))')
#     plt.show()
    
    return shape, scale

#corrpoe is the function I call to calculate POE. 
def corrpoe(df, n, cga_rate=10):
    """
    Calculates POE using Monte Carlo approach
    :param df:          input pandas dataframe with input columns.
	:param n:           number of iterations to run Monte Carlo on.
	:param cga_rate:	default value to use for fixed growth rate. Omitt, as it shouldn't be used in calculation anyways.
    :return:            returns a pandas dataframe (tabular object) with the inputs for each feature, and 33 columns for leak, rupture, and total POE for the year of analysis, and each of year 1 through year 10 after year of analysis.
    """
    # i is the Number of features, equal to number of rows in csv file
    i = df.shape[0]
    now = datetime.date.today()
    
    # Pipe properties extracted from the dataframe df.
    # Please follow the "unpacker.py" file for explanation as to how this is carried out.
    OD, WTm, Sm, Tm, OPm, Inst = unpacker.pipe_prop(df)
    pipe_age = ((now - Inst).dt.days.values) / 365.25
    
    # Feature properties extracted from the dataframe df.
    # Please follow the "unpacker.py" file for explanation as to how this is carried out.
    fPDP, fL_measured, fW_measured, fstatus, ftype, fchainage = unpacker.feature_dim(df)
    fID = df['FeatureID'].values
    f_v_CGR = df['vendor_cgr_mmpyr'].fillna(np.nan).values
    f_v_sd_CGR = df['vendor_cgr_sd'].fillna(np.nan).values
    shape = df['shape'].values
    scale = df['scale'].values

    # Inline inspection range properties extracted from the dataframe df.
    # Please follow the "unpacker.py" file for explanation as to how this is carried out.
    Insp, vendor, tool = unpacker.range_prop(df)

    # Calculating the time difference between date of ILI and today, in fractional years.
    time_delta = ((now - Insp).dt.days.values) / 365.25

	# defining the function for model error. This function uses the gamma.ppf() function part of the import statement above. The equivalent Excel function is GAMMA.INV()
    def model_error(p):
        return 0.914 + gamma.ppf(p, 2.175, scale=0.225)

    # unit conversion to US units
    WT = WTm / 25.4
    S = (Sm * 1000) / 6.89476
    OP = OPm / 6.89476

    # Statistics fueled by CSA Z662 Annex O Tables O.6 and O.7
    meanWT = 1.01
    sdWT = 0.01
    meanS = 1.10 #recommended from CSA Z662 2019
    sdS = 0.035 #recommended from CSA Z662 2019

    tool_D = 0.078 * WT# in fraction WT
    tool_L = 7.80 / 25.4 # in inches

    # OD and MOP are not distributed in this approach. This function creates an array of size (n,i) of repeating OD and OP values.
    # Please refer to "distributer.py" file for details.
    OD, OP = distributer.tiler(OD, OP, tuple_size=(n, 1))

    # Setting a random seed
    np.random.seed()

	#  Initializing arrays of size (n, i), containing random numbers from 0 to 1, for each of model error, wall thickness, grade, feature length, feature depth, and growth rate.
    #  Please refer to "distributer.py" file for details.
    mE_n_1, WT_n_2, S_n_3, fL_n_5, fD_n_6, fGR_n_7 = distributer.random_prob_gen(6, iterations=n, features=i)

    # the following 2 statements distribute the wall thickness and grade, using normal distributions. Equivalent Excel equation is NORM.INV()
    WTd = norm.ppf(WT_n_2, loc=WT * meanWT, scale=WT * sdWT)
    Sdist = norm.ppf(S_n_3, loc=S * meanS, scale=S * sdS)

    # the following statement distributes the feature length in inches, using normal distribution, and ensures the length is not distributed below 0.0 mm.
    fL = np.maximum(0, norm.ppf(fL_n_5, loc=fL_measured * 1.0 * (1 / 25.4), scale=tool_L))

    # the following statement distributes the feature depth in inches, using normal distribution, and ensures the depth is not distributed below 0.0 mm.
    fD_run = np.maximum(0, norm.ppf(fD_n_6, loc=fPDP * WT * 1.0, scale=tool_D))

    # the following statement applies the growth rate assumptions. where vendor provided growth rate data is provided, use it, otherwise use the shape and scale parameters as part of the sample_inputs.csv
    # the np.where() function is used to make conditional checks of arrays.
    # Please refer to the "cgr.py" file for the details on the cgr_mpy() and cgr_weibull() functions
    
    # alpha = f_v_CGR / f_v_sd_CGR
    alpha = 0.0

## Original PPL Logic.
##    fD_GR = np.where(np.isnan(f_v_CGR),
##                    np.where(fD_run>0.40,
##                            (fD_run)/((Insp-Inst).dt.days.values/365.25),
##                            np.maximum(0, cgr.cgr_weibull(fGR_n_7, shape, scale) / 25.4)),
##                            np.where(alpha > 0.25,
##                                    np.maximum(0, norm.ppf(fGR_n_7, loc=f_v_CGR * 1.0 * (1/25.4), scale=f_v_sd_CGR * (1/25.4))),
##                                    np.maximum(0, cgr.cgr_weibull(fGR_n_7, shape, scale) / 25.4)))

## 2020-08-18 Request by Scott and Tyler
    fD_GR = np.maximum(0, norm.ppf(fGR_n_7, loc=f_v_CGR * 1.0 * (1/25.4), scale=f_v_sd_CGR * (1/25.4)))

    # calculating the depth at the year of analysis
    fD = fD_run + fD_GR * 0

    # calculating the Failure Stress in psi. 
    # Please refer to the "failureStress.py" file for the details on the calculation.
    failure_stress = failureStress.modified_b31g(OD, WTd, Sdist, fL, fD, units="US")

    # calculating Failure pressure in psi, using the Barlow formula
    failPress = 2 * failure_stress * WTd / OD

    # comparing the failure pressures to the operating pressure of the pipeline. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    # Please refer to the "corrLimitStates.py" file for details on the limit state checking.
    ruptures, rupture_count = corrLimitStates.ls_corr_rupture(failPress, OP, thresh=1.25, bulk=True)

    # comparing the depth at year of analysis with the wall thickness to see if the 80.0%WT criteria is exceeded. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    # Please refer to the "corrLimitStates.py" file for details on the limit state checking.
    leaks, leak_count = corrLimitStates.ls_corr_leak(WTd, fD, bulk=True)

    # checking both leak and rupture limit states at the same time. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    # Please refer to the "corrLimitStates.py" file for details.
    fails, fail_count = corrLimitStates.ls_corr_tot(ruptures, leaks, bulk=True)

    # setting up the projection of 10 years after year of analysis.
    failure_projection = {'leaks':[None]*30,
                         'ruptures':[None]*30,
                         'fails':[None]*30}
    
    # Calculate POE for each of 10 years from year of analysis
    for x in range(30):
        # Add 1 year of corrosion growth
        fD += fD_GR*(1.0)
        
        # recalculate Failure Stress in psi
        failure_stress = failureStress.modified_b31g(OD, WTd, Sdist, fL, fD, units="US")

        # recalculate Failure pressure in psi
        failPress = 2 * failure_stress * WTd / OD

        # redetermine limit state exceedances
        rupture_proj, failure_projection['ruptures'][x] = corrLimitStates.ls_corr_rupture(failPress, OP, thresh=1.25, bulk=True)
        leak_proj, failure_projection['leaks'][x] = corrLimitStates.ls_corr_leak(WTd, fD, bulk=True)
        _, failure_projection['fails'][x] = corrLimitStates.ls_corr_tot(rupture_proj, leak_proj, bulk=True)
        
    # setting up a container to hold all the information to be returned as results
    return_dict = {"FeatureID":fID,
                   "PDP_frac": fPDP,
                   "flength": fL_measured,
                   "alpha":alpha,
                   "iterations": np.size(fails, axis=0),
                   "rupture_count_YOA": rupture_count,
                   "leak_count_YOA": leak_count,
                   "fail_count_YOA": fail_count,
                   "rupture_count_Y1": failure_projection['ruptures'][0],
                   "leak_count_Y1": failure_projection['leaks'][0],
                   "fail_count_Y1": failure_projection['fails'][0],
                   "rupture_count_Y2": failure_projection['ruptures'][1],
                   "leak_count_Y2": failure_projection['leaks'][1],
                   "fail_count_Y2": failure_projection['fails'][1],
                   "rupture_count_Y3": failure_projection['ruptures'][2],
                   "leak_count_Y3": failure_projection['leaks'][2],
                   "fail_count_Y3": failure_projection['fails'][2],
                   "rupture_count_Y4": failure_projection['ruptures'][3],
                   "leak_count_Y4": failure_projection['leaks'][3],
                   "fail_count_Y4": failure_projection['fails'][3],
                   "rupture_count_Y5": failure_projection['ruptures'][4],
                   "leak_count_Y5": failure_projection['leaks'][4],
                   "fail_count_Y5": failure_projection['fails'][4],
                   "rupture_count_Y6": failure_projection['ruptures'][5],
                   "leak_count_Y6": failure_projection['leaks'][5],
                   "fail_count_Y6": failure_projection['fails'][5],
                   "rupture_count_Y7": failure_projection['ruptures'][6],
                   "leak_count_Y7": failure_projection['leaks'][6],
                   "fail_count_Y7": failure_projection['fails'][6],
                   "rupture_count_Y8": failure_projection['ruptures'][7],
                   "leak_count_Y8": failure_projection['leaks'][7],
                   "fail_count_Y8": failure_projection['fails'][7],
                   "rupture_count_Y9": failure_projection['ruptures'][8],
                   "leak_count_Y9": failure_projection['leaks'][8],
                   "fail_count_Y9": failure_projection['fails'][8],
                   "rupture_count_Y10": failure_projection['ruptures'][9],
                   "leak_count_Y10": failure_projection['leaks'][9],
                   "fail_count_Y10": failure_projection['fails'][9],
                   "rupture_count_Y11": failure_projection['ruptures'][10],
                   "leak_count_Y11": failure_projection['leaks'][10],
                   "fail_count_Y11": failure_projection['fails'][10],
                   "rupture_count_Y12": failure_projection['ruptures'][11],
                   "leak_count_Y12": failure_projection['leaks'][11],
                   "fail_count_Y12": failure_projection['fails'][11],
                   "rupture_count_Y13": failure_projection['ruptures'][12],
                   "leak_count_Y13": failure_projection['leaks'][12],
                   "fail_count_Y13": failure_projection['fails'][12],
                   "rupture_count_Y14": failure_projection['ruptures'][13],
                   "leak_count_Y14": failure_projection['leaks'][13],
                   "fail_count_Y14": failure_projection['fails'][13],
                   "rupture_count_Y15": failure_projection['ruptures'][14],
                   "leak_count_Y15": failure_projection['leaks'][14],
                   "fail_count_Y15": failure_projection['fails'][14],
                   "rupture_count_Y16": failure_projection['ruptures'][15],
                   "leak_count_Y16": failure_projection['leaks'][15],
                   "fail_count_Y16": failure_projection['fails'][15],
                   "rupture_count_Y17": failure_projection['ruptures'][16],
                   "leak_count_Y17": failure_projection['leaks'][16],
                   "fail_count_Y17": failure_projection['fails'][16],
                   "rupture_count_Y18": failure_projection['ruptures'][17],
                   "leak_count_Y18": failure_projection['leaks'][17],
                   "fail_count_Y18": failure_projection['fails'][17],
                   "rupture_count_Y19": failure_projection['ruptures'][18],
                   "leak_count_Y19": failure_projection['leaks'][18],
                   "fail_count_Y19": failure_projection['fails'][18],
                   "rupture_count_Y20": failure_projection['ruptures'][19],
                   "leak_count_Y20": failure_projection['leaks'][19],
                   "fail_count_Y20": failure_projection['fails'][19],
                   "rupture_count_Y21": failure_projection['ruptures'][20], ###### special request for Scott and Tyler 2020-08-18
                   "leak_count_Y21": failure_projection['leaks'][20],
                   "fail_count_Y21": failure_projection['fails'][20],
                   "rupture_count_Y22": failure_projection['ruptures'][21],
                   "leak_count_Y22": failure_projection['leaks'][21],
                   "fail_count_Y22": failure_projection['fails'][21],
                   "rupture_count_Y23": failure_projection['ruptures'][22],
                   "leak_count_Y23": failure_projection['leaks'][22],
                   "fail_count_Y23": failure_projection['fails'][22],
                   "rupture_count_Y24": failure_projection['ruptures'][23],
                   "leak_count_Y24": failure_projection['leaks'][23],
                   "fail_count_Y24": failure_projection['fails'][23],
                   "rupture_count_Y25": failure_projection['ruptures'][24],
                   "leak_count_Y25": failure_projection['leaks'][24],
                   "fail_count_Y25": failure_projection['fails'][24],
                   "rupture_count_Y26": failure_projection['ruptures'][25],
                   "leak_count_Y26": failure_projection['leaks'][25],
                   "fail_count_Y26": failure_projection['fails'][25],
                   "rupture_count_Y27": failure_projection['ruptures'][26],
                   "leak_count_Y27": failure_projection['leaks'][26],
                   "fail_count_Y27": failure_projection['fails'][26],
                   "rupture_count_Y28": failure_projection['ruptures'][27],
                   "leak_count_Y28": failure_projection['leaks'][27],
                   "fail_count_Y28": failure_projection['fails'][27],
                   "rupture_count_Y29": failure_projection['ruptures'][28],
                   "leak_count_Y29": failure_projection['leaks'][28],
                   "fail_count_Y29": failure_projection['fails'][28],
                   "rupture_count_Y30": failure_projection['ruptures'][29],
                   "leak_count_Y30": failure_projection['leaks'][29],
                   "fail_count_Y30": failure_projection['fails'][29],
                   "nan": np.zeros((i,))
                   }
    # converting the container to a tabulat object using pandas
    return_df = pd.DataFrame(return_dict).set_index('FeatureID')

    return return_df

#the next statement loads the CSV into a pandas dataframe
CSV_FILE_NAME = 'FTS sample_of_inputs for Armando Rock Tunnel 08312020.csv'
CSV_FILE_NAME = abspath(join(dirname(__file__), CSV_FILE_NAME))

df = pd.read_csv(CSV_FILE_NAME, header=0)
print('Features loaded...')

#This next set of lines converts the install_date and ILIRStartDate to datetime objects
#   calculates the time delta between the ILI and the pipe installation date
#   calculates the linear full-life growth rate for each feature

df.fillna({'incubation_yrs':10.0}, inplace=True)
df.install_date = pd.to_datetime(df.install_date)
df.ILIRStartDate = pd.to_datetime(df.ILIRStartDate)
df.loc[:, 'ILI_age_yrs'] = (df.ILIRStartDate - df.install_date).dt.days/365.25
df.loc[:, 'growth_rate_mmpyr'] = (df.depth_fraction*df.WT_mm)/df.ILI_age_yrs
print('Growth rates calculated. Fitting Weibull distributions...')


## UPDATE: Now we are only checking for age at ILI less than 5 and greater than or equal to 5.
query_1 = "ILI_age_yrs < 5."
query_2 = "ILI_age_yrs >= 5"

depthScale = 0.30

## UPDATE
#following statements calculate weibull distributions for each of the 2 assumptions, and sets the shape and scale parameters for the features, in accordance to each feature's assumption to use
df = df.combine_first(df.apply(lambda x: [1.24959, 0.107359 + depthScale * ((x.depth_fraction-0.2))],result_type='expand', axis=1).loc[df.query(query_1).index,:])
print('CGR Assumption #1 determined...')

df = df.combine_first(df.apply(lambda x: [1.24959, 0.107359 + depthScale * ((x.depth_fraction/(x.ILI_age_yrs/5)-0.2))],result_type='expand', axis=1).loc[df.query(query_2).index,:])
print('CGR Assumption #2 determined...')

df.rename({0:'shape',1:'scale'}, axis=1, inplace=True)

filters = {'FeatureIDs':[
                        ]}

print(f'Filtering features for {filters["FeatureIDs"]}.')

if len(filters['FeatureIDs']) == 0:
    pass
else:
    df = df.loc[lambda x: (x.FeatureID.isin(filters['FeatureIDs']))]
      
print('Calculating POE...')
iterations = 1_000_000
split_iterations = False
iterations_split = 10
s1 = time.time()

if not(split_iterations):
    result = corrpoe(df,iterations)
else:
    split = iterations//iterations_split
    if split >= iterations:
        raise AttributeError("Please select a higher number to split iterations.")
    remainder = iterations - split*iterations_split
    if remainder == 0:
        iter_list = [split]*iterations_split
    else:
        iter_list = [split]*iterations_split + [remainder]

    update_cols = ["iterations",
                    "rupture_count_YOA",
                    "leak_count_YOA",
                    "fail_count_YOA",
                    "rupture_count_Y1",
                    "leak_count_Y1",
                    "fail_count_Y1",
                    "rupture_count_Y2",
                    "leak_count_Y2",
                    "fail_count_Y2",
                    "rupture_count_Y3",
                    "leak_count_Y3",
                    "fail_count_Y3",
                    "rupture_count_Y4",
                    "leak_count_Y4",
                    "fail_count_Y4",
                    "rupture_count_Y5",
                    "leak_count_Y5",
                    "fail_count_Y5",
                    "rupture_count_Y6",
                    "leak_count_Y6",
                    "fail_count_Y6",
                    "rupture_count_Y7",
                    "leak_count_Y7",
                    "fail_count_Y7",
                    "rupture_count_Y8",
                    "leak_count_Y8",
                    "fail_count_Y8",
                    "rupture_count_Y9",
                    "leak_count_Y9",
                    "fail_count_Y9",
                    "rupture_count_Y10",
                    "leak_count_Y10",
                    "fail_count_Y10",
                    "rupture_count_Y11",
                    "leak_count_Y11",
                    "fail_count_Y11",
                    "rupture_count_Y12",
                    "leak_count_Y12",
                    "fail_count_Y12",
                    "rupture_count_Y13",
                    "leak_count_Y13",
                    "fail_count_Y13",
                    "rupture_count_Y14",
                    "leak_count_Y14",
                    "fail_count_Y14",
                    "rupture_count_Y15",
                    "leak_count_Y15",
                    "fail_count_Y15",
                    "rupture_count_Y16",
                    "leak_count_Y16",
                    "fail_count_Y16",
                    "rupture_count_Y17",
                    "leak_count_Y17",
                    "fail_count_Y17",
                    "rupture_count_Y18",
                    "leak_count_Y18",
                    "fail_count_Y18",
                    "rupture_count_Y19",
                    "leak_count_Y19",
                    "fail_count_Y19",
                    "rupture_count_Y20",
                    "leak_count_Y20",
                    "fail_count_Y20",
                    "rupture_count_Y21",
                    "leak_count_Y21",
                    "fail_count_Y21",
                    "rupture_count_Y22",
                    "leak_count_Y22",
                    "fail_count_Y22",
                    "rupture_count_Y23",
                    "leak_count_Y23",
                    "fail_count_Y23",
                    "rupture_count_Y24",
                    "leak_count_Y24",
                    "fail_count_Y24",
                    "rupture_count_Y25",
                    "leak_count_Y25",
                    "fail_count_Y25",
                    "rupture_count_Y26",
                    "leak_count_Y26",
                    "fail_count_Y26",
                    "rupture_count_Y27",
                    "leak_count_Y27",
                    "fail_count_Y27",
                    "rupture_count_Y28",
                    "leak_count_Y28",
                    "fail_count_Y28",
                    "rupture_count_Y29",
                    "leak_count_Y29",
                    "fail_count_Y29",
                    "rupture_count_Y30",
                    "leak_count_Y30",
                    "fail_count_Y30",'nan']

    result = corrpoe(df,iter_list.pop())
    for i, count in enumerate(iter_list):
        print(f"Performing {i+2}th iteration...")
        result[update_cols] = result[update_cols] + corrpoe(df,count)[update_cols]


print(f'Done. {time.time() - s1:.4f} seconds')

result.iloc[:, 4:] = result.iloc[:,4:].apply(lambda x: x/iterations)#.applymap('{:.3e}'.format)
result.loc[:,'iterations'] = result.loc[:,'iterations'].apply('{:,.0f}'.format)

df.set_index('FeatureID').join(result).loc[:,:].to_csv("fortis_outputs.csv")



## Excerpt from general_POE.py to calculate 10MM iterations and more
# if not(split_iterations):
#     self.result, self.qc = self.mcpoe(self.df,self.iterations)
# else:
