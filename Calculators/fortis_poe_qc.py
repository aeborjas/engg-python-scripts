import datetime, time, numpy as np, pandas as pd
from os.path import abspath, dirname, join
from scipy.stats import gamma, norm

pd.set_option("display.max_rows",50)
pd.set_option("display.max_columns",50)

def pipe_prop(df, units="native"):
    """
    Unpacks dataframe into set of arrays of variables
    :param df: input dataframe with set columns from SQL statement
    :return: OD, WTm, Sm, OPm, Inst, i
    """
    od = df['OD_inch'].values
    wt = df['WT_mm'].values
    s = df['grade_MPa'].values
    df['toughness_J'] = df['toughness_J'].fillna(10)
    T = df['toughness_J'].values
    op = df['MAOP_kPa'].values
    Inst = pd.to_datetime(df['install_date']).dt.date

    if units=="SI":
        return od*25.4, wt, s, T, op, Inst
    elif units=="US":
        return od, wt/25.4, s/6.89476, T*0.7376, op/6.89476, Inst
    else:
        return od, wt, s, T, op, Inst

def feature_dim(df, units="native"):
    PDP = df['depth_fraction'].values
    L_measured = df['length_mm'].values
    W_measured = df['width_mm'].values
    chainage = df['chainage_m'].values
    status = df['status'].values
    f_type = df['type'].values

    if units=="SI":
        return PDP, L_measured, W_measured, status, f_type, chainage
    elif units=="US":
        return PDP, L_measured/25.4, W_measured/25.4, status, f_type, chainage
    else:
        return PDP, L_measured, W_measured, status, f_type, chainage

def range_prop(df):
    Insp = pd.to_datetime(df['ILIRStartDate']).dt.date
    vendor = df['vendor'].values
    tool = df['tool'].values

    return Insp, vendor, tool

def tiler(*args, tuple_size=(1,1)):
    temp = []
    for arg in args:
        temp.append(np.tile(arg, tuple_size))
    return np.array(temp)

def random_prob_gen(x, iterations=1, features=1):
    temp=[]
    for var in range(x):
        temp.append(np.random.rand(iterations,features))
    return np.array(temp)

def ls_corr_rupture(fail_pressure, operating_pressure, thresh=1.0, bulk=False):
    """
    Limit state for rupture failure mode of corrosion
    :param fail_pressure: failure pressure, in kPa (SI), or psi (US)
    :param operating_pressure: operating pressure, in kPa (SI), or psi (US)
    :param bulk: Flag for applying limit state on a bulk basis, default is False
    :return: returns array(s) of 1's and 0's, where 1 indicates a failure, and 0 indicates no failure
    """

    ruptures = fail_pressure <= operating_pressure*thresh
    ruptures = ruptures.astype(int)
    if bulk:

        rupture_count = np.sum(ruptures, axis=0)
    else:

        rupture_count = np.sum(ruptures)
    return ruptures, rupture_count

def ls_corr_leak(wt, fD, thresh=0.8, bulk=False):
    """
    Limit state for leak failure mode of corrosion
    :param wt: pipe wall thickness, in mm (SI), or inch (US)
    :param fD: feature depth, in mm (SI), or inch (US)
    :param thresh: leak threshold, in fraction, default of 0.80
    :param bulk: Flag for applying limit state on a bulk basis, default is False
    :return: returns array(s) of 1's and 0's where 1 indicates a failure, and 0 indicates no failure
    """

    leaks = fD >= thresh * wt
    leaks = leaks.astype(int)
    if bulk:

        leak_count = np.sum(leaks, axis=0)
    else:

        leak_count = np.sum(leaks)
    return leaks, leak_count

def ls_corr_tot(fail_1, fail_2, bulk=False):

    fails = np.maximum(fail_1, fail_2)
    if bulk:

        fail_count = np.sum(fails, axis=0)
    else:

        fail_count = np.sum(fails)
    return fails, fail_count

def modified_b31g(od, wt, s, fL, fD, units="SI"):
    """
    Calculates the failure stress using the Modified B31G Equation
    :param od:  Pipe outside diameter, in mm (SI), or inches (US)
    :param wt:  Pipe wall thickness, in mm (SI), or inches (US)
    :param s:   Pipe grade, in kPa (SI), or psi (US)
    :param fL:  feature length, in mm (SI), or inches (US)
    :param fD:  feature depth, in mm (SI), or inches (US)
    :param units: flag for which units to use, "SI" or "US", default "SI"
    :return: Failure stress, in kPa (SI), or psi (US)
    """

    l2Dt = np.power(fL, 2.0)/(od*wt)
    Mt = np.where(l2Dt <= 50.0,
                  np.sqrt( 1.0 +(0.6275*l2Dt)-(0.003375*np.power(l2Dt, 2.0))),
                  0.032*l2Dt+3.3)
    if units=="SI":
        flowS = s + 68947.6
    else:
        flowS = s + 10000.0

    failStress = flowS *( 1.0- 0.85 * (fD/wt)) / (1.0 - 0.85 * (fD/(wt * Mt)))
    return failStress

#corrpoe is the function I call to calculate POE. 
def corrpoe(df, n, projection_years=5):
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
    OD, WTm, Sm, Tm, OPm, Inst = pipe_prop(df)
    pipe_age = ((now - Inst).dt.days.values) / 365.25
    
    # Feature properties extracted from the dataframe df.
    fPDP, fL_measured, fW_measured, fstatus, ftype, fchainage = feature_dim(df)
    fID = df['FeatureID'].values
    f_v_CGR = df['vendor_cgr_mmpyr'].fillna(np.nan).values
    f_v_sd_CGR = df['vendor_cgr_sd'].fillna(np.nan).values
    shape = 1.24959
    scale = 0.107359

    # Inline inspection range properties extracted from the dataframe df.
    Insp, vendor, tool = range_prop(df)

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

##    tool_D = 0.117 * WT# in fraction WT
##    tool_L = 15.61 / 25.4 # in inches
    
    #special request for feature distribution
    tool_D = 0.025 * WT
    tool_L = 75 / 25.4

    # OD and MOP are not distributed in this approach. This function creates an array of size (n,i) of repeating OD and OP values.
    OD, OP = tiler(OD, OP, tuple_size=(n, 1))

    # Setting a random seed
    np.random.seed()

    #  Initializing arrays of size (n, i), containing random numbers from 0 to 1, for each of model error, wall thickness, grade, feature length, feature depth, and growth rate.
    mE_n_1, WT_n_2, S_n_3, fL_n_5, fD_n_6, fGR_n_7 = random_prob_gen(6, iterations=n, features=i)

    # the following 2 statements distribute the wall thickness and grade, using normal distributions. Equivalent Excel equation is NORM.INV()
    WTd = norm.ppf(WT_n_2, loc=WT * meanWT, scale=WT * sdWT)
    Sdist = norm.ppf(S_n_3, loc=S * meanS, scale=S * sdS)

    # the following statement distributes the feature length in inches, using normal distribution, and ensures the length is not distributed below 0.0 mm.
##    fL = np.maximum(0, norm.ppf(fL_n_5, loc=fL_measured * 1.0 * (1 / 25.4), scale=tool_L))
    fL = np.tile(fL_measured * (1/25.4), (n,1))

    # the following statement distributes the feature depth in inches, using normal distribution, and ensures the depth is not distributed below 0.0 mm.
##    fD_run = np.clip(0, norm.ppf(fD_n_6, loc=fPDP * WT * 1.0, scale=tool_D),WTd)
    fD_run = np.tile(fPDP * WT, (n,1))

    # calculating the depth at the year of analysis
    fD = fD_run

    # calculating the Failure Stress in psi. 
    failure_stress = modified_b31g(OD, WTd, Sdist, fL, fD, units="US")

    # calculating Failure pressure in psi, using the Barlow formula
    failPress = 2 * failure_stress * WTd / OD

    # comparing the failure pressures to the operating pressure of the pipeline. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    ruptures, rupture_count = ls_corr_rupture(failPress, OP, thresh=1.25, bulk=True)

    # comparing the depth at year of analysis with the wall thickness to see if the 80.0%WT criteria is exceeded. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    leaks, leak_count = ls_corr_leak(WTd, fD, bulk=True)

    # checking both leak and rupture limit states at the same time. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    fails, fail_count = ls_corr_tot(ruptures, leaks, bulk=True)
    
    failure_projection = {'leaks':[None]*projection_years,
                         'ruptures':[None]*projection_years,
                         'fails':[None]*projection_years}
    
    # Calculate POE for each of 10 years from year of analysis
    for x in range(projection_years):
        #growth rate must be in inches/year
        fD_GR = np.maximum(0, norm.ppf(fGR_n_7, loc=f_v_CGR * 1.0 * (1/25.4), scale=f_v_sd_CGR * (1/25.4)))
##        if x <= 19:
##            fD_GR = np.maximum(0, norm.ppf(fGR_n_7, loc=0.10 * 1.0 * (1/25.4), scale=0.03 * (1/25.4)))
##        else:
##            fD_GR = np.maximum(0, norm.ppf(fGR_n_7, loc=0.30 * 1.0 * (1/25.4), scale=0.03 * (1/25.4)))

        # Add 1 year of corrosion growth
        fD = np.minimum(WTd,fD + fD_GR*(1.0))
        
        # recalculate Failure Stress in psi
        failure_stress = modified_b31g(OD, WTd, Sdist, fL, fD, units="US")

        # recalculate Failure pressure in psi
        failPress = 2 * failure_stress * WTd / OD

        # redetermine limit state exceedances
        rupture_proj, failure_projection['ruptures'][x] = ls_corr_rupture(failPress, OP, thresh=1.25, bulk=True)
        leak_proj, failure_projection['leaks'][x] = ls_corr_leak(WTd, fD, bulk=True)
        _, failure_projection['fails'][x] = ls_corr_tot(rupture_proj, leak_proj, bulk=True)
        
    # setting up a container to hold all the information to be returned as results
    fPDP_final = fD / WT

    return_dict = {"FeatureID":fID,
                   "PDP_frac": fPDP,
                   "Final_mean_PDP_frac":np.mean(fPDP_final,axis=0),
                   "flength": fL_measured,
                   "iterations": np.size(fails, axis=0),
                   "rupture_poe_YOA": rupture_count,
                   "leak_poe_YOA": leak_count,
                   "fail_poe_YOA": fail_count}

## Might be able to use this to automate the creation of the 50 year dictionary output.
    for x in range(projection_years):
##	exec(f"d['row{x}']=x")
        return_dict[f"leak_poe_Y{x+1}"] = failure_projection['leaks'][x]
        return_dict[f"rupture_poe_Y{x+1}"] = failure_projection['ruptures'][x]
        return_dict[f"fail_poe_Y{x+1}"] = failure_projection['fails'][x]

    return_dict["nan"] = np.zeros((i,))
    

    # converting the container to a tabulat object using pandas
    return_df = pd.DataFrame(return_dict).set_index('FeatureID')

    return return_df

#the next statement loads the CSV into a pandas dataframe
CSV_FILE_NAME = 'FTS sample_of_inputs for Armando Rock Tunnel 10082020_23_6WT.csv'
CSV_FILE_NAME = abspath(join(dirname(__file__), CSV_FILE_NAME))
out_file = "fortis_outputs (20201008 rock tunnel feature matrix).csv"

df = pd.read_csv(CSV_FILE_NAME, header=0)
print('Features loaded...')

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
projection_years = 30
iterations_split = 10
s1 = time.time()

if not(split_iterations):
    result = corrpoe(df,iterations,projection_years)
else:
    split = iterations//iterations_split
    if split >= iterations:
        raise AttributeError("Please select a higher number to split iterations.")
    remainder = iterations - split*iterations_split
    if remainder == 0:
        iter_list = [split]*iterations_split
    else:
        iter_list = [split]*iterations_split + [remainder]

    update_cols = ["iterations","leak_poe_YOA",
                   "rupture_poe_YOA",
                   "fail_poe_YOA"]

    for x in range(projection_years):
        update_cols.append(f"leak_poe_Y{x+1}")
        update_cols.append(f"rupture_poe_Y{x+1}")
        update_cols.append(f"fail_poe_Y{x+1}")

    result = corrpoe(df,iter_list.pop(),projection_years)
    for i, iters in enumerate(iter_list):
        print(f"Performing {i+2}th iteration...",end="\n",flush=True)
        result[update_cols] = result[update_cols] + corrpoe(df,iters,projection_years)[update_cols]

# convert all fail count columns to POE. 5th element is always the first POE at YOA
result.iloc[:,5:] = result.apply(lambda x: x[5:]/x["iterations"], axis=1)

print(f'Done. {time.time() - s1:.4f} seconds')


df = df.set_index('FeatureID').join(result).to_csv(out_file)


## Excerpt from general_POE.py to calculate 10MM iterations and more
# if not(split_iterations):
#     self.result, self.qc = self.mcpoe(self.df,self.iterations)
# else:
