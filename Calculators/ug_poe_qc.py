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


#corrpoe is the function I call to calculate POE.
def corrpoe(df, n, qcflag=False, **config):
    """
    Calculates POE using Monte Carlo approach
    :param df:          input pandas dataframe with input columns.
	:param n:           number of iterations to run Monte Carlo on.
    :return:            returns a pandas dataframe (tabular object) with the inputs for each feature, and 33 columns for leak, rupture, and total POE for the year of analysis, and each of year 1 through year 10 after year of analysis.
    """
    # i is the Number of features, equal to number of rows in csv file
    i = df.shape[0]
    now = datetime.date.today()
    
    # Pipe properties extracted from the dataframe df.
    # Please follow the "unpacker.py" file for explanation as to how this is carried out.
    ODi, WTm, Sm, Tm, OPm, Inst = unpacker.pipe_prop(df)
    pipe_age = ((now - Inst).dt.days.values) / 365.25
    
    # Feature properties extracted from the dataframe df.
    # Please follow the "unpacker.py" file for explanation as to how this is carried out.
    fPDP, fL_measured, fW_measured, fstatus, ftype, fchainage = unpacker.feature_dim(df)
    fID = df['FeatureID'].values

    # Inline inspection range properties extracted from the dataframe df.
    # Please follow the "unpacker.py" file for explanation as to how this is carried out.
    Insp, vendor, tool = unpacker.range_prop(df)

    # Calculating the time difference between date of ILI and today, in fractional years.
    # if config['CORR_YEAR_START'] is None:
    #     corr_start = Inst
    # corr_start = Insp + ((Insp - Inst).dt.values)/2
    time_delta = ((now - Insp).dt.days.values) / 365.25

	# defining the function for model error. This function uses the gamma.ppf() function part of the import statement above. The equivalent Excel function is GAMMA.INV()
    def model_error(p):
        return 0.914 + gamma.ppf(p, 2.175, scale=0.225)

    # unit conversion to US units
    WT = WTm / 25.4
    S = (Sm * 1000) / 6.89476
    OPi = OPm / 6.89476

    # Statistics fueled by CSA Z662 Annex O Tables O.6 and O.7
    meanWT = 1.01
    sdWT = 0.01
    meanS = 1.09 #recommended from CSA Z662 2019
    sdS = 0.044 #recommended from CSA Z662 2019

    tool_D = 0.078 # in fraction WT
    # tool_L = 7.80 / 25.4 # in inches
    tool_L = 0.61 # in inches

    # OD and MOP are not distributed in this approach. This function creates an array of size (n,i) of repeating OD and OP values.
    # Please refer to "distributer.py" file for details.
    OD, OP = distributer.tiler(ODi, OPi, tuple_size=(n, 1))

    # Setting a random seed
    np.random.seed()

	#  Initializing arrays of size (n, i), containing random numbers from 0 to 1, for each of model error, wall thickness, grade, feature length, feature depth, and growth rate.
    #  Please refer to "distributer.py" file for details.
    mE_n_1, WT_n_2, S_n_3, fL_n_5, fD_n_6 = distributer.random_prob_gen(5, iterations=n, features=i)

    # the following 2 statements distribute the wall thickness and grade, using normal distributions. Equivalent Excel equation is NORM.INV()
    WTd = norm.ppf(WT_n_2, loc=WT * meanWT, scale=WT * meanWT * sdWT)
    Sdist = norm.ppf(S_n_3, loc=S * meanS, scale=S * meanS * sdS)

    # the following statement distributes the feature length in inches, using normal distribution, and ensures the length is not distributed below 0.0 mm.
    fL_run = np.maximum(0, norm.ppf(fL_n_5, loc=fL_measured * 1.0 * (1 / 25.4), scale=tool_L))

    # the following statement distributes the feature depth in inches, using normal distribution, and ensures the depth is not distributed below 0.0 mm.
    fD_run = np.maximum(0, norm.ppf(fD_n_6, loc=fPDP * WT * 1.0, scale=tool_D * WT))

    # the following statement applies the growth rate assumptions. where vendor provided growth rate data is provided, use it, otherwise use the shape and scale parameters as part of the sample_inputs.csv
    # the np.where() function is used to make conditional checks of arrays.
    # Please refer to the "cgr.py" file for the details on the cgr_mpy() and cgr_weibull() functions
    if config['LENGTH_GROW_FLAG']:
        if config['USER_L_CGR'] is None:
            fL_GR = cgr.half_life(fL_run, Inst, Insp)
        else:
            fL_GR = distributer.tile(1.0*config['USER_L_CGR'], tuple_size=(n, 1))
    if config['USER_D_CGR'] is None:
        fD_GR = np.where((Insp-Inst).dt.days.values/365.25 < 20,
                        cgr.pct_wt(WT),
                        cgr.half_life(fD_run, Inst, Insp))

    # calculate the length at the year of analysis
    fL = fL_run + fL_GR * time_delta

    # calculating the depth at the year of analysis
    fD = fD_run + fD_GR * time_delta

    # calculating the model error
    modelError = model_error(mE_n_1)

    # calculating the Failure Stress in psi. 
    # Please refer to the "failureStress.py" file for the details on the calculation.
    failure_stress = failureStress.modified_b31g(OD, WTd, Sdist, fL, fD, units="US") * modelError

    # calculating Failure pressure in psi, using the Barlow formula
    failPress = 2 * failure_stress * WTd / OD

    # comparing the failure pressures to the operating pressure of the pipeline. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    # Please refer to the "corrLimitStates.py" file for details on the limit state checking.
    ruptures, rupture_count = corrLimitStates.ls_corr_rupture(failPress, OP, bulk=True)

    # comparing the depth at year of analysis with the wall thickness to see if the 80.0%WT criteria is exceeded. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    # Please refer to the "corrLimitStates.py" file for details on the limit state checking.
    leaks, leak_count = corrLimitStates.ls_corr_leak(WTd, fD, bulk=True)

    # checking both leak and rupture limit states at the same time. This statement sums the n rows vertically for each i, to determine the number of times the limit was exceeded.
    # Please refer to the "corrLimitStates.py" file for details.
    fails, fail_count = corrLimitStates.ls_corr_tot(ruptures, leaks, bulk=True)

    # setting up the projection of 10 years after year of analysis.
    failure_projection = {'leaks':[None]*20,
                         'ruptures':[None]*20,
                         'fails':[None]*20}
    
    # Calculate POE for each of 10 years from year of analysis
    for x in range(20):
        # Add 1 year of corrosion growth
        fL += fL_GR*(1.0)
        fD += fD_GR*(1.0)
        
        # recalculate Failure Stress in psi
        failure_stress = failureStress.modified_b31g(OD, WTd, Sdist, fL, fD, units="US") * modelError

        # recalculate Failure pressure in psi
        failPress = 2 * failure_stress * WTd / OD

        # redetermine limit state exceedances
        rupture_proj, failure_projection['ruptures'][x] = corrLimitStates.ls_corr_rupture(failPress, OP, bulk=True)
        leak_proj, failure_projection['leaks'][x] = corrLimitStates.ls_corr_leak(WTd, fD, bulk=True)
        _, failure_projection['fails'][x] = corrLimitStates.ls_corr_tot(rupture_proj, leak_proj, bulk=True)
        
    # setting up a container to hold all the information to be returned as results
    return_dict = {"FeatureID":fID,
                    "OD":ODi,
                    "WT":WT,
                    "S":S,
                    "OP":OPi,
                    "Inst":Inst,
                    "Insp":Insp,
                   "PDP_frac": fPDP,
                   "flength": fL_measured,
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
                   "nan": np.zeros((i,))
                   }
    # converting the container to a tabulat object using pandas
    return_df = pd.DataFrame(return_dict).set_index('FeatureID')
    
    if qcflag:
        fIDd, Insp_d, Inst_d, time_delta_d = distributer.tiler(fID, Insp, Inst, time_delta, tuple_size=(n, 1))
        qc_shape = (n*i)
        qc_dict = {
            "FeatureID":fIDd.T.reshape(qc_shape),
            "OD":OD.T.reshape(qc_shape),
            "WT":WTd.T.reshape(qc_shape),
            "S":Sdist.T.reshape(qc_shape),
            "OP":OP.T.reshape(qc_shape),
            "Inst":Inst_d.T.reshape(qc_shape),
            "Insp":Insp_d.T.reshape(qc_shape),
            'time_delta':time_delta_d.T.reshape(qc_shape),
            "fL_run": fL_run.T.reshape(qc_shape),
            "fL_GR":fL_GR.T.reshape(qc_shape),
            "fD_run":fD_run.T.reshape(qc_shape),
            "fD_GR": fD_GR.T.reshape(qc_shape),
            "modelError":modelError.T.reshape(qc_shape),
            "failure_stress": failure_stress.T.reshape(qc_shape),
            "failPress":failPress.T.reshape(qc_shape),
            "rupture_count_YOA": ruptures.T.reshape(qc_shape),
            "leak_count_YOA": leaks.T.reshape(qc_shape),
            "fail_count_YOA": fails.T.reshape(qc_shape),
        }
    else:
        qc_dict={}

    qc_df = pd.DataFrame(qc_dict)

    return return_df, qc_df


def index_marks(nrows, chunk_size):
    return range(chunk_size, np.ceil(nrows / chunk_size).astype(int) * chunk_size, chunk_size)

def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

#the next statement loads the CSV into a pandas dataframe
CSV_FILE_NAME = 'ug_poe_inputs.csv'
CSV_FILE_NAME = abspath(join(dirname(__file__), CSV_FILE_NAME))

df = pd.read_csv(CSV_FILE_NAME, header=0)
print('Features loaded...')

filters = {'FeatureIDs':[
                            # 422434,
                            # 423061,
                            # 422713,
                            # 421758,
                            # 422909,
                            # 422911,
                            # 421659,
                            # 422106,
                            # 422842,
                            # 422668,
                            # 421508,
                        ]}
print(f'Filtering features for {filters["FeatureIDs"]}.')
if len(filters['FeatureIDs']) == 0:
    pass
else:
    df = df.loc[lambda x: (x.FeatureID.isin(filters['FeatureIDs']))]

configuration ={'CORR_YEAR_START':None,
                'POF_THRESH':0.0023,
                'LENGTH_GROW_FLAG':True,
                'USER_L_CGR':None,
                'USER_D_CGR':None}

print('Calculating POE...')
num = 1_000_000

# chunks = split(df, 5)
s1 = time.time()
if not('chunks' in locals()):
    result, qc = corrpoe(df,num, qcflag=True,**configuration)
else:
    results = [corrpoe(df,num, qcflag=True,**configuration)[0] for x in chunks]
    result = pd.concat(results) 
print(f'Done. {time.time() - s1:.4f} seconds')

result.iloc[:, 9:] = result.iloc[:,9:].apply(lambda x: x/num)#.applymap('{:.3e}'.format)
result.loc[:,'iterations'] = result.loc[:,'iterations'].apply('{:,.0f}'.format)

print(result)

#EXPORTING
print('Writing Outputs to Excel...')
s2 = time.time()
with pd.ExcelWriter(abspath(join(dirname(__file__), 'UG_POE_output.xlsx')), engine='xlsxwriter',date_format='YYYY-MM-DD',datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
    workbook = writer.book
    df.to_excel(writer, sheet_name='inputs')
    result.to_excel(writer, sheet_name='outputs')
print(f'Done. {time.time() - s2:.4f} seconds')

qc_filters = {'FeatureIDs':[
                       
                        ]}

print(f'Filtering QC features for {qc_filters["FeatureIDs"]}.')

if len(qc_filters['FeatureIDs']) == 0:
    pass
else:
    qc = qc.loc[lambda x: (x.FeatureID.isin(qc_filters['FeatureIDs']))]
    print('Writing Intermediates to Excel...')
    s3 = time.time()
    with pd.ExcelWriter(abspath(join(dirname(__file__), 'UG_POE_output_intermediates.xlsx')), engine='xlsxwriter',date_format='YYYY-MM-DD',datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
        workbook = writer.book
        qc.to_excel(writer, sheet_name='intermediates')
    
    print(f'Done. {time.time() - s3:.4f} seconds')
