from scipy.stats import norm, gamma
import time, datetime
import numpy as np, pandas as pd

from simpoe.cracks import failureStress, crackLimitStates
from simpoe import unpacker, model_constants, distributer, cgr

now = datetime.date.today()
start = time.time()

CSV_FILE_NAME = 'feature_list.csv'
##CSV_FILE_NAME = r"C:\Users\aeborjas\Documents\Projects\Python\experimental\simpoe\calculator_template-2018-10-15-21-40-4.csv"

df = pd.read_csv(CSV_FILE_NAME, header=0)

pd.set_option('display.max_columns',500)

#Function starts----------------------------------------------------------------------------------
def mdpoe(df,n):

    #Number of features is equal to number of rows in csv file
    i = df.shape[0]

    #Pipe properties
    OD, WTm, Sm, Tm, OPm, Inst = unpacker.pipe_prop(df)

    #Feature properties
    cPDP, cL_measured, cW_measured, cstatus, ctype, cchainage = unpacker.feature_dim(df)

    #Inline inspection range properties
    Insp, vendor, tool = unpacker.range_prop(df)

    time_delta = ((now-Insp).dt.days.values)/365.25

    #Growth Rate mechanism
    shape, scale = 2.55, 0.10

    pivar = np.pi

    #Young's modulus in psi
    E = model_constants.const['E']

    #Crack fracture area in sq. inches
    fa = model_constants.const['fa']

    meanOD = model_constants.pipe_specs['od']['mean']
    sdOD = model_constants.pipe_specs['od']['sd']
    meanWT = model_constants.pipe_specs['wt']['mean']
    sdWT = model_constants.pipe_specs['wt']['sd']
    meanS = model_constants.pipe_specs['s']['mean']
    sdS = model_constants.pipe_specs['s']['sd']
    meanE = model_constants.pipe_specs['youngs']['mean']
    sdE = model_constants.pipe_specs['youngs']['sd']

    #Unit conversion to US units
    WT = WTm/25.4
    S = (Sm*1000)/6.89476
    T = Tm*0.7375621
    OP = OPm/6.894069706666676

    sdDepthTool = np.where(WTm < 10, 0.117, 0.156)  #fraction of WT

    defectTol = {
        "AFD": {
            "sdPDP": 0.195,  # fraction of WT
            "sdL": 15.6  # in mm
        },
        "EMAT": {
            "sdPDP": sdDepthTool,
            "sdL": 7.8  # in mm
        }
    }

    tool_D = 0.121*WTm
    tool_L = 8.034  #in mm

    # tool_D = np.where(vendor=="Rosen",defectTol["Rosen"]["sdPDP"],defectTol["PII"]["sdPDP"])
    # tool_L = np.where(vendor=="Rosen",defectTol["Rosen"]["sdL"],defectTol["PII"]["sdL"])

    #non-distributed variables
    OP, T = distributer.tiler(OP, T, tuple_size=(n, 1))

    #distributed variables
    np.random.seed()

    OD_n_1, WT_n_2, S_n_3, Y_n_4, cL_n_5, cD_n_6 = distributer.random_prob_gen(6, iterations=n, features=i)

    #distributed variables imperial
    ODd = norm.ppf(OD_n_1, loc=OD*meanOD, scale=OD*sdOD)
    WTd = norm.ppf(WT_n_2, loc=WT*meanWT, scale=WT*sdWT)
    Sdist = norm.ppf(S_n_3, loc=S*meanS, scale=S*sdS)
    Ydist = norm.ppf(Y_n_4, loc=E*meanE, scale=E*sdE)

    #crack length in inches
    cL_run = np.maximum(1, norm.ppf(cL_n_5, loc=cL_measured * 1.0, scale=tool_L)) / 25.4
    #HALD LIFE GROWTH RATE DOESNT WORK. NEED TO FIX.
    # cL_GR = cgr.half_life(cL_run, Inst, Insp)
    cL = cL_run +  0     #not growing features

    #Crack detpth in inches
    cD_run = np.maximum(0, norm.ppf(cD_n_6, loc=cPDP*WTm*1.0, scale=tool_D))/25.4
    #HALD LIFE GROWTH RATE DOESNT WORK. NEED TO FIX.
    # cD_GR = cgr.half_life(cD_run, Inst, Insp)
    cD = cD_run +  0     #not growing features

    failStress = failureStress.modified_lnsec(ODd, WTd, Sdist, T, Ydist, cL, cD, units="US")

    #Failure pressure in psi
    failPress = 2*failStress*WTd/ODd

    fsNaNs = np.extract(np.isnan(failStress),failStress)
    fsNaNs_count = fsNaNs.size

    ruptures, rupture_count = crackLimitStates.ls_md_rupture(failPress, OP, bulk=True)

    leaks, leak_count = crackLimitStates.ls_md_leak(WTd, cD, thresh=1.0, bulk=True)

    fails, fail_count = crackLimitStates.ls_crack_tot(ruptures, leaks, bulk=True)

    return_dict = {"fail_count":fail_count,
                    "iterations":np.size(fails, axis=0),
                    "rupture_count":rupture_count,
                    "leak_count":leak_count,
                    "nan": fsNaNs_count,#np.zeros((i,)),
                    "PDP_frac":cPDP,
                    "clength":cL_measured}

    return_df = pd.DataFrame(return_dict)

    return return_df


if __name__ == '__main__':
    iter = 100000
    order = ['PDP_frac', 'clength', 'iterations', 'POE_l', 'POE_r', 'POE']
    result = mdpoe(df,iter)
    end = time.time()
    result["POE"] = result["fail_count"] / result["iterations"]
    result["POE_l"] = result["leak_count"] / result["iterations"]
    result["POE_r"] = result["rupture_count"] / result["iterations"]

    result["1-POE"] = 1 - result["POE"]
    agg_POE = 1 - np.prod(result["1-POE"])

    print("Manufacturing Defects", "POE Simulation\n")
    print(result[order], '\n')
    print("Aggregated POE for these features is {}.\n".format(agg_POE))
    print("Calculation took {} seconds.".format(end - start))
