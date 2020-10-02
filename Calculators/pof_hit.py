from scipy.stats import norm, gamma
import time, datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from os.path import abspath, dirname, join

def random_prob_gen(x, iterations=1, features=1):
    temp=[]
    for var in range(x):
        temp.append(np.random.rand(iterations,features))
    return np.array(temp, dtype=np.float64)

def dist_weibull(p,shp=1.2,scl=3.2):
    '''
    Returns the imperfection growth rate based on a Weibull distribution inverse cumulative density function, using shape and scale parameters.
    Requires Numpy module
    :param p:   random number between 0 and 1
    :param shp: shape parameter, default of 1.439
    :param scl: scale parameter, default of 0.1
    :return:    imperfection growth rate in mm/yr
    '''
    return  scl*np.power( -np.log(1.0-p) ,1.0/shp)

def excavator_force_class1_2(rand,force_reduc_1,force_reduc_2):
    F = 3778.585582*np.power(rand,6) + 1316.518325*np.power(rand,5) - 13422.057887*np.power(rand,4) + 12439.187305*np.power(rand,3) - 3902.560258*np.power(rand,2) + 517.745042*rand + 15.848802
    return force_reduc_1*force_reduc_2*F

def excavator_force_class3_4(rand,force_reduc_1,force_reduc_2):
    F = 22059.01771688*np.power(rand,6) - 61203.11722147*np.power(rand,5) + 62797.73990864*np.power(rand,4) - 28797.41597699*np.power(rand,3) + 5785.487471463*np.power(rand,2) - 296.1418563905*rand + 26.36112543804
    return force_reduc_1*force_reduc_2*F

def NG18QFactor(od, wt, flow, T, dD, gD, gL, gA=1.00, units="SI"):

    if units=="SI":
        od_c = od/25.4
        wt_c = wt/25.4
        flow_c = flow/6.89476
        T_c = T/1.3558
        dD_c = dD/25.4
        gD_c = gD/25.4
        gL_c = gL/25.4
    else:
        od_c = od
        wt_c = wt
        flow_c = flow
        T_c = T
        dD_c = dD
        gD_c = gD
        gL_c = gL

    c2 = 300.0
    c3 = 90.0

    #Maxey Q Parameter (ft-lbs/in)
    Q = np.maximum(T*( (od_c*0.5*wt_c)/( dD_c*gD_c*(gL_c*0.5)*gA ) ), c2)

    # stress in ksi
    failStress = flow_c*(np.power(Q-c2,0.6)/c3)

    return failStress, Q

def force_for_puncture(od, wt, uts, tooth_perimeter, model_error=0.):
    #returns force required to puncture pipe in kN
    failForce = 0.000464*np.power(wt*uts,1.087)*tooth_perimeter
    # failForce = (1.17-0.0029*(od/wt))*tooth_perimeter*wt*uts+model_error
    return failForce

def ls_gouge_dent_fail(value, limit, bulk=False):

	fails = value < limit
	fails = fails.astype(int)
	if bulk:
		fail_count = np.sum(fails, axis=0)
	else:
		fail_count = np.sum(fails)
	return fails, fail_count

def ls_puncture_fail(value, limit, bulk=False):

	fails = limit <= value
	fails = fails.astype(int)
	if bulk:
		fail_count = np.sum(fails, axis=0)
	else:
		fail_count = np.sum(fails)
	return fails, fail_count

def ls_total(fail_1, fail_2, bulk=False):

    fails = np.maximum(fail_1, fail_2)
    if bulk:

        fail_count = np.sum(fails, axis=0)
    else:

        fail_count = np.sum(fails)
    return fails, fail_count

def pof_hit(df,n):

    i = df.shape[0]

# inputs in US units
    OD = df['OD_inch'].values   #inches
    WT = df['WT_mm'].values  #mm
    S = df['grade_MPa'].values    #MPa
    OP = df['MAOP_kPa'].values #kPa
    T = df['T_J'].values   #J
    class_loc = df['class_loc'].values

    sdOD = 0.0006
    sdWT = 0.01
    sdS = 0.035
    sdT = 0.0001*1.3558   #T



    conditions = [S<=25,
                 S<=30,
                 S<=35,
                 S<=42,
                 S<=46,
                 S<=52,
                 S<=56,
                 S<=60,
                 S<=65,
                 S<=70,
                 S>70]

    UTS_list = np.array([59.,62.5,72.5,72.5,76.,78.6,82.5,84.,87.,92.7,98.5])
    sdUTS_list = np.array([2.065,2.1875,2.5375,2.5375,2.66,2.751,2.8875,2.94,3.045,3.2445,3.4475])

    UTS = np.select(conditions,UTS_list)
    sdUTS = np.select(conditions,sdUTS_list)

    #converting units to metric
    OD = OD*25.4    #mm
    UTS = UTS*6.89476   #MPa
    sdUTS = sdUTS*6.89476 #MPa

    opStr = (OP*OD/1000)/(2*WT) #MPa
    flStr = 1.1*S

    #distributed variables
    np.random.seed()

    OD_n_1, WT_n_2, S_n_3, UTS_n_4, T_n_5, EX_n_6, FREX_n_7, gL_n_8, gA_n_9 = random_prob_gen(9, iterations=n, features=i)

    ODd = norm.ppf(OD_n_1, loc=OD, scale=sdOD*OD)
    WTd = norm.ppf(WT_n_2, loc=WT, scale=sdWT*WT)
    flStr_d = norm.ppf(S_n_3, loc=flStr, scale=sdS*flStr) + (10*6.89476)
    UTSd = norm.ppf(UTS_n_4, loc=UTS, scale=sdUTS)
    Td = norm.ppf(T_n_5, loc=T, scale=sdT)
    Td_23 = Td*(2/3)
    
    opStr_d = (OP*ODd/1000)/(2*WTd)

    #random excavator force (kN)    
    #Force Reduction Factor to Account for Operator Control (Gouge in Dent Calibration Factor) (p21 of C-Fer Report)
    op_gouge_control = 0.5216

    exFd = np.where(np.logical_or(class_loc=='class 1',class_loc=='class 2'),
                    excavator_force_class1_2(EX_n_6,op_gouge_control,FREX_n_7),
                    excavator_force_class3_4(EX_n_6,op_gouge_control,FREX_n_7)
                    )

    #random excavator mass (tonnes)
    exMd = np.exp(np.log(exFd/14.2)/0.928)

    #random excavator 1/2 tooth perimeter (L+W) (mm)
    exLWd = 29.4*np.power(exMd,0.4)

    #random excavator tooth length L (mm)
    exLd = 24.6*np.power(exMd,0.42)

    #Gouge length (mm) and Gouge Depth (mm)
    gL = dist_weibull(gL_n_8)*25.4
    gD = np.maximum((0.0003268381*exFd - 0.005851569),0.000001)*25.4
    gA = 0.2171976296487*np.power(gA_n_9,2) - 1.218006976505*gA_n_9 + 1.001165088866

    #Pipe resistance parameter mmsqrt(N)
    resistance = np.power(np.power(WTd,3)*UTSd*exLd,0.5)*(1+0.7*(OP*ODd)/(WTd*UTSd*1000.0))

    #dent depth (mm)
    dD_aRR = np.minimum(np.where(resistance <= 2000,
                        np.power(exFd/(0.007*resistance),2),
                        np.power(exFd/(0.31*np.power(resistance,0.5)),2)
                        ),
                        ODd-2*WTd)
    dD_bRR = np.minimum(1.43*dD_aRR,
                        ODd-2*WTd)

    #Test of Gouge-in-dent failure: NG-18 Q-Factor Relation (results in stress:ksi and Q:ftlb/in)
    failStress, Q = NG18QFactor(ODd, WTd, flStr_d, Td_23, dD_bRR, gD, gL, gA)
    failStress = failStress*6.89476
    fails_gd, fail_count_gd = ls_gouge_dent_fail(failStress, opStr_d, bulk=True)

    #Test of Failure due to Puncture
    op_puncture_control = 0.81
    failForce = force_for_puncture(ODd, WTd, UTSd, exLWd)
    fails_p, fail_count_p = ls_puncture_fail(exFd*(op_puncture_control/op_gouge_control),failForce, bulk=True)

    fails, fail_count = ls_total(fails_gd, fails_p, bulk=True)

    return_dict = {'OD': OD,
                    'WT': WT,
                    'S': S,
                    'OP': OP,
                    'T': T,
                    'UTS': UTS,
                    "iterations": np.size(fails, axis=0),
                    "fail_gD_pof": fail_count_gd/n,
                    "fail_p_pof": fail_count_p/n,
                    "fail_pof": fail_count/n}

    return_df = pd.DataFrame(return_dict)

    if i==1:
        qc_dict = {'OD': ODd.reshape(-1),
                    'WT': WTd.reshape(-1),
                    'UTS': UTSd.reshape(-1),
                    'T': Td.reshape(-1),
                    'Td_23':Td_23.reshape(-1),
                    'opStr':opStr_d.reshape(-1),
                    'flStr':flStr_d.reshape(-1),
                    'excavator_force':exFd.reshape(-1),
                    'excavator_mass':exMd.reshape(-1),
                    'half_tooth_perimeter':exLWd.reshape(-1),
                    'tooth_length':exLd.reshape(-1),
                    'gouge_length':gL.reshape(-1),
                    'gouge_depth':gD.reshape(-1),
                    'gouge_orientation_factor':gA.reshape(-1),
                    'resistance':resistance.reshape(-1),
                    'dent_depth_after_RR':dD_aRR.reshape(-1),
                    'dent_depth_before_RR':dD_bRR.reshape(-1),
                    'Q':Q.reshape(-1),
                    'failStress':failStress.reshape(-1),
                    'failForce':failForce.reshape(-1),
                    "fail_gD_count": fails_gd.reshape(-1),
                    "fail_p_count": fails_p.reshape(-1),
                    "fail_count": fails.reshape(-1)}

        qc_df = pd.DataFrame(qc_dict)
    else:
        qc_df = None

    return return_df, qc_df


scenarios = 100

df = dict(OD_inch=np.full(scenarios,36.0*25.4),
        WT_mm=np.linspace(3.20,18.9,scenarios),
        grade_MPa=np.full(scenarios,359.0),
        MAOP_kPa=np.full(scenarios,7000.0),
        T_J=np.full(scenarios,20.0),
        class_loc=np.full(scenarios,"class 4")
        )
df = pd.DataFrame(df)

result, qc = pof_hit(df, 100_000)
print(result)