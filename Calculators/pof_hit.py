from scipy.stats import norm, gamma
import time, datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib as mpl
from mpl_toolkits import mplot3d
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
    # excavator force (kN)
    F = 3778.585582*np.power(rand,6) + 1316.518325*np.power(rand,5) - 13422.057887*np.power(rand,4) + 12439.187305*np.power(rand,3) - 3902.560258*np.power(rand,2) + 517.745042*rand + 15.848802
    return force_reduc_1*force_reduc_2*F

def excavator_force_class3_4(rand,force_reduc_1,force_reduc_2):
    # excavator force (kN)
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

    c2 = 300.0 #this is in ft-lbs/inch. Apparently, the SI counterpart is 16. J/mm
    c3 = 90.0   #this is in (ft-lbs/inch)^0.6. Apparently, the SI counterpart is 4.8 (J/mm)^0.6

    #Maxey Q Parameter (ft-lbs/in)
    Q = np.maximum(T_c*( (od_c*0.5*wt_c)/( dD_c*gD_c*(gL_c*0.5)*gA ) ), c2)

    # stress in ksi
    failStress = flow_c*(np.power(Q-c2,0.6)/c3)

    return failStress, Q

def dentGouge_BSIPD6493(od, wt, S, uts, T, P, dD_bRR, gD, gL, q, exL, gA=1.00, units="SI"):
    '''
    od  - outside diameter in mm
    wt  - wall thickness in mm
    S   - SMYS in MPa
    uts - ultimate tensile strength in MPa
    T   - charpy 2/3 toughness in J
    P   - pressure in kPa
    dD  - dent aepth at zero pressure, in mm
    gD  - gouge depth in mm
    q   - excavator normal impact force in kN
    exL - excavator tooth length in mm
    gA  - random gouge orientation factor, dimensionless

    return failStress   - calculated hoop stress resstance in MPa
    '''
    cv0 = 110.3     #empirical coefficient in J, for when E is in MPa and ac is in mm2
    ac = 53.6       #cross sectional area of 2/3 charpy v-notch specimens in mm2
    E = (30.e6)*6.89476/1000. #Young's modulus in MPa
    
    gD_wt = gD/wt
    # i believe the following 2 parameters are dimensionless
    Yb = 1.12 - 1.39*(gD_wt) + 7.32*np.power(gD_wt, 2) - 13.1*np.power(gD_wt, 3) + 14.0*np.power(gD_wt, 4)
    Ym = 1.12 - 0.23*(gD_wt) + 10.6*np.power(gD_wt, 2) - 21.7*np.power(gD_wt, 3) + 30.4*np.power(gD_wt, 4)

    # dD0 = 1.43*np.power(q / ( 0.49*np.power(exL * S * wt, 0.25)*np.sqrt(wt + 0.7*1000.*P/uts) ), 2.381)
    dD0 = dD_bRR

    # CSAZ662 Annex O
    # i believe the following 4 parameters are dimensionless
    Sm = 1. - (1.8*dD0/od)
    Mt = np.power(1 + (0.52*np.power(gL*gA,2))/(od*wt), 0.5)

    #CSA Z662 indicates that for this term, the gouge depth is divided solely by the Folias factor. However
    #if this is true, then the units of this factor would be in mm/MPa, which wouldn't cancel out in the main equation
    b2 = (Sm*(1 - gD/(Mt))) / (1.0*S*(1 - gD/wt))
    b1 = Sm*Ym + (5.1*Yb*dD0/wt)

    # Stress intensity, in MPa-m**0.5. It's important to pay attention to units. Here should be fine, if we keep cv0 as J, ac in mm2, and E in MPa
    KIC = np.sqrt(E * cv0 / ac) * np.power(T / cv0, 0.95)

    failStress = (2/(np.pi*b2) )*np.arccos(np.exp( -125.*np.power(np.pi,2)*np.power(b2/b1,2)*(np.power(KIC,2)/(np.pi*gD)) ))
    
    return failStress

def force_for_puncture(od, wt, uts, tooth_perimeter, model_error=0.):
    #returns force required to puncture pipe in kN
##    failForce = 0.000464*np.power(wt*uts,1.087)*tooth_perimeter
    failForce = (1/1000.0)*(1.17-0.0029*(od/wt))*tooth_perimeter*wt*uts
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

    sdOD = 0.0006   #COV fraction
    sdWT = 0.01     #COV fraction
    sdS = 0.035     #COV fraction
    sdT = 0.0001*1.3558   #T absolute, in J

    Suts = np.round(S/6.89476)

    conditions = [Suts<=25,
                 Suts<=30,
                 Suts<=35,
                 Suts<=42,
                 Suts<=46,
                 Suts<=52,
                 Suts<=56,
                 Suts<=60,
                 Suts<=65,
                 Suts<=70,
                 Suts>80]

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
    
    # puncture_force_error = norm.ppf(mE_n_10, loc=0.833, scale=26.7)
    
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

    #Test of dent-gouge failure: CSA Z662 Annex O Clause O.2.6.3.2, Hopkins et al. 1992 model for EPRG modified by Francis et al. 1997
    # failStress = dentGouge_BSIPD6493(ODd, WTd, flStr_d, UTSd, Td_23, OP, dD_bRR, gD, gL, exFd, exLd, gA)
    fails_gd, fail_count_gd = ls_gouge_dent_fail(failStress, opStr_d, bulk=True)

    #Test of Failure due to Puncture
    op_puncture_control = 0.81
    failForce = np.maximum(force_for_puncture(ODd, WTd, UTSd, exLWd),0.00)
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


od = np.array([4., 6.625, 8.625, 10.75, 12.75, 16., 20., 24., 30., 32., 34., 36.])
wt = np.linspace(3.20,18.9,num = od.size)
t = np.linspace(10.0,25.0,num = od.size)
##s = np.array([25.,30.,35.,42.,46.,52.,56.,60.,65.,70.,80., 90.])*6.89476

od, wt, t = np.meshgrid(od, wt, t)

scenarios = od.size

df = dict(OD_inch=od.reshape(-1),
       WT_mm=wt.reshape(-1),
       grade_MPa=np.full(scenarios,359.0),
       MAOP_kPa=np.full(scenarios,7000.0),
       T_J=t.reshape(-1),
       class_loc=np.full(scenarios,"class 4")
       )

# df = dict(OD_inch=[24.],
#         WT_mm=[6.35],
#         grade_MPa=[359.0],
#         MAOP_kPa=[5382.418],
#         T_J=[21.7],
#         class_loc=['class 2']
#         )

df = pd.DataFrame(df)
##df.MAOP_kPa = 0.80*2.0*df.grade_MPa*1000*df.WT_mm/(df.OD_inch*25.4)

result, qc = pof_hit(df,10000)
print(result)

def generate_plots(df, col1, col2, convert_xticks=False):
    
    pof = df.pivot_table(index=col1,columns=col2,values="fail_pof").values
    pofgd = df.pivot_table(index=col1,columns=col2,values="fail_gD_pof").values
    pofp = df.pivot_table(index=col1,columns=col2,values="fail_p_pof").values

    par1 = np.sort(df.loc[:,col1].unique())
    par2 = np.sort(df.loc[:,col2].unique())

    par1, par2 = np.meshgrid(par1, par2)

    fig, ax = plt.subplots(3,1, sharex=True)
    levels = np.geomspace(0.001,1.0, 10)
    cpf = ax[0].contourf(par1, par2, pof, levels=levels, cmap=cm.Reds)
    line_colors = ['black' for l in cpf.levels]
    cp = ax[0].contour(par1, par2, pof, levels=levels, colors=line_colors)
    ax[0].clabel(cp, fontsize=10, colors=line_colors)
    ax[0].set_title("Total POF")

    cpf = ax[1].contourf(par1, par2, pofgd, levels=levels, cmap=cm.Reds)
    cp = ax[1].contour(par1, par2, pofgd, levels=levels, colors=line_colors)
    ax[1].clabel(cp, fontsize=10, colors=line_colors)
    ax[1].set_title("Gouge-in-Dent POF")

    cpf = ax[2].contourf(par1, par2, pofp, levels=levels, cmap=cm.Reds)
    cp = ax[2].contour(par1, par2, pofp, levels=levels, colors=line_colors)
    ax[2].clabel(cp, fontsize=10, colors=line_colors)
    ax[2].set_title("Puncture Resistance POF")

    ax[2].set_xlabel(col1)
    ax[1].set_ylabel(col2)

    if convert_xticks:
        locs, _ = plt.xticks()
        plt.xticks(locs, (locs/25.4).round(0))

    fig.suptitle(f"{col1} versus {col2}")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cpf, cax=cbar_ax)

generate_plots(result, "OD", "WT", convert_xticks=True)
# generate_plots(result, "OD", "T", convert_xticks=True)
# generate_plots(result, "WT", "T")

def generate_3dplot(df, col1, col2, col3):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    pof_12 = df.pivot_table(index=col1,columns=col2,values="fail_pof").values
    pof_23 = df.pivot_table(index=col2,columns=col3,values="fail_pof").values
    pof_13 = df.pivot_table(index=col1,columns=col3,values="fail_pof").values

    par1 = np.sort(df.loc[:,col1].unique())
    par2 = np.sort(df.loc[:,col2].unique())
    par3 = np.sort(df.loc[:,col3].unique())

##    par1, par2, par3 = np.meshgrid(par1, par2, par3)

##    ax.plot_surface(od, wt, pof, cmap=cm.Reds)
    cset = ax.plot_surface(pof_12, pof_23, pof_13, cmap=cm.Reds)

    cset = ax.contourf(pof_12, pof_23, pof_12, zdir='z', offset=0., cmap=cm.Reds)
    cset = ax.contourf(pof_12, pof_23, pof_13, zdir='x', offset=0., cmap=cm.Reds)
    cset = ax.contourf(pof_12, pof_23, pof_13, zdir='y', offset=0., cmap=cm.Reds)

    ax.set_xlabel(col1)
    ax.set_xlim(0,)
    ax.set_ylabel(col2)
    ax.set_ylim(0,)
    ax.set_zlabel(col3)
    ax.set_zlim(0,)
    return (par1, par2, par3), (pof_12, pof_23, pof_13)

##x = generate_3dplot(result,"OD","WT","T")

##fig, axs = plt.subplots(1,2)
##result.set_index("WT")[["fail_gD_pof","fail_p_pof"]].plot(ax=axs[0])
##result.set_index("WT")[["fail_pof"]].plot(ax=axs[1])
##
##for ax in axs:
##    ax.set_yscale("log")
##    ax.grid()
##    ax.set_xlabel("Sensitivity Parameter")
##    ax.set_ylabel("POF")

plt.show()
