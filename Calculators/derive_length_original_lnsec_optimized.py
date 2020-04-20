import math
import csv
import pandas as pd
from scipy import optimize

pd.set_option('display.max_columns',101)
pd.set_option('expand_frame_repr',True)

##DATABASE INPUTS------------------------------------------
#toughness in ft.lbs
cvn = 14.7512
#grade in psi
s = 52000.0
#diameter in in
od = 10.75
#wall thickness in in
wt = 0.25
#maximum operating pressure in psi
mop = 1030.0

input_data = (20,0.281102,52068.53,9.145771,1052.974)

def critical_flaw_size(le, od, wt, s, cvn, mop):
##    od, wt, s, cvn, mop, le = x[0], x[1], x[2], x[3], x[4], x[5]

    ##CONSTANTS------------------------------------------
    #young's modulus in psi
    E = 30000000.0
    #fracture area of charpy notch test in in2
    ac=0.124
    #flow stress in psi
    sf = s + 10000

    ##CALCULATIONS------------------------------------------
    #pipe radius in inches
    r = od/2
    #pipe stress from Barlow formula in psi
    stress = (mop*od)/(2*wt)

    ##ITERATION SETUP------------------------------------------
    #assumed 60%wt
    pdp=0.6

    elliptical_c = min(math.pi*le*0.25,math.sqrt(67.2*r*wt))
    c2rt= pow(elliptical_c,2)/(r*wt)

    Mt=max(math.sqrt(1+1.255*c2rt-0.0135*pow(c2rt,2)),pdp+0.001)
    Mp=(1-(pdp/Mt))/(1-pdp)
    XP = (12*E*math.pi*cvn)/(8*elliptical_c*ac*pow(sf,2))
    fail_stress = (sf/Mp)*math.acos(math.exp(-1*XP))

    e = abs(fail_stress - stress)
    
    return e

def lnsec(od, wt, s, cvn, mop, pdp, le):
    '''
    Modified ln-sec formula calculation
    
    @param od: outside diameter in inch.
    @param wt: wall thickness in inch.
    @param s: grade in psi.
    @param cvn: charpy toughness in ft-lbs.
    @param mop: maximum operating pressure in psi.
    @param pdp: peak depth percentage in fraction of wt.
    @param le: length of crack imperfection in inch.
    '''
    ##CONSTANTS------------------------------------------
    #young's modulus in psi
    E = 30000000.0
    #fracture area of charpy notch test in in2
    ac=0.124
    #flow stress in psi
    sf = s + 10000

    ##CALCULATIONS------------------------------------------
    #pipe radius in inches
    r = od/2
    #pipe stress from Barlow formula in psi
    stress = (mop*od)/(2*wt)

    elliptical_c = min(math.pi*le*0.25,math.sqrt(67.2*r*wt))
    c2rt= pow(elliptical_c,2)/(r*wt)

    Mt=max(math.sqrt(1+1.255*c2rt-0.0135*pow(c2rt,2)),pdp+0.001)
    Mp=(1-(pdp/Mt))/(1-pdp)
    XP = (12*E*math.pi*cvn)/(8*elliptical_c*ac*pow(sf,2))
    fail_stress = (sf/Mp)*math.acos(math.exp(-1*XP))

    if (fail_stress <= stress):
        fail_flag = 'FAIL'
    else:
        fail_flag = 'PASS'
    
    return stress, fail_stress, fail_flag


if __name__ == "__main__":
##    pass
    result = optimize.minimize(critical_flaw_size, 0.1, args=input_data, method='nelder-mead')
    print(result)
    
    with open(r'Z:\Plains Midstream\2018_01_IRAS_Implementation\3_Engineering\Quantitative Risk Run (V6)\Results & QC\22 Determining Critical Flaw Lengths for Pipelines\raw_data.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader,None)
        results = []
        for row in reader:
            ssid = row[0]
            line_name = row[1]
            od = float(row[4])
            wt = float(row[5])
            s = float(row[6])
            cvn = float(row[7])
            mop = float(row[10])
            ssb = row[2]
            sse = row[3]
            db = row[8]
            de = row[9]
            data = (od,wt,s,cvn,mop)
            x0= 0.1
            try:
                result = optimize.minimize(critical_flaw_size, x0, args=data, method='nelder-mead')
                cl = result.x[0]
            except Exception as e:
                cl = None
            
            results.append([ssid,line_name,od,wt,s,cvn,mop,cl,ssb,sse,db,de])
    headers = ['StationSeriesId','LineName','od_inch','wt_inch','grade_psi','toughness_ftlbs','mop_psi','critical_length_inch','SSBeginChainage_m','SSEndchainage_m','DataBeginChainage_m','DataEndChaiange_m']
    df = pd.DataFrame(results, columns=headers)

