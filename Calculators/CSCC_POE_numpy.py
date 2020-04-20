from scipy.stats import norm
import time
import datetime
import numpy as np
import pandas as pd

now = datetime.date.today()
start = time.time()

XLSX_FILE_NAME = 'N:\Python\Calculators\cscc_bulk_offline.xlsx'

df = pd.read_excel(XLSX_FILE_NAME, sheet_name=0)
df['ILIRStartDate'] = pd.to_datetime(df['ILIRStartDate']) #needs to receive date in M/D/Y format
pd.set_option('display.max_columns',700)
pd.set_option('display.max_rows',700)
pd.set_option('display.expand_frame_repr',False)

#Function starts
def csccpoe(df,n):
    pivar = np.pi

    i = df.shape[0]  
#Outside Diameter in mm    
    OD = df['OD_inch'].values
     
#Pipe Wall Thickness in mm
    WTm = df['WT_mm'].values

#Crack Depth fraction of WT
    cPDP = df['depth_fraction'].values

#Crack Length in mm
    cL_measured = df['width_mm'].values
    
#Year of Inspection

    Insp = df['ILIRStartDate'].dt.date
    time_delta = ((now-Insp).dt.days.values)/365.25
    vendor = "PII"

    #outside diameter stats
    OD = OD*25.4
    meanOD = 1.0    #fraction of OD
    sdOD = 0.0006   #fraction of OD

    #Wall thickness stats
    WT = WTm
    meanWT = 1.01   #fraction of WT
    sdWT = 0.01     #fraction of WT

    #Grade in ksi
#    S = (Sm*1000)/6.89476
    meanS = 1.10    #fraction of S
    sdS = 0.035     #fraction of S

    #Young's modulus in psi
#    E = 30000000 
    meanE = 1.0     #fraction of E
    sdE = 0.04      #fraction of E

    sdDepthTool = np.where(WTm<10,0.117,0.156)

    #tool tolerances
    defectTol = {
        "Rosen": {
            "sdPDP":0.78,   # in mm
            "sdL":7.80      #in mm
            },
        "RosenF": {
            "sdPDP":sdDepthTool*WTm, #in mm
            "sdL":7.80  #in mm
            },
        "PII": {
            "sdPDP": 0.31,  # in mm
            "sdL":6.10      #in mm
            }
        }

    tool_D = defectTol[vendor]["sdPDP"]
    tool_L = defectTol[vendor]["sdL"]
    
    np.random.seed()
    
    OD_n_1 = np.random.rand(n,i)
    WT_n_2 = np.random.rand(n,i)
    cL_n_5 = np.random.rand(n,i)
    cD_n_6 = np.random.rand(n,i)
    cGR_n_7 =np.random.rand(n,i)
    cGR_n_8 =np.random.rand(n,i)

    #distributed variables metric
    ODd = norm.ppf(OD_n_1, loc=OD*meanOD, scale=OD*sdOD)
    WTd = norm.ppf(WT_n_2, loc=WT*meanWT, scale=WT*sdWT)

    #Crack width in mm
    cL_run = np.maximum(1, norm.ppf(cL_n_5, loc=cL_measured * 1.0, scale=tool_L))
    cL_GR = 3.126402*np.power( -np.log(1-cGR_n_7) ,1/1.874228039)	#PMC from width distribution
    cL = cL_run + cL_GR*(time_delta)

    #Crack detpth in mm
    cD_run = np.maximum(0, norm.ppf(cD_n_6, loc=cPDP*WTm*1.0, scale=tool_D))
    cD_GR = 0.26*np.power( -np.log(1-cGR_n_8) ,1/2.0)	#Standard
    cD = cD_run + cD_GR*(time_delta)

    depth_thresh = 0.6
    depth_fails = cD >= depth_thresh*WTd
    depth_fails = depth_fails.astype(int)
    
    circ_thresh = 0.4
    circ_thresh_depth = 0.4
    circ_fails = (cL >= circ_thresh*pivar*ODd) & (cD >= circ_thresh_depth*WTd)
    circ_fails = circ_fails.astype(int)

    # fails = np.where((depth_fails==1) & (circ_fails==1), 1, 0)
    fails = np.maximum(depth_fails,circ_fails)
    fails = fails.astype(int)

    fail_count = np.sum(fails, axis=0)
    depth_count = np.sum(depth_fails, axis=0)
    circ_count = np.sum(circ_fails, axis=0)

    return_dict = {'depth_%':cPDP,
                    'width_mm':cL_measured,
                    'WT_mm':WTm,
                    'OD_mm':OD,
                   'fail_count':fail_count,
                   'iterations':np.size(fails, axis=0),
                   'depth_count':depth_count,
                   'circ_count':circ_count}

    return_df = pd.DataFrame(return_dict)

    return return_df

def csccpoe_sing(n):
    pivar = np.pi

#Outside Diameter inches
    OD = 10

#Wall thickness in mm
    WTm = 4

#Crack Depth fraction of WT
    cPDP = 0.1

#Crack Length in mm
    cL_measured = 100

#Year of Inspection
    Insp = datetime.date(2016,2,10)
    time_delta = ((now-Insp).days)/365.25
    vendor = "PII"

    #outside diameter stats
    OD = OD*25.4
    meanOD = 1.0    #fraction of OD
    sdOD = 0.0006   #fraction of OD

    #Wall thickness stats
    WT = WTm
    meanWT = 1.01   #fraction of WT
    sdWT = 0.01     #fraction of WT

    #Grade in ksi
#    S = (Sm*1000)/6.89476
    meanS = 1.10    #fraction of S
    sdS = 0.035     #fraction of S

    #Young's modulus in psi
#    E = 30000000 
    meanE = 1.0     #fraction of E
    sdE = 0.04      #fraction of E

    if WTm < 10:
        sdDepthTool = 0.117 #fraction of WT
    else:
        sdDepthTool = 0.156 #fraction of WT

    #tool tolerances
    defectTol = {
        "Rosen": {
            "sdPDP":0.78,   # in mm
            "sdL":7.80      #in mm
            },
        "RosenF": {
            "sdPDP":sdDepthTool*WTm, #in mm
            "sdL":7.80  #in mm
            },
        "PII": {
            "sdPDP": 0.31,  # in mm
            "sdL":6.10      #in mm
            }
        }

    tool_D = defectTol[vendor]["sdPDP"]
    tool_L = defectTol[vendor]["sdL"]
    
    np.random.seed()
    
    OD_n_1 = np.random.rand(1,n)
    WT_n_2 = np.random.rand(1,n)
    cL_n_5 = np.random.rand(1,n)
    cD_n_6 = np.random.rand(1,n)
    cGR_n_7 =np.random.rand(1,n)    
    cGR_n_8 =np.random.rand(1,n)

    #distributed variables metric
    ODd = norm.ppf(OD_n_1, loc=OD*meanOD, scale=OD*sdOD)
    WTd = norm.ppf(WT_n_2, loc=WT*meanWT, scale=WT*sdWT)

    #Crack length in mm
    cL_run = np.maximum(1, norm.ppf(cL_n_5, loc=cL_measured * 1.0, scale=tool_L))
    cL_GR = 3.126402*np.power( -np.log(1-cGR_n_7) ,1/1.874228039)	#PMC from width distribution
    cL = cL_run + cL_GR*(time_delta)

    #Crack detpth in mm
    cD_run = np.maximum(0, norm.ppf(cD_n_6, loc=cPDP*WTm*1.0, scale=tool_D))
    cD_GR = 0.26*np.power( -np.log(1-cGR_n_8) ,1/2.0)	#Standard
    cD = cD_run + cD_GR*(time_delta)

    depth_thresh = 0.6
    depth_fails = cD >= depth_thresh*WTd
    depth_fails = depth_fails.astype(int)

    circ_thresh = 0.4
    circ_thresh_depth = 0.4
    circ_fails = (cL >= circ_thresh*pivar*ODd) & (cD >= circ_thresh_depth*WTd)
    circ_fails = circ_fails.astype(int)

    # fails = np.where((depth_fails==1) & (circ_fails==1), 1, 0)
    fails = np.maximum(depth_fails, circ_fails)
    fails = fails.astype(int)

    fail_count = np.sum(fails)
    depth_count = np.sum(depth_fails)
    circ_count = np.sum(circ_fails)

    return_dict = {'depth_%':[cPDP],
                    'length_mm':[cL_measured],
                    'WT_mm':[WTm],
                    'OD_mm':[OD],
                   'fail_count':[fail_count],
                   'iterations':[np.size(fails)],
                   'depth_count':[depth_count],
                   'circ_count':[circ_count]}

    return_df = pd.DataFrame(return_dict)

    return_qc_dict = {'start_depth_mm':cD_run,
                        'depth_growth_rate_mm/yr':cD_GR,
                        'final_depth_mm':cD,
                        'depth_thresh_mm':depth_thresh*WTd,
                        'depth_fail':depth_fails,
                        'start_length_mm':cL_run,
                        'length_growth_rate_mm/yr':cL_GR,
                        'final_length_mm':cL,
                        'circ_thresh_mm':circ_thresh*pivar*ODd,
                        'circ_thresh_depth_mm':circ_thresh_depth*WTd,
                        'circ_fails':circ_fails,
                        'final_fail':fails
                        }

##    return_qc_dict = {'random_wall_thickness':WTd,
##                        }

    return_qc_dict = {key: value.reshape(n) for key, value in return_qc_dict.items()}
    return_qc = pd.DataFrame(return_qc_dict)

    return return_df, return_qc


if __name__ == '__main__':

    iter = 100000
##    result = csccpoe(df,iter)
    
    result ,qc = csccpoe_sing(iter)
    
    result['POE'] = result['fail_count']/result['iterations']
    result['POE_d'] = result['depth_count']/result['iterations']
    result['POE_c'] = result['circ_count']/result['iterations']
    print(result)
    
    end = time.time()
    print("C-SCC POE Simulation\n")

    result.to_csv('cscc-output.csv')
##    qc.to_csv('QC_cscc-output.csv')
    
    print("Calculation took {} seconds.".format(end-start))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




