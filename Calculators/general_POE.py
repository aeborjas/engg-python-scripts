from scipy.stats import norm, gamma
import time, datetime
import numpy as np, pandas as pd
from os.path import abspath, dirname, join

import simpoe.cracks.failureStress, simpoe.cracks.crackLimitStates
import simpoe.corrosion.failureStress, simpoe.corrosion.corrLimitStates
import simpoe.dents.failureCycles, simpoe.dents.dentLimitStates
from simpoe import unpacker, model_constants, distributer, cgr

import importlib.util
spec = importlib.util.spec_from_file_location("useful_func", r"C:\Users\armando_borjas\Documents\Python\useful_func.py")
useful_func = importlib.util.module_from_spec(spec)
spec.loader.exec_module(useful_func)

pd.set_option('display.max_columns',500)

class MonteCarlo:
    def __init__(self, model, run_date=None):
        self.model_type = model
        self.set_model()
        self.set_config(run_date)
        return None

    def set_config(self,run_date=None):
        if run_date == None:
            self.now = datetime.date.today()
        else:
            self.now = datetime.datetime.strptime(run_date, '%Y-%m-%d').date()
        pass

    def get_data(self, filepath):
        CSV_FILE_NAME = abspath(join(dirname(__file__), filepath))
        self.df = pd.read_csv(CSV_FILE_NAME, header=0)
        self.process_dates()
        return None
    
    def process_dates(self):
        self.df['ILIRStartDate'] = pd.to_datetime(self.df['ILIRStartDate']).dt.date
        self.df['install_date'] = pd.to_datetime(self.df['install_date']).dt.date

    # Need to refine this section so that it can create a template spreadsheet to input data into
    def build_template(self):
        
        cols = ['line',
                'FeatureID',
                'vendor',
                'tool',
                'ILIRStartDate',
                'status',
                'type',
                'ILIFSurfaceInd',
                'chainage_m',
                'depth_fraction',
                'length_mm',
                'width_mm',
                'vendor_cgr_mmpyr',
                'vendor_cgr_sd',
                'OD_inch',
                'WT_mm',
                'grade_MPa',
                'toughness_J',
                'install_date',
                'coating_type',
                'incubation_yrs',
                'MAOP_kPa']

        data = [[None]] * len(cols)

        temp_dict = {x:y for x,y in zip(cols,data)}
        temp_df = pd.DataFrame(temp_dict)
        temp_df.to_csv(abspath(join(dirname(__file__), 'inputs_POE_template.csv')), index=False)
        return None

    def build_df(self, od_i, wt_mm, grade_mpa, maop_kpa, installdate, ILIdate, pdf, lengthmm, create=True, **kwargs):
            
        if create:
            temp_dict = dict(OD_inch=[od_i],
                            WT_mm=[wt_mm],
                            grade_MPa=[grade_mpa],
                            install_date=[installdate],
                            MAOP_kPa=[maop_kpa],
                            ILIRStartDate=[ILIdate],
                            depth_fraction=[pdf],
                            length_mm=[lengthmm]
                            )
                
            return pd.DataFrame(temp_dict)
        else:
            temp_df = pd.DataFrame(dict(OD_inch=[od_i],
                            WT_mm=[wt_mm],
                            grade_MPa=[grade_mpa],
                            install_date=[installdate],
                            MAOP_kPa=[maop_kpa],
                            ILIRStartDate=[ILIdate],
                            depth_fraction=[pdf],
                            length_mm=[lengthmm]
                            ))
            return kwargs['df'].append(temp_df)

    def set_iterations(self, iterations):
        self.iterations = iterations
        return None

    def index_marks(self, nrows, chunk_size):
        return range(chunk_size, np.ceil(nrows / chunk_size).astype(int) * chunk_size, chunk_size)

    def split_df(self, chunk_size):
        indices = self.index_marks(self.df.shape[0], chunk_size)
        return np.split(self.df, indices)

    def set_model(self):
        model_dict = {'CORR':self.corrpoe,
                        'SCC':self.sccpoe,
                        'MD':self.mdpoe,
                        'RD':self.rdpoe}
        self.model = model_dict[self.model_type]
        return None 

    def run(self, split_calculation=False, buffer_size=1000):
        t1 = time.time()

        if hasattr(self,'result'):
            del self.result

        if not(split_calculation):
            self.result, self.qc = self.model(self.df,self.iterations)
        else:
            self.result = [self.model(x,self.iterations)[0] for x in self.split_df(buffer_size)]
            self.result = pd.concat(self.result)
            self.qc = None

        self.result["POE"] = self.result["fail_count"] / self.result["iterations"]
        self.result["POE_l"] = self.result["leak_count"] / self.result["iterations"]
        self.result["POE_r"] = self.result["rupture_count"] / self.result["iterations"]

        self.result["1-POE"] = 1 - self.result["POE"]
        agg_POE = 1 - np.prod(self.result["1-POE"])       

        print(f"{self.model_type} POE Simulation")
        print("Aggregated POE for these features is {}.\n".format(agg_POE))
        print(f"Calculation took {time.time()-t1:.4f} seconds.")
        return None

    def merge_result(self, key):
        return self.result.merge(self.df, on=key)

    def corrpoe(self, df, n):
        # Number of features is equal to number of rows in csv file
        i = df.shape[0]

        OD = df['OD_inch'].values
        WTm = df['WT_mm'].values
        Sm = df['grade_MPa'].values
        df['toughness_J'] = df['toughness_J'].fillna(10)
        Tm = df['toughness_J'].values
        Inst = df['install_date']

        OPm = df['MAOP_kPa'].values

        fPDP = df['depth_fraction'].values
        fL_measured = df['length_mm'].values
        fW_measured = df['width_mm'].values
        fchainage = df['chainage_m'].values
        fstatus = df['status'].values
        ftype = df['type'].values
        fids = df['FeatureID'].values

        vendor = df['vendor'].values
        tool = df['tool'].values
        Insp = df['ILIRStartDate']

        time_delta = ((self.now - Insp).dt.days.values) / 365.25

        # Growth Rate mechanism
        mechanism = "weibull"

        # CGA
        CGR_CGA_MPY = 10  # MPY (mili inches per year)

        # weibull Distribution
        shape = 1.439
        scale = 0.1

        def model_error(p):
            return 0.914 + gamma.ppf(p, 2.175, scale=0.225)

        # unit conversion to US units
        WT = WTm / 25.4
        S = (Sm * 1000) / 6.89476
        OP = OPm / 6.89476

        meanWT = 1.01
        sdWT = 0.01
        meanS = 1.09
        sdS = 0.044

        # ----------Maybe these tool tolerances can be a property of the range class? or perhaps a dictionary that I can store somewhere collecting information on numerous tool types.
        # tool tolerances
        defectTol = {
            "Rosen": {
                "sdPDP": 0.078,  # in fraction WT
                "sdL": 0.61  # in inches
            }
        }

        tool_D = np.where(vendor == "Rosen", defectTol["Rosen"]["sdPDP"], defectTol["Rosen"]["sdPDP"])
        tool_L = np.where(vendor == "Rosen", defectTol["Rosen"]["sdL"], defectTol["Rosen"]["sdL"])

        # ----------------Need a tiler function that is dynamic (e.g., take in multiply inputs, and spit out multiple outputs)
        # non-distributed variables

        OD = np.tile(OD, (n, 1)) 
        OP = np.tile(OP, (n, 1))

        # ------------------Need a randomized function that is dynamic (see comment above)
        np.random.seed()

        mE_n_1, WT_n_2, S_n_3, Y_n_4, fL_n_5, fD_n_6, fGR_n_7 = distributer.random_prob_gen(7, iterations=n, features=i)

        # distributed variables
        ##    ODd = norm.ppf(OD_n_1, loc=OD*meanOD, scale=OD*sdOD)
        WTd = norm.ppf(WT_n_2, loc=WT * meanWT, scale=WT * sdWT)
        Sdist = norm.ppf(S_n_3, loc=S * meanS, scale=S * sdS)

        # feature length in inches
        fL = np.maximum(0, norm.ppf(fL_n_5, loc=fL_measured * 1.0 * (1 / 25.4), scale=tool_L))

        # feature depth in inches
        fD_run = np.maximum(0, norm.ppf(fD_n_6, loc=fPDP * WT * 1.0, scale=tool_D * WT))

        if mechanism == "CGA":
            fD_GR = cgr.cgr_mpy(CGR_CGA_MPY)
        elif mechanism == "weibull":
            fD_GR = cgr.cgr_weibull(fGR_n_7, shape, scale) / 25.4
        elif mechanism == "logic":
            if time_delta >= 20:
                fD_GR = cgr.half_life(fD_run, Inst, Insp)
            else:
                fD_GR = cgr.pct_wt(WTd)
        elif mechanism == "half-life":
            fD_GR = cgr.half_life(fD_run, Inst, Insp)  # fD_run is in inches
        elif mechanism == "2.2%WT":
            fD_GR = cgr.pct_wt(WTd)
        else:
            raise Exception("Please select a valid mechanism: half-life | 2.2%WT | CGA | logic | weibull")

        fD = fD_run + fD_GR * time_delta

        modelError = model_error(mE_n_1)

        # Failure Stress in psi
        failure_stress = simpoe.corrosion.failureStress.modified_b31g(OD, WTd, Sdist, fL, fD, units="US") * modelError

        # Failure pressure in psi
        failPress = 2 * failure_stress * WTd / OD

        ruptures, rupture_count = simpoe.corrosion.corrLimitStates.ls_corr_rupture(failPress, OP, bulk=True)

        leaks, leak_count = simpoe.corrosion.corrLimitStates.ls_corr_leak(WTd, fD, bulk=True)

        fails, fail_count = simpoe.corrosion.corrLimitStates.ls_corr_tot(ruptures, leaks, bulk=True)

        # Perhaps can turn the next section into a triggerable function.
        # Function would take in inputs that the user desires to create an export of, and fire off a csv
        # for a given feature
        #
        # otpt_ftr = 2                 #has to be less than i-1
        # output = pd.DataFrame({"OD":np.take(OD,otpt_ftr, axis=1),
        #                        "WTd":np.take(WTd,otpt_ftr, axis=1),
        #                        "Sdist":np.take(Sdist,otpt_ftr, axis=1),
        #                        "fL":np.take(fL,otpt_ftr, axis=1),
        #                        "fD_run":np.take(fD_run,otpt_ftr, axis=1),
        #                        "fD":np.take(fD,otpt_ftr, axis=1),
        #                        "fD_GR":np.take(fD_GR,otpt_ftr, axis=1),
        ##                           "l2Dt":np.take(l2Dt,otpt_ftr, axis=1),
        ##                           "Mt":np.take(Mt,otpt_ftr, axis=1),
        ##                           "flowS":np.take(flowS,otpt_ftr, axis=1),
        ##                           "modelError":np.take(modelError,otpt_ftr, axis=1),
        ##                           "failPress":np.take(failPress,otpt_ftr, axis=1),
        ##                           "leaks":np.take(leaks,otpt_ftr, axis=1),
        ##                           "ruptures":np.take(ruptures,otpt_ftr, axis=1),
        ##                           "fails":np.take(fails,otpt_ftr, axis=1)
        ##                           })
        ##
        ##    print(output.head())

        ##    return_list=[fail_count, np.size(fails, axis=0), rupture_count, leak_count, 0]  #No NaN

        return_dict = {"FeatureID":fids,
                    "fail_count": fail_count,
                    "iterations": np.size(fails, axis=0),
                    "rupture_count": rupture_count,
                    "leak_count": leak_count,
                    "nan": np.zeros((i,)),
                    "PDP_frac": fPDP,
                    "flength": fL_measured}

        return_df = pd.DataFrame(return_dict)

        return return_df, None

    def mdpoe(self, df,n):

        #Number of features is equal to number of rows in csv file
        i = df.shape[0]

        OD = df['OD_inch'].values
        WTm = df['WT_mm'].values
        Sm = df['grade_MPa'].values
        df['toughness_J'] = df['toughness_J'].fillna(10)
        Tm = df['toughness_J'].values
        Inst = df['install_date']

        OPm = df['MAOP_kPa'].values

        cPDP = df['depth_fraction'].values
        cL_measured = df['length_mm'].values
        cW_measured = df['width_mm'].values
        cchainage = df['chainage_m'].values
        cstatus = df['status'].values
        ctype = df['type'].values
        cids = df['FeatureID'].values

        vendor = df['vendor'].values
        tool = df['tool'].values
        Insp = df['ILIRStartDate']

        time_delta = ((self.now-Insp).dt.days.values)/365.25
        ILI_age = ((Insp-Inst).dt.days.values)/365.25

        #Growth Rate mechanism
        shape, scale = 2.55, 0.10

        pivar = np.pi

        #Young's modulus in psi
        E = model_constants.const['E']

        #Crack fracture area in sq. inches
        fa = model_constants.const['fa']

        meanOD = 1.0
        sdOD = 0.0006
        meanWT = 1.01
        sdWT = 0.01
        meanS = 1.1
        sdS = 0.035
        meanE = 1.0
        sdE = 0.04

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
        
        OP = np.tile(OP, (n, 1))
        T = np.tile(T, (n, 1))

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
        # cL_GR = cL_run/ILI_age
        cL = cL_run +  0*time_delta

        #Crack detpth in inches
        cD_run = np.maximum(0, norm.ppf(cD_n_6, loc=cPDP*WTm*1.0, scale=tool_D))/25.4
        cD_GR = cD_run/ILI_age
        cD = cD_run +  cD_GR*time_delta

        failStress = simpoe.cracks.failureStress.modified_lnsec(ODd, WTd, Sdist, T, Ydist, cL, cD, units="US")

        #Failure pressure in psi
        failPress = 2*failStress*WTd/ODd

        fsNaNs = np.extract(np.isnan(failStress),failStress)
        fsNaNs_count = fsNaNs.size

        ruptures, rupture_count = simpoe.cracks.crackLimitStates.ls_md_rupture(failPress, OP, bulk=True)

        leaks, leak_count = simpoe.cracks.crackLimitStates.ls_md_leak(WTd, cD, thresh=1.0, bulk=True)

        fails, fail_count = simpoe.cracks.crackLimitStates.ls_crack_tot(ruptures, leaks, bulk=True)

        return_dict = {"FeatureID":cids,
                    "fail_count":fail_count,
                        "iterations":np.size(fails, axis=0),
                        "rupture_count":rupture_count,
                        "leak_count":leak_count,
                        "nan": fsNaNs_count,#np.zeros((i,)),
                        "PDP_frac":cPDP,
                        "clength":cL_measured}

        return_df = pd.DataFrame(return_dict)

        return return_df, None

    def sccpoe(self, df,n):

        #Number of features is equal to number of rows in csv file
        i = df.shape[0]

        OD = df['OD_inch'].values
        WTm = df['WT_mm'].values
        Sm = df['grade_MPa'].values
        df['toughness_J'] = df['toughness_J'].fillna(10)
        Tm = df['toughness_J'].values
        Inst = df['install_date']

        OPm = df['MAOP_kPa'].values

        cPDP = df['depth_fraction'].values
        cL_measured = df['length_mm'].values
        cW_measured = df['width_mm'].values
        cchainage = df['chainage_m'].values
        cstatus = df['status'].values
        ctype = df['type'].values
        cids = df['FeatureID'].values

        vendor = df['vendor'].values
        tool = df['tool'].values
        Insp = df['ILIRStartDate']

        time_delta = ((self.now-Insp).dt.days.values)/365.25

        #Growth Rate mechanism
        # shape, scale = 2.0, 0.26
        shape, scale = 2.55, 0.1

        pivar = np.pi

        #Young's modulus in psi
        E = model_constants.const['E']

        #Crack fracture area in sq. inches
        fa = model_constants.const['fa']

        meanOD = 1.0
        sdOD = 0.0006
        meanWT = 1.01
        sdWT = 0.01
        meanS = 1.10
        sdS = 0.035
        meanE = 1.0
        sdE = 0.04

        #Unit conversion to US units
        WT = WTm/25.4
        S = (Sm*1000)/6.89476
        T = Tm*0.7375621
        OP = OPm/6.894069706666676

        sdDepthTool = np.where(WTm < 10, 0.117, 0.156)  #fraction of WT

        #tool tolerances
        defectTol = {
            "RosenF": {
                "sdPDP": 0.78,   # in mm (0.78)
                "sdL": 7.80      #in mm (7.80)
                },
            "Rosen": {
                "sdPDP":sdDepthTool*WTm, #in mm
                "sdL":7.80  #in mm
                },
            "PII": {
                "sdPDP": 0.31,  # in mm
                "sdL":6.10      #in mm
                }
            }

        tool_D = np.where(vendor=="Rosen",defectTol["Rosen"]["sdPDP"],defectTol["PII"]["sdPDP"])
        tool_L = np.where(vendor=="Rosen",defectTol["Rosen"]["sdL"],defectTol["PII"]["sdL"])


        ####Artificially added------------------------------
        qc_list = [OD, WTm, Sm, Tm, OPm, Inst.values,
                    cPDP, cL_measured, cW_measured, cstatus, ctype, cchainage]
        qc_cols = ['OD_inch','WT_mm','grade_MPa','toughness_J','MAOP_kPa','install_date',
                'depth_fraction','length_mm','width_mm','status','type','chainage_m']
        qc_dict = dict()
        qc_dict = {x:y for x,y in zip(qc_cols,qc_list)}
        
        qc_df = pd.DataFrame(qc_dict)
        qc_df['E']=E
        qc_df['fa']=fa
        qc_df['Insp']=Insp
        qc_df['vendor']=vendor
        qc_df['tool']=tool
        qc_df['tool_D']=tool_D
        qc_df['tool_L']=tool_L
        ####-----------------------------------------------
        
        #non-distributed variables
        OP = np.tile(OP, (n, 1))
        T = np.tile(T, (n, 1))

        #distributed variables
        np.random.seed()

        OD_n_1, WT_n_2, S_n_3, Y_n_4, cL_n_5, cD_n_6, cGR_n_7 = distributer.random_prob_gen(7, iterations=n, features=i)

        #distributed variables imperial
        ODd = norm.ppf(OD_n_1, loc=OD*meanOD, scale=OD*sdOD)
        WTd = norm.ppf(WT_n_2, loc=WT*meanWT, scale=WT*sdWT)
        Sdist = norm.ppf(S_n_3, loc=S*meanS, scale=S*sdS)
        Ydist = norm.ppf(Y_n_4, loc=E*meanE, scale=E*sdE)

        #crack length in inches
        cL = np.maximum(1, norm.ppf(cL_n_5, loc=cL_measured * 1.0, scale=tool_L)) / 25.4
        #cL = cL + time_delta*22.5/25.4 #remember to remove this later

        #Crack detpth in inches
        cD_run = np.maximum(0, norm.ppf(cD_n_6, loc=cPDP*WTm*1.0, scale=tool_D))/25.4
        cD_GR = cgr.cgr_weibull(cGR_n_7, shape, scale) / 25.4
        #cD_GR = 0.30 / 25.4

        cD = cD_run + cD_GR * time_delta

        failStress = simpoe.cracks.failureStress.modified_lnsec(ODd, WTd, Sdist, T, Ydist, cL, cD, units="US")

        #Failure pressure in psi
        failPress = 2*failStress*WTd/ODd

        fsNaNs = np.extract(np.isnan(failStress),failStress)
        fsNaNs_count = fsNaNs.size

        ruptures, rupture_count = simpoe.cracks.crackLimitStates.ls_scc_rupture(failPress, OP, bulk=True)

        leaks, leak_count = simpoe.cracks.crackLimitStates.ls_scc_leak(WTd, cD, bulk=True)

        fails, fail_count = simpoe.cracks.crackLimitStates.ls_crack_tot(ruptures, leaks, bulk=True)

        return_dict = {"FeatureID":cids,
                    "fail_count":fail_count,
                        "iterations":np.size(fails, axis=0),
                        "rupture_count":rupture_count,
                        "leak_count":leak_count,
                        "nan": fsNaNs_count,#np.zeros((i,)),
                        "PDP_frac":cPDP,
                        "clength":cL_measured}

        return_df = pd.DataFrame(return_dict)

        return return_df, qc_df

    # RD WIP
    def rdpoe(self, df, n):
        # Number of features is equal to number of rows in csv file
        i = df.shape[0]

        OD = df['OD_inch'].values
        WTm = df['WT_mm'].values
        Sm = df['grade_MPa'].values
        df['toughness_J'] = df['toughness_J'].fillna(10)
        Tm = df['toughness_J'].values
        Inst = df['install_date']

        OPm = df['MAOP_kPa'].values

        # outside diameter in mm
        ODm = OD * 25.4

        ###SOMETHING IS WRONG HERE###--------------------------------------
        UTSm = model_constants.ultimate_determiner(Sm)

        # Operating pressure in kPa
        MAXPm = 7375.85919407132
        MINPm = 0

        MAXStr = (MAXPm * ODm) / (2000 * WTm)
        MINStr = (MINPm * ODm) / (2000 * WTm)

        cycles = 24

        dPDP = df['depth_fraction'].values
        dL_measured = df['length_mm'].values
        dW_measured = df['width_mm'].values
        dchainage = df['chainage_m'].values
        dstatus = df['status'].values
        dtype = df['type'].values
        dids = df['FeatureID'].values

        # gouge depth pct
        gPDP = 0

        # Inline inspection range properties
        vendor = df['vendor'].values
        tool = df['tool'].values
        Insp = df['ILIRStartDate']

        time_delta = (self.now - Inst).dt.days.values / 365.25
        
        # Sensitivity Factor
        sf = 1

        meanOD = 1.0
        sdOD = 0.0006
        meanWT = 1.01
        sdWT = 0.01
        meanUTS = 1.12
        sdUTS = 0.035

        NPS = np.floor(OD)
        rosensd = np.where(NPS < 18, 0.005, np.where(NPS < 30, 0.003, np.where(NPS < 40, 0.002, 0.0015)))

        # tool tolerances
        defectTol = {
            "Rosen": {
                "sdPDP": rosensd,  # in fraction
                "sdL": 19.5,  # in mm
                "sdW": 39.0  # in mm
            },
            "BJ": {
                "sdPDP": 2.5,  # in mm
                "sdL": 2.5,  # in mm
                "sdW": 2.5  # in mm
            },
            "TDW": {
                "sdPDP": 0.0056,  # in fraction
                "sdL": 7.6,  # in mm
                "sdW": 25.4  # in mm
            }
        }

        tool_D = np.where(vendor == "BJ", defectTol["BJ"]["sdPDP"],
                        np.where(vendor == "Rosen", defectTol["Rosen"]["sdPDP"] * (NPS * 25.4),
                                defectTol["TDW"]["sdPDP"] * (NPS * 25.4)))
        tool_L = np.where(vendor == "BJ", defectTol["BJ"]["sdL"],
                        np.where(vendor == "Rosen", defectTol["Rosen"]["sdL"], defectTol["TDW"]["sdL"]))
        tool_W = np.where(vendor == "BJ", defectTol["BJ"]["sdW"],
                        np.where(vendor == "Rosen", defectTol["Rosen"]["sdW"], defectTol["TDW"]["sdW"]))

        # gouge depth specs in fraction of WT
        sdg = 0.078

        # distributed variables
        np.random.seed()

        OD_n_1, WT_n_2, UTS_n_3, g_n_4, dW_n_5, dL_n_6, dD_n_7 = distributer.random_prob_gen(7, iterations=n, features=i)

        # distributed variables
        ODd = norm.ppf(OD_n_1, loc=ODm * meanOD, scale=ODm * sdOD)
        WTd = norm.ppf(WT_n_2, loc=WTm * meanWT, scale=WTm * sdWT)
        UTSd = norm.ppf(UTS_n_3, loc=UTSm * meanUTS, scale=UTSm * sdUTS)

        # gouge depth in mm
        gD = np.where(gPDP == 0, np.zeros((n,i), dtype=float),np.maximum(0, norm.ppf(g_n_4, loc=gPDP * WTm * 1.0, scale=WTm * sdg)))

        # dent width in mm
        dW_run = np.maximum(0, norm.ppf(dW_n_5, loc=dW_measured * 1.0, scale=tool_W))

        # dent length in mm
        dL_run = np.maximum(0, norm.ppf(dL_n_6, loc=dL_measured * 1.0, scale=tool_L))

        # dent depth in mm
        dD_run = np.maximum(0.01, norm.ppf(dD_n_7, loc=dPDP * OD * 25.4 * 1.0, scale=tool_D))

        NF = simpoe.dents.failureCycles.nf_EPRG(ODd, WTd, UTSd, dL_run, dW_run, dD_run, gD, MAXStr, MINStr)

        n_cycles = time_delta * cycles

        fails, fail_count = simpoe.dents.dentLimitStates.ls_dent_fail(NF, n_cycles, bulk=True)

        return_dict = {"FeatureID":dids,
                    "fail_count": fail_count,
                    "iterations": np.size(fails, axis=0),
                    "NPS_Frac": dPDP,
                    "dlength": dL_measured,
                    "dwidth": dW_measured}

        return_df = pd.DataFrame(return_dict)

        ##UPDATE HERE-------------------------------------------------------------------------------
        feat = 0
        cols = [ODd,WTd,UTSd,gD,dW_run,dL_run,dD_run,NF,fails]
        cols_lab = ['ODd','WTd','UTSd','gD','dW_run','dL_run','dD_run','NF','fails']

        qc_dict = dict()
        qc_dict = {x:np.take(y,feat,axis=1) for x,y in zip(cols_lab,cols)}
        qc_df = pd.DataFrame(qc_dict)
        qc_df['n_cycles']=n_cycles[0]
        qc_df['Inst']=Inst[0]
        qc_df['vendor']=vendor[0]
        qc_df['tool']=tool[0]
        qc_df['tool_D']=tool_D[0]
        qc_df['tool_L']=tool_L[0]
        qc_df['tool_W']=tool_W[0]
        ##-----------------------------------------------------------------------------------------

        
        return return_df, qc_df


if __name__ == '__main__':
    scc = MonteCarlo('SCC')
    scc.get_data('sample_of_inputs.csv')
    scc.set_iterations(1_000_0)
    scc.run(split_calculation=True, buffer_size=1500)
    # for i,x in enumerate(scc.df.columns):
    #     vars()[x.strip()] = scc.df.to_numpy()[:,i]