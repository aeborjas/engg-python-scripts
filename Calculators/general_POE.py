from scipy.stats import norm, gamma
import time, datetime, re
import numpy as np, pandas as pd
from os.path import abspath, dirname, join

# import importlib.util
# spec = importlib.util.spec_from_file_location("useful_func", r"C:\Users\armando_borjas\Documents\Python\useful_func.py")
# useful_func = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(useful_func)

def ultimate_determiner(grade):
    """
    Takes grade in MPa as input, and returns Ultimate tensile strength in MPa
    :param grade:
    :return:
    """
    lookupUTS = np.array([(206.84, 406.79),
                 (241.32, 430.92),
                 (289.58, 499.87),
                 (317.16, 499.87),
                 (358.53, 524.00),
                 (386.11, 541.93),
                 (413.69, 568.82),
                 (448.16, 579.16),
                 (482.63, 599.84),
                 (551.58, 639.14),
                 (9999.00, 679.13)
                 ])

    index_match = np.searchsorted(lookupUTS[:,0], grade)
    return lookupUTS[index_match,1]

#CGA
def cgr_mpy(mpy=10.0):
    """
    :param mpy: imperfection growth rate in mili-inches per year, MPY. Default of 10 MPY.
    :return:    imperfection growth rate in mm/yr
    """
    return (mpy/1000.0)*25.4

#half-life
def half_life(d,start,end):
    '''

    :param d:       feature property (in any units), at the time of measuring
    :param start:   start year, datetime.date dataframe object, or integer year.
    :param end:     end year, datetime.date dataframe object, or integer year.
    :return:        returns the half-life growth rate based on the input feature units
    '''
    try:
        return (2*d)/((end-start).dt.days.values/365.25)
    except AttributeError:
        return (2*d)/(end-start)

#2.2%WT
def pct_wt(wt):
    """

    :param wt:  returns the imperfection growth rate as 2.2% of the wall thickness, *
    :return:
    """
    return 0.0220*wt

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

def modified_lnsec(od, wt, s, t, e, cL, cD, units="SI"):
    """
    Calculates the failure stress using the Modified ln-sec Equation
    :param od:      Pipe outside diameter, in mm (SI), or inches (US)
    :param wt:      Pipe wall thickness, in mm (SI), or inches (US)
    :param s:       Pipe grade, in kPa (SI), or psi (US)
    :param t:       Pipe toughness, in J (SI), or ft.lbs (US)
    :param e:       Pipe Young's Modulus, in kPa (SI), or psi (US)
    :param cL:      crack length, in mm (SI), or inches (US)
    :param cD:      crack depth, in mm (SI), or inches (US)
    :param units:   flag for which units to use, "SI" or "US", default "SI"
    :return:        Failure stress, in kPa (SI), or psi (US)
    """

    if units=="SI":
        od = od/25.4
        wt = wt/25.4
        s = s/6.89476
        t = t*0.737562
        e = e/6.89476
        cL = cL/25.4
        cD = cD/25.4

    flowS = s + 10000.0

    ellipticalC = np.minimum(np.pi*cL/4.0,np.power(67.2*od*wt*0.5,0.5))

    CRTRatio = np.minimum(93.753,np.power(ellipticalC,2)/(od*0.5*wt))

    Mt = np.maximum(np.power(1.0+1.255*CRTRatio-0.0135*np.power(CRTRatio,2.0),0.5), (cD/wt)+0.001)
    Mp = (1.0-cD/(wt*Mt))/(1-(cD/wt))

    x = (12.0*e*np.pi*(t/0.124))/(8.0*ellipticalC*np.power(flowS,2.0))
    y = x*np.power(1.0-np.power(cD/wt,0.8),-1.0)

    failStress = (flowS/Mp)*np.arccos(np.exp(-x))/np.arccos(np.exp(-y))

    if units=="SI":
        failStress = failStress*6.89476

    return failStress

def nf_EPRG(od, wt, uts, dL, dW, dD, gD, MAX, MIN, sf=1.0, model='EPRG-2000', units="SI"):
    """
    calculates the number of cycles until failure given equivalent stress cycles
    :param od:      Pipe diameter in mm (SI), or in inches (US)
    :param wt:      Pipe wall thickness in mm (SI), or in inches (US)
    :param uts:     Pipe ultimate tensile strength in MPa (SI), or in ksi (US)
    :param dL:      Feature length in mm (SI), or in inches (US)
    :param dW:      Feature width in mm (SI), or in inches (US)
    :param dD:      Feature depth in mm (SI), or in inches (US)
    :param gD:      Gouge depth in mm (SI), or in inches (US)
    :param MAX:     Maximum stress in equivalent cycle in MPa (SI), or in ksi (US)
    :param MIN:     Minimum stress in equivalent cycle in MPa (SI), or in ksi (US)
    :param sf:      Sensitivity factor, default of 1.0
    :param units:   Unit system, SI be default, or US
    :return:    Number of cycles until failure
    """

    #od, wt, uts, dL, dW, dD, gD, MAX, MIN = float(od), float(wt), float(uts), float(dL), float(dW), float(dD), float(gD), float(MAX), float(MIN)

    if units != "SI":
        od = od*25.4
        wt = wt*25.4
        uts = uts*6.89476
        dL = dL * 25.4
        dW = dW * 25.4
        dD = dD * 25.4
        gD = gD*25.4
        MAX = MAX * 6.89476
        MIN = MIN * 6.89476

    if model == 'EPRG-2000':
    # EPRG-2000 logic
        dRL = (np.power(dL, 2.0) + 4.0 * np.power(dD, 2.0)) / (8.0 * dD)
        dRW = (np.power(dW, 2.0) + 4.0 * np.power(dD, 2.0)) / (8.0 * dD)
        dR = np.minimum(dRW, dRL)

        Crd = np.where(dR < 5.0*wt, 1.0, 2.0)

        dSEF = 1 + Crd*np.power(np.power(dD,1.5)*wt/od,0.5)
        gSEF = 1 + 9.0*(gD/wt)

        # this following sigma_a(lternating) has been multiplied by the 2 in the main fatigue life equation
        sigma = (MAX-MIN)/(1- np.power((MAX+MIN)/(2.0*uts),2.0) )

        sigma = (1/2)*(MAX-MIN)/(1- np.power((MAX+MIN)/(2.0*uts),2.0) )

        NF = (5622.0/sf)*np.power(uts/(2.0*sigma*gSEF*dSEF),5.26)

    elif model == 'Refined-EPRG-2000':
    # EPRG-2000 logic
        dRL = (np.power(dL, 2.0) + 4.0 * np.power(dD, 2.0)) / (8.0 * dD)
        dRW = (np.power(dW, 2.0) + 4.0 * np.power(dD, 2.0)) / (8.0 * dD)
        dR = np.minimum(dRW, dRL)

        Crd = np.where(dR < 5.0*wt, 1.0, 2.0)

        dSEF = 1 + Crd*np.power(np.power(dD,1.5)*wt/od,0.5)
        gSEF = 1 + 9.0*(gD/wt)

        # this following sigma_a(lternating) has been multiplied by the 2 in the main fatigue life equation
        sigma_a = (MAX-MIN)/2
        sigma = (-1 + np.sqrt(1+4*np.power(sigma_a/uts,2) )) / (2*sigma_a/np.power(uts,2))

        NF = (5622.0/sf)*np.power(uts/(2.0*sigma*gSEF*dSEF),5.26)

    elif model == 'EPRG-1995':
    # EPRG-1995 logic
        dSEF = 2.871*np.power(dD*wt/od,0.5)
        gSEF = 1 + 9.0*(gD/wt)

        mean_sigma = (MAX+MIN)/2.
        sigma_a = (MAX-MIN)/2

        B = (sigma_a/uts)/np.power(1- (mean_sigma/uts), 0.5)

        sigma_A_x2 = uts*(B*np.sqrt(4+np.power(B,2)) - np.power(B,2))

        NF = (1/sf)*(1000.0)*np.power((uts-50.0)/(sigma_A_x2*dSEF*gSEF),4.292)

    elif model == 'PETROBRAS':
    #PETROBRAS logic
        dLdW = dL/dW
        dLdD = dL/dD
        od_wt = od/wt

        conditions = [(dLdW > 0.9) & (dLdW < 2),
                    dLdW <= 0.9,
                    (dLdW >= 2) & (dLdD >= 3.0*np.power(od_wt,0.37)),
                    (dLdW >= 2) & (dLdD < 3.0*np.power(od_wt,0.37))]

        Achoices = [2.40, 2.11, 1.38, 6.50]
        Bchoices = [0.737, 0.853, 2.398, 0.825]

        A = np.select(conditions, Achoices, default=2.40)
        B = np.select(conditions, Bchoices, default=0.737)
        K = A + B*(dD/od)*np.power(od_wt,1.14)
        ka = 56.1*np.power(uts,-0.719)
        Se = 0.5*uts*(ka/K)
        C = uts + 345.0
        b = (1/6)*np.log10(Se/C)
        
        mean_sigma = (MAX+MIN)/2.
        sigma_a = (MAX-MIN)/2.

        sigma_A_fully_rev = sigma_a/(1 - np.power(mean_sigma/uts,2.0) )

        NF = (1/sf)*np.power(sigma_A_fully_rev/C, 1/b)

    return NF

class StatisticalPOE:
    def __init__(self,  config=None):
        self.set_config(config)
        self.set_model()
        return None

    def set_config(self,config):
        """Allows to se the monte carlo run date to something other than current datetime

        Keyword Arguments:
            run_date {string} -- the date in YYYY-MM-DD format to set (default: {None})
        """
        if 'run_date' in config:
            self.now = datetime.datetime.strptime(config['run_date'], '%Y-%m-%d').date()
        else:
            self.now = datetime.date.today()

        if 'leak_thresh' in config:
            self.leak_thresh = config['leak_thresh']
        else:
            self.leak_thresh = 0.80

        if 'rupt_thresh' in config:
            self.rupt_thresh = config['rupt_thresh']
        else:
            self.rupt_thresh = 1.0

        if 'cgr' in config:
            if config['cgr'] == 'full-life':
                self.cgr_life_flag = 1.0
            elif cgr == 'half-life':
                self.cgr_life_flag = 2.0
            else:
                self.cgr_life_flag = 1.0
        else:
            self.cgr_life_flag = 1.0

        pass

    def get_data(self, filepath):
        """searches for a CSV file in the same folder of this file and loads it to a pandas dataframe.

        Arguments:
            filepath {string} -- name of the CSV file containing the data

        Returns:
            None -- 
        """        
        CSV_FILE_NAME = abspath(join(dirname(__file__), filepath))
        self.df = pd.read_csv(CSV_FILE_NAME, header=0)
        self.process_dates()
        return None
    
    def process_dates(self):
        """converts specific fields in the input data to pandas datetime objects
        """        
        self.df['ILIRStartDate'] = pd.to_datetime(self.df['ILIRStartDate']).dt.date
        self.df['install_date'] = pd.to_datetime(self.df['install_date']).dt.date

    # Need to refine this section so that it can create a template spreadsheet to input data into
    def build_template(self):
        """exports a CSV file with column templates signaling all inputs possible to the POE calculation

        Returns:
            None -- 
        """        
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
                'MAOP_kPa',
                'PMax_kPa',
                'PMin_kPa',
                'AESC']

        data = [[None]] * len(cols)

        temp_dict = {x:y for x,y in zip(cols,data)}
        temp_df = pd.DataFrame(temp_dict)
        temp_df.to_csv(abspath(join(dirname(__file__), 'inputs_POE_template.csv')), index=False)
        return None

    def build_df(self, od_i, wt_mm, grade_mpa, maop_kpa, installdate, ILIdate, pdf, lengthmm, pmax_kpa, pmin_kpa, aesc, create=True, **kwargs):
        """Allows user to build a dataframe right in the prompt to be used as the input data. If the create flaf is set to False,
            then a df keyword can be specified, and the entry is appended to the dataframe df
        Arguments:
            od_i {float} -- outside diameter in inches
            wt_mm {float} -- wall thickness in mm
            grade_mpa {float} -- grade in MPa
            maop_kpa {float} -- maximum allowable operating pressure in kPa
            installdate {datetime.date} -- installation date
            ILIdate {datetime.date} -- ILI survey date
            pdf {float} -- peak depth fraction
            lengthmm {float} -- length in mm
            pmax_kpa {float} -- maximum pressure in stress cycle in kPa
            pmin_kpa {float} -- minimum pressure in stress cycle in kPa
            aesc {float} -- annual equivalent stress cycles

        Keyword Arguments:
            create {bool} -- setting this to True will return a new dataframe, setting it to False will append to a dataframe kwarg 'df' (default: {True})

        Returns:
            [pandas.DataFrame] -- dataframe containing the entry
        """        
        if create:
            temp_dict = dict(OD_inch=[od_i],
                            WT_mm=[wt_mm],
                            grade_MPa=[grade_mpa],
                            install_date=[installdate],
                            MAOP_kPa=[maop_kpa],
                            ILIRStartDate=[ILIdate],
                            depth_fraction=[pdf],
                            length_mm=[lengthmm],
                            PMax_kPa=[pmax_kpa],
                            PMin_kPa=[pmin_kpa],
                            AESC=[aesc]
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
                            length_mm=[lengthmm],
                            PMax_kPa=[pmax_kpa],
                            PMin_kPa=[pmin_kpa],
                            AESC=[aesc]
                            ))
            return kwargs['df'].append(temp_df)

    def set_model(self):
        """method used to specify which model to be ran

        Returns:
            None -- 
        """        
        # model_dict = {'CORR':self.statpoe,
        #                 'SCC':self.sccpoe,
        #                 'MD':self.mdpoe,
        #                 'RD':self.rdpoe,
        #                 'CSCC':self.csccpoe}
        self.model = self.statpoe
        return None 

    def run(self):
        """Method used to start the Monte Carlo run

        Keyword Arguments:
            split_calculation {bool} -- Set to true, the calculation will use the buffer_size to split the instance dataframe by that number (default: {False})
            buffer_size {int} -- value to split the instance dataframe (default: {1000})

        Returns:
            None -- 
        """        
        t1 = time.time()

        if hasattr(self,'result'):
            del self.result

        self.result, self.qc = self.model(self.df)

        self.result["1-POE"] = 1 - self.result["POE"]
        agg_POE = 1 - np.prod(self.result["1-POE"])       

        print(f"Statistical POE Simulation")
        print("Aggregated POE for these features is {}.\n".format(agg_POE))
        print(f"Calculation took {time.time()-t1:.4f} seconds.")
        return None

    def merge_result(self, key):
        return self.result.merge(self.df, on=key)

    def statpoe(self, df):

        # Number of features is equal to number of rows in csv file
        i = df.shape[0]

        # Setting the inputs to the appropriate variables
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
        CGA_mean_m = df['vendor_cgr_mmpyr'].values
        CGA_sd_m = df['vendor_cgr_sd'].values
        tool = df['tool'].values
        Insp = df['ILIRStartDate']

        time_delta = ((self.now - Insp).dt.days.values) / 365.25
        ILI_age = ((Insp - Inst).dt.days.values) / 365.25

        # unit conversion to US units
        WT = WTm / 25.4
        S = (Sm * 1000) / 6.89476
        OP = OPm / 6.89476

        # tool tolerances
        defectTol = {
            "MFL": {
                "sdPDP": 0.078,  # in fraction WT
            }
        }

        tool_D = np.where(vendor == "MFL", defectTol["MFL"]["sdPDP"], defectTol["MFL"]["sdPDP"])

        # feature length in inches
        fL = np.maximum(0, fL_measured * 1.0 * (1 / 25.4))

        # feature depth in inches
        fD_run = np.maximum(0, fPDP * WT * 1.0)

        # feature growth rate in inches per year
        fD_GR = np.where(np.isnan(CGA_mean_m), 
                        self.cgr_life_flag*fD_run/ILI_age,
                        self.cgr_life_flag*CGA_mean_m / 25.4)

        fD_mean = (fD_run + 0.0) +  fD_GR * time_delta

        # logic for computing the feature standard deviation
        fD_sd =  np.where(np.isnan(CGA_sd_m), 
                        np.sqrt(np.power( 1 + self.cgr_life_flag*(time_delta/ILI_age) ,2)*np.power(tool_D*WT,2)),
                        np.sqrt(np.power(tool_D*WT,2) + np.power(time_delta,2)*np.power(CGA_sd_m/25.4,2)))

        l2Dt = np.power(fL, 2.0)/(OD*WT)
        Mt = np.where(l2Dt <= 50.0,
                        np.sqrt( 1.0 +(0.6275*l2Dt)-(0.003375*np.power(l2Dt, 2.0))),
                        0.032*l2Dt+3.3)
        
        flowS = S + 10000.0
    
        # Failure depth in inches
        operating_stress = (OD*OP)/(2*WT)
        failure_depth = ((operating_stress - flowS )*(WT)) / ( ((operating_stress/Mt) - flowS)*0.85 )

        POE_l = 1 - norm.cdf(0.80*WT, loc=fD_mean, scale=fD_sd)
        POE_r = 1 - norm.cdf(failure_depth, loc=fD_mean, scale=fD_sd)
        
        POE = np.where(failure_depth/WT >= 0.80, POE_l, POE_r)

        return_dict = {"FeatureID":fids,
                    "time_delta":time_delta,
                    "ILI_age":ILI_age,
                    "WT":WT,
                    "S":S,
                    "OP":OP,
                    "tool_D":tool_D,
                    "fL_in":fL,
                    "fD_run_in":fD_run,
                    "fD_GR_inpyr":fD_GR,
                    "fD_mean_in":fD_mean,
                    "fD_sd_in":fD_sd,
                    "l2Dt":l2Dt,
                    "Mt":Mt,
                    "flowS":flowS,
                    "op_stress":operating_stress,
                    "failure_depth":failure_depth,
                    "POE_l":POE_l,
                    "POE_r":POE_r,
                    "POE":POE}

        return_df = pd.DataFrame(return_dict)

        return return_df, None


class MonteCarlo:
    def __init__(self, config=None):
        self.config = config
        self.set_config()
        return None

    @staticmethod
    def PrintConfigurationTemplate():
        print("""config = dict(
              iterations = float,
              model = [CORR, SCC, MD, RD, CSCC],
              run_date = str,
              weibull_shape = float,
              weibull_scale = float,
              leak_thresh = float,
              rupt_thresh = float,
              cgr = func,
              lcgr = func
              )     
        """)

    @staticmethod
    def random_prob_gen(x, iterations=1, features=1):
        temp=[]
        for var in range(x):
            temp.append(np.random.rand(iterations,features))
        return np.array(temp, dtype=np.float64)

    @staticmethod
    def ls_rupture(operating_pressure, fail_pressure, thresh=1.0, bulk=False):
        """
        Limit state for rupture failure mode of corrosion
        :param fail_pressure: failure pressure, in kPa (SI), or psi (US)
        :param operating_pressure: operating pressure, in kPa (SI), or psi (US)
        :param bulk: Flag for applying limit state on a bulk basis, default is False
        :return: returns array(s) of 1's and 0's, where 1 indicates a failure, and 0 indicates no failure
        """

        ruptures = fail_pressure <= thresh * operating_pressure
        ruptures = ruptures.astype(int)
        if bulk:

            rupture_count = np.sum(ruptures, axis=0)
        else:

            rupture_count = np.sum(ruptures)
        return ruptures, rupture_count

    @staticmethod
    def ls_leak(wt, fD, thresh=0.8, bulk=False):
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

    @staticmethod
    def ls_total(fail_1, fail_2, bulk=False):

        fails = np.maximum(fail_1, fail_2)
        if bulk:

            fail_count = np.sum(fails, axis=0)
        else:

            fail_count = np.sum(fails)
        return fails, fail_count

    @staticmethod
    def ls_dent_fail(NF, n, bulk=False):

        fails = NF < n
        fails = fails.astype(int)
        if bulk:
            fail_count = np.sum(fails, axis=0)
        else:
            fail_count = np.sum(fails)
        return fails, fail_count

    #weibull Distribution
    # shape = 1.439
    # scale = 0.1
    @staticmethod
    def cgr_weibull(p,shp=1.439,scl=0.1):
        '''
        Returns the imperfection growth rate based on a Weibull distribution inverse cumulative density function, using shape and scale parameters.
        Requires Numpy module
        :param p:   random number between 0 and 1
        :param shp: shape parameter, default of 1.439
        :param scl: scale parameter, default of 0.1
        :return:    imperfection growth rate in mm/yr
        '''
        return  scl*np.power( -np.log(1.0-p) ,1.0/shp)

    def set_config(self):
        """Allows to se the monte carlo run date to something other than current datetime

        Keyword Arguments:
            run_date {string} -- the date in YYYY-MM-DD format to set (default: {None})
        """
        model_dict = {'CORR':self.corrpoe,
                        'SCC':self.sccpoe,
                        'MD':self.mdpoe,
                        'RD':self.rdpoe,
                        'CSCC':self.csccpoe}
        self.model = model_dict[self.config['model']]

        if 'iterations' in self.config:
            self.iterations = self.config['iterations']
        else:
            self.iterations = 10

        if 'run_date' in self.config:
            self.now = datetime.datetime.strptime(self.config['run_date'], '%Y-%m-%d').date()
        else:
            self.now = datetime.date.today()

        if 'weibull_shape' in self.config:
            self.weibull_shape = self.config['weibull_shape']
        else:
            self.weibull_shape = 1.439

        if 'weibull_scale' in self.config:
            self.weibull_scale = self.config['weibull_scale']
        else:
            self.weibull_scale = 0.1

        if 'leak_thresh' in self.config:
            self.leak_thresh = self.config['leak_thresh']
        else:
            self.leak_thresh = 0.80

        if 'rupt_thresh' in self.config:
            self.rupt_thresh = self.config['rupt_thresh']
        else:
            self.rupt_thresh = 1.0

        pass

    def get_data_from_csv(self, filepath):
        """searches for a CSV file in the same folder of this file and loads it to a pandas dataframe.

        Arguments:
            filepath {string} -- name of the CSV file containing the data

        Returns:
            None -- 
        """        
        CSV_FILE_NAME = abspath(join(dirname(__file__), filepath))
        self.df = pd.read_csv(CSV_FILE_NAME, header=0)
        self.process_dates()
        return None
    
    def process_dates(self):
        """converts specific fields in the input data to pandas datetime objects
        """        
        self.df['ILIRStartDate'] = pd.to_datetime(self.df['ILIRStartDate']).dt.date
        self.df['install_date'] = pd.to_datetime(self.df['install_date']).dt.date

    # Need to refine this section so that it can create a template spreadsheet to input data into
    def build_template(self, output=False):
        """exports a CSV file with column templates signaling all inputs possible to the POE calculation

        Returns:
            None -- 
        """        
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
                'MAOP_kPa',
                'PMax_kPa',
                'PMin_kPa',
                'AESC']

        data = [[None]] * len(cols)

        temp_dict = {x:y for x,y in zip(cols,data)}
        temp_df = pd.DataFrame(temp_dict)
        if output:
            temp_df.to_csv(abspath(join(dirname(__file__), 'inputs_POE_template.csv')), index=False)
        return temp_df

    def index_marks(self, nrows, chunk_size):
        return range(chunk_size, np.ceil(nrows / chunk_size).astype(int) * chunk_size, chunk_size)

    def split_df(self, chunk_size):
        indices = self.index_marks(self.df.shape[0], chunk_size)
        return np.split(self.df, indices)

    def run(self, split_features=False, buffer_size=1000, split_iterations=False, iterations_split=100):
        """Method used to start the Monte Carlo run

        Keyword Arguments:
            split_features {bool} -- Set to true, the calculation will use the buffer_size to split the instance dataframe by that number (default: {False})
            buffer_size {int} -- value to split the instance dataframe (default: {1000})

        Returns:
            None -- 
        """        
        t1 = time.time()

        if hasattr(self,'result'):
            del self.result

        # if (split_features) and (split_iterations):
        #     raise AttributeError("Not possible to split calculation by features and by iterations. Please select only one.")

        if not(split_iterations):
            self.result, self.qc = self.model(self.df, self.iterations)
        else:
            split = self.iterations//iterations_split
            if split >= self.iterations:
                raise AttributeError("Please select a higher number to split iterations.")
            remainder = self.iterations - split*iterations_split
            if remainder == 0:
                iter_list = [split]*iterations_split
            else:
                iter_list = [split]*iterations_split + [remainder]
            update_cols = ['fail_count','iterations','rupture_count','leak_count','nan']
            self.result, self.qc = self.model(self.df,iter_list.pop())
            for count in iter_list:
                self.result[update_cols] = self.result[update_cols] + self.model(self.df,count)[0][update_cols]

            self.qc = None

        # if not(split_features):
        #     self.result, self.qc = self.model(self.df,self.iterations)
        # else:
        #     self.result = [self.model(x,self.iterations)[0] for x in self.split_df(buffer_size)]
        #     self.result = pd.concat(self.result)
        #     self.qc = None

        self.result["POE"] = self.result["fail_count"] / self.result["iterations"]
        self.result["POE_l"] = self.result["leak_count"] / self.result["iterations"]
        self.result["POE_r"] = self.result["rupture_count"] / self.result["iterations"]

        self.result["1-POE"] = 1 - self.result["POE"]
        agg_POE = 1 - np.prod(self.result["1-POE"])       

        print(f"Model: {self.config['model']} POE Simulation")

        print(f"Count of anomalies: {self.df.shape[0]}")
        if split_iterations:
            print(f"Iterations: {split*iterations_split + remainder:,}")
        else:
            print(f"Iterations: {self.iterations:,}")
        print(f"Date of analysis: {self.now}")
        if 'weibull_shape' in self.config:
            print(f"Weibull Shape: {self.weibull_shape}")
        if 'weibull_scale' in self.config:
            print(f"Weibull Scale: {self.weibull_scale}")
        print(f"Leak threshold modifier: {self.leak_thresh}")
        print(f"Rupture threshold modifier: {self.rupt_thresh}")

        print("Aggregated POE for these features is {}.\n".format(agg_POE))
        print(f"Calculation took {time.time()-t1:.4f} seconds.")
        return None

    def merge_result(self, key):
        return self.result.merge(self.df, on=key)

    def corr_d_cgr(self, **kwargs):
        # weibull Distribution parameters
        WTm = kwargs['WTm']
        surface = kwargs['surface']
        random = kwargs['random']
        tool_D = kwargs['tool_D']
        time_delta = kwargs['time_delta']
        vendor_cgr = kwargs['vendor_cgr']
        vendor_cgr_sd = kwargs['vendor_cgr_sd']
      
        ## default CGR methodology.
        alpha = np.where(np.isnan(vendor_cgr),
                        np.nan,
                        np.where(np.isnan(vendor_cgr_sd),
                                vendor_cgr/(WTm*tool_D/time_delta),
                                vendor_cgr/vendor_cgr_sd))

        fD_GR_default = np.where(surface=="E",
                                MonteCarlo.cgr_weibull(random,1.6073,0.1)/25.4,
                                MonteCarlo.cgr_weibull(random,3.29625,0.1)/25.4)

        fD_GR = np.where(np.logical_or(np.isnan(vendor_cgr),alpha<3.00),
                        fD_GR_default,
                        np.where(np.isnan(vendor_cgr_sd),
                                np.maximum(0, norm.ppf(random, loc=vendor_cgr, scale=vendor_cgr/(WTm*tool_D/time_delta))) / 25.4,
                                np.maximum(0, norm.ppf(random, loc=vendor_cgr, scale=vendor_cgr_sd)) / 25.4))

        return fD_GR

    def corr_l_cgr(self, **kwargs):
        fL_GR = 0.00
        return fL_GR

    def model_error(self, p):
            return 0.914 + gamma.ppf(p, 2.175, scale=0.225)

    def md_d_cgr(self, **kwargs):
        cD_run = kwargs['cD_run']
        ILI_age = kwargs['ILI_age']

        cD_GR = 2*cD_run/ILI_age

        return cD_GR
    
    def md_l_cgr(self, **kwargs):
        cL_run = kwargs['cL_run']
        ILI_age = kwargs['ILI_age']

        cL_GR = 2*cL_run/ILI_age

        return cL_GR

    def scc_d_cgr(self, **kwargs):
        random = kwargs['random']
        cD_GR = self.cgr_weibull(random, self.weibull_shape, self.weibull_scale)
        return cD_GR

    def scc_l_cgr(self, **kwargs):
        cL_GR = 0.00
        return cL_GR

    def cscc_d_cgr(self, **kwargs):
        random = kwargs['random']
        cD_GR = 0.26*np.power( -np.log(1-random) ,1/2.0)
        return cD_GR

    def cscc_l_cgr(self, **kwargs):
        random = kwargs['random']
        cL_GR = 3.126402*np.power( -np.log(1-random) ,1/1.874228039)
        return cL_GR

    def corrpoe(self, df, n):

        # Number of features is equal to number of rows in csv file
        i = self.df.shape[0]

        # Setting the inputs to the appropriate variables
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
        surface = df['ILIFSurfaceInd'].values
        vendor_cgr = df['vendor_cgr_mmpyr'].values
        # df['vendor_cgr_sd'] = df['vendor_cgr_sd'].fillna(df['vendor_cgr_mmpyr']*0.10)
        vendor_cgr_sd = df['vendor_cgr_sd'].values

        vendor = df['vendor'].values
        tool = df['tool'].values
        Insp = df['ILIRStartDate']

        time_delta = ((self.now - Insp).dt.days.values) / 365.25
        ILI_age = ((Insp - Inst).dt.days.values) / 365.25

        # unit conversion to US units
        WT = WTm / 25.4
        S = (Sm * 1000) / 6.89476
        OP = OPm / 6.89476

        # statistics for the inputs
        meanWT = 1.01
        sdWT = 0.01
        meanS = 1.09
        sdS = 0.044

        # tool tolerances
        defectTol = {
            "Rosen": {
                "sdPDP": 0.078,  # in fraction WT
                "sdL": 0.61  # in inches
            },
            "RosenF": {
                "sdPDP": 0.078,  # in fraction WT
                "sdL": 0.61  # in inches
            }
        }

        tool_D = np.select([vendor =="Rosen",vendor=="RosenF"],
                            [defectTol["Rosen"]["sdPDP"],defectTol["RosenF"]["sdPDP"]], default=0.078)
        tool_L = np.select([vendor =="Rosen",vendor=="RosenF"],
                            [defectTol["Rosen"]["sdL"],defectTol["RosenF"]["sdL"]], default=0.61)

        # non-distributed variables
        OD = np.tile(OD, (n, 1)) 
        OP = np.tile(OP, (n, 1))

        np.random.seed()

        mE_n_1, WT_n_2, S_n_3, Y_n_4, fL_n_5, fD_n_6, fGR_n_7 = MonteCarlo.random_prob_gen(7, iterations=n, features=i)

        # distributed variables
        WTd = norm.ppf(WT_n_2, loc=WT * meanWT, scale=WT * sdWT)
        Sdist = norm.ppf(S_n_3, loc=S * meanS, scale=S * sdS)

        # feature length in inches
        fL_run = np.maximum(0, norm.ppf(fL_n_5, loc=fL_measured * 1.0 * (1 / 25.4), scale=tool_L))

        fL_GR = self.corr_l_cgr()

        fL = fL_run + fL_GR * time_delta

        # feature depth in inches
        fD_run = np.maximum(0, norm.ppf(fD_n_6, loc=fPDP * WT * 1.0, scale=tool_D * WT))

        fD_GR = self.corr_d_cgr(WTm=WTm,
                                surface=surface, 
                                random=fGR_n_7,
                                tool_D = tool_D, 
                                time_delta=time_delta, 
                                vendor_cgr=vendor_cgr, 
                                vendor_cgr_sd=vendor_cgr_sd)

        fD = fD_run + fD_GR * time_delta    

        modelError = self.model_error(mE_n_1)

        # Failure Stress in psi
        failure_stress = modified_b31g(OD, WTd, Sdist, fL, fD, units="US") * modelError

        # Failure pressure in psi
        failPress = 2 * failure_stress * WTd / OD

        ruptures, rupture_count = MonteCarlo.ls_rupture(OP, failPress, thresh=self.rupt_thresh, bulk=True)

        leaks, leak_count = MonteCarlo.ls_leak(WTd, fD, thresh=self.leak_thresh, bulk=True)

        fails, fail_count = MonteCarlo.ls_total(ruptures, leaks, bulk=True)

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

    def mdpoe(self, df, n):

        #Number of features is equal to number of rows in csv file
        i = self.df.shape[0]

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
        vendor_cgr = df['vendor_cgr_mmpyr'].values
        df['vendor_cgr_sd'] = df['vendor_cgr_sd'].fillna(df['vendor_cgr_mmpyr']*0.10)
        vendor_cgr_sd = df['vendor_cgr_sd'].values

        vendor = df['vendor'].values
        tool = df['tool'].values
        Insp = df['ILIRStartDate']

        time_delta = ((self.now-Insp).dt.days.values)/365.25
        ILI_age = ((Insp-Inst).dt.days.values)/365.25

        #Young's modulus in psi
        E = 30.0e6

        #Crack fracture area in sq. inches
        fa = 0.124

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

        defectTol = {
            "UTCD": {
                "sdPDP": np.where(WTm*cPDP < 4, 0.780, 3.120),  # absolute mm
                "sdL": 7.80  # in mm
            },
            "AFD": {
                "sdPDP": 0.195,  # fraction of WT
                "sdL": 15.6  # in mm
            },
            "EMAT": {
                "sdPDP": np.where(WTm < 10, 0.117, 0.156),
                "sdL": 7.8  # in mm
            }
        }

        # tool_D = 0.121*WTm
        # tool_L = 8.034  #in mm
        # tool_D = 0.78
        # tool_L = 7.80  #in mm

        tool_D = np.select([tool =="AFD",tool=="EMAT",tool=="UTCD"],
                            [defectTol["AFD"]["sdPDP"]*WTm,defectTol["EMAT"]["sdPDP"]*WTm,defectTol["UTCD"]["sdPDP"]])
        tool_L = np.select([tool =="AFD",tool=="EMAT",tool=="UTCD"],
                            [defectTol["AFD"]["sdL"],defectTol["EMAT"]["sdL"],defectTol["UTCD"]["sdL"]])

        #non-distributed variables
        
        OP = np.tile(OP, (n, 1))
        T = np.tile(T, (n, 1))

        #distributed variables
        np.random.seed()

        OD_n_1, WT_n_2, S_n_3, Y_n_4, cL_n_5, cD_n_6, cGR_n_7, cGRL_n_8 = MonteCarlo.random_prob_gen(8, iterations=n, features=i)

        #distributed variables imperial
        ODd = norm.ppf(OD_n_1, loc=OD*meanOD, scale=OD*sdOD)
        WTd = norm.ppf(WT_n_2, loc=WT*meanWT, scale=WT*sdWT)
        Sdist = norm.ppf(S_n_3, loc=S*meanS, scale=S*sdS)
        Ydist = norm.ppf(Y_n_4, loc=E*meanE, scale=E*sdE)

        #crack length in inches
        cL_run = np.maximum(1, norm.ppf(cL_n_5, loc=cL_measured * 1.0, scale=tool_L)) / 25.4
        
        cL_GR = self.md_l_cgr(
                                cL_run = cL_run,
                                ILI_age = ILI_age
                            )
        
        cL = cL_run +  cL_GR*time_delta

        #Crack detpth in inches
        cD_run = np.maximum(0, norm.ppf(cD_n_6, loc=cPDP*WTm*1.0, scale=tool_D))/25.4
        
       
        cD_GR = self.md_d_cgr(
                                cD_run = cD_run,
                                ILI_age = ILI_age
                            )
        
        cD = cD_run +  cD_GR*time_delta

        failStress = modified_lnsec(ODd, WTd, Sdist, T, Ydist, cL, cD, units="US")

        #Failure pressure in psi
        failPress = 2*failStress*WTd/ODd

        fsNaNs = np.extract(np.isnan(failStress),failStress)
        fsNaNs_count = fsNaNs.size

        ruptures, rupture_count = MonteCarlo.ls_rupture(OP, failPress, thresh=self.rupt_thresh, bulk=True)

        leaks, leak_count = MonteCarlo.ls_leak(WTd, cD, thresh=self.leak_thresh, bulk=True)

        fails, fail_count = MonteCarlo.ls_total(ruptures, leaks, bulk=True)

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

    def sccpoe(self, df, n):

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
        shape, scale = self.weibull_shape, self.weibull_scale

        #Young's modulus in psi
        E = 30.0e6

        #Crack fracture area in sq. inches
        fa = 0.124

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
            "Rosen": {
                "sdPDP": 0.78,   # in mm (0.78)
                "sdL": 7.80      #in mm (7.80)
                },
            "RosenF": {
                "sdPDP":np.where(WTm < 10, 0.117, 0.156)*WTm, #in mm
                "sdL":15.61  #in mm
                },
            "Rosen3": {
                "sdPDP":np.where(cPDP*WTm < 4, 0.780, 3.120), #in mm
                "sdL":7.80  #in mm
                },
            "PII": {
                "sdPDP": 0.31,  # in mm
                "sdL":6.10      #in mm
                }
            }

        tool_D = np.select([vendor =="Rosen",vendor=="RosenF",vendor=="PII",vendor=='Rosen3'],
                            [defectTol["Rosen"]["sdPDP"],defectTol["RosenF"]["sdPDP"],defectTol["PII"]["sdPDP"],defectTol["Rosen3"]["sdPDP"]])
        tool_L = np.select([vendor =="Rosen",vendor=="RosenF",vendor=="PII",vendor=='Rosen3'],
                            [defectTol["Rosen"]["sdL"],defectTol["RosenF"]["sdL"],defectTol["PII"]["sdL"],defectTol["Rosen3"]["sdL"]])

        
        #non-distributed variables
        OP = np.tile(OP, (n, 1))
        T = np.tile(T, (n, 1))

        #distributed variables
        np.random.seed()

        OD_n_1, WT_n_2, S_n_3, Y_n_4, cL_n_5, cD_n_6, cGR_n_7 = MonteCarlo.random_prob_gen(7, iterations=n, features=i)

        #distributed variables imperial
        ODd = norm.ppf(OD_n_1, loc=OD*meanOD, scale=OD*sdOD)
        WTd = norm.ppf(WT_n_2, loc=WT*meanWT, scale=WT*sdWT)
        Sdist = norm.ppf(S_n_3, loc=S*meanS, scale=S*sdS)
        Ydist = norm.ppf(Y_n_4, loc=E*meanE, scale=E*sdE)

        #crack length in inches
        cL_run = np.maximum(1, norm.ppf(cL_n_5, loc=cL_measured * 1.0, scale=tool_L)) / 25.4
        cL_GR = self.scc_l_cgr()
        cL = cL_run + cL_GR * time_delta

        #Crack detpth in inches
        cD_run = np.maximum(0, norm.ppf(cD_n_6, loc=cPDP*WTm*1.0, scale=tool_D))/25.4
        cD_GR = self.scc_d_cgr(random = cGR_n_7) / 25.4

        cD = cD_run + cD_GR * time_delta

        failStress = modified_lnsec(ODd, WTd, Sdist, T, Ydist, cL, cD, units="US")

        #Failure pressure in psi
        failPress = 2*failStress*WTd/ODd

        fsNaNs = np.extract(np.isnan(failStress),failStress)
        fsNaNs_count = fsNaNs.size

        ruptures, rupture_count = MonteCarlo.ls_rupture(OP, failPress, thresh=self.rupt_thresh, bulk=True)

        leaks, leak_count = MonteCarlo.ls_leak(WTd, cD, thresh=self.leak_thresh, bulk=True)

        fails, fail_count = MonteCarlo.ls_total(ruptures, leaks, bulk=True)

        return_dict = {"FeatureID":cids,
                    "fail_count":fail_count,
                        "iterations":np.size(fails, axis=0),
                        "rupture_count":rupture_count,
                        "leak_count":leak_count,
                        "nan": fsNaNs_count,#np.zeros((i,)),
                        "PDP_frac":cPDP,
                        "clength":cL_measured}

        ####Artificially added------------------------------
        # amt = 1000
        # qc_list = [ODd, WTd, Sdist, T, OP, np.tile(Inst.values, (n,1)),
        #             cD_run, cD, cD_GR, cL, failStress, failPress, ruptures, leaks, fails]
        # qc_cols = ['ODd', 'WTd', 'Sdist', 'T', 'OP', 'Inst',
        #             'cD_run', 'cD', 'cD_GR', 'cL', 'failStress', 'failPress', 'ruptures', 'leaks', 'fails']
        # qc_dict = dict()
        # qc_dict = {x:y[:amt,0] for x,y in zip(qc_cols,qc_list)}
        
        # qc_df = pd.DataFrame(qc_dict)
        # qc_df.loc[:,'E']=E
        # qc_df.loc[:,'fa']=fa
        # qc_df.loc[:,'Insp']=Insp.values[0]
        # qc_df.loc[:,'vendor']=vendor[0]
        # qc_df.loc[:,'tool']=tool[0]
        # qc_df.loc[:,'tool_D']=tool_D[0]
        # qc_df.loc[:,'tool_L']=tool_L[0]
        ####-----------------------------------------------

        return_df = pd.DataFrame(return_dict)
        return return_df, None

    def csccpoe(self, df, n):
        i = self.df.shape[0]
        
        OD = df['OD_inch'].values
        WTm = df['WT_mm'].values

        cPDP = df['depth_fraction'].values
        cL_measured = df['width_mm'].values
        cchainage = df['chainage_m'].values
        cstatus = df['status'].values
        ctype = df['type'].values
        cids = df['FeatureID'].values


        vendor = df['tool'].values
        Insp = df['ILIRStartDate']

        time_delta = ((self.now-Insp).dt.days.values)/365.25

        # Unit conversion to metric
        OD = OD*25.4
        WT = WTm

        meanOD = 1.0    #fraction of OD
        sdOD = 0.0006   #fraction of OD
        meanWT = 1.01   #fraction of WT
        sdWT = 0.01     #fraction of WT

        #fraction of WT
        sdDepthTool = np.where(WTm<10,0.117,0.156)

        #tool tolerances
        defectTol = {
            "RosenF": {
                "sdPDP":0.78,   # in mm
                "sdL":7.80      #in mm
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

        tool_D = np.select([vendor =="Rosen",vendor=="RosenF",vendor=="PII"],
                            [defectTol["Rosen"]["sdPDP"],defectTol["RosenF"]["sdPDP"],defectTol["PII"]["sdPDP"]])
        tool_L = np.select([vendor =="Rosen",vendor=="RosenF",vendor=="PII"],
                            [defectTol["Rosen"]["sdL"],defectTol["RosenF"]["sdL"],defectTol["PII"]["sdL"]])

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
        cL_GR = self.cscc_l_cgr(random=cGR_n_7)	#PMC from width distribution
        cL = cL_run + cL_GR*(time_delta)

        #Crack detpth in mm
        cD_run = np.maximum(0, norm.ppf(cD_n_6, loc=cPDP*WTm*1.0, scale=tool_D))
        cD_GR = self.cscc_d_cgr(random = cGR_n_8)	#Standard
        cD = cD_run + cD_GR*(time_delta)

        depth_thresh = 0.6
        leaks = cD >= depth_thresh*WTd
        leaks = leaks.astype(int)
        
        circ_thresh = 0.4
        circ_thresh_depth = 0.4
        ruptures = (cL >= circ_thresh*np.pi*ODd) & (cD >= circ_thresh_depth*WTd)
        ruptures = ruptures.astype(int)

        fails = np.maximum(leaks,ruptures)
        fails = fails.astype(int)

        fail_count = np.sum(fails, axis=0)
        leak_count = np.sum(leaks, axis=0)
        rupture_count = np.sum(ruptures, axis=0)

        return_dict = {"FeatureID":cids,
                        "fail_count":fail_count,
                        "iterations":np.size(fails, axis=0),
                        'leak_count':leak_count,
                        'rupture_count':rupture_count,
                        "PDP_frac":cPDP,
                        "clength":cL_measured}

        return_df = pd.DataFrame(return_dict)

        return return_df, None

    # RD WIP
    def rdpoe(self, df, n):
        # Number of features is equal to number of rows in csv file
        i = self.df.shape[0]

        OD = df['OD_inch'].values
        WTm = df['WT_mm'].values
        Sm = df['grade_MPa'].values
        df['toughness_J'] = df['toughness_J'].fillna(10)
        Tm = df['toughness_J'].values
        Inst = df['install_date']

        OPm = df['MAOP_kPa'].values
        
        MAXPm = df['PMax_kPa'].values
        MINPm = df['PMin_kPa'].values
        cycles = df['AESC'].values

        # outside diameter in mm
        ODm = OD * 25.4

        ###SOMETHING IS WRONG HERE###--------------------------------------
        UTSm = ultimate_determiner(Sm)

        # Operating Stress in MPa
        MAXStr = (MAXPm * ODm) / (2000 * WTm)
        MINStr = (MINPm * ODm) / (2000 * WTm)

        dPDP = self.df['depth_fraction'].values
        dL_measured = self.df['length_mm'].values
        dW_measured = self.df['width_mm'].values
        dchainage = self.df['chainage_m'].values
        dstatus = self.df['status'].values
        dtype = self.df['type'].values
        dtype_gouge = self.df['type'].str.contains(r'.*dent.*gouge.*',case=False).values
        dids = self.df['FeatureID'].values

        # gouge depth pct
        gPDP = np.where(dtype_gouge,
                        0.10,
                        0.00)

        # Inline inspection range properties
        vendor = df['vendor'].values
        tool = df['tool'].values
        Insp = df['ILIRStartDate']

        # time_delta = (self.now - Inst).dt.days.values / 365.25
        time_delta = (self.now - Insp).dt.days.values / 365.25

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

        OD_n_1, WT_n_2, UTS_n_3, g_n_4, dW_n_5, dL_n_6, dD_n_7 = MonteCarlo.random_prob_gen(7, iterations=n, features=i)

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

        NF = nf_EPRG(ODd, WTd, UTSd, dL_run, dW_run, dD_run, gD, MAXStr, MINStr, sf=sf, model='Refined-EPRG-2000')

        ## TEST For TMC 20200831
        # ODt = ODm/WTd
        #kMAX 1
        # kMAX = 7.5*(1-np.exp(-0.065*ODt))
        #kMAX 2
        # kMAX = 9.4*(1-np.exp(-0.045*ODt))
        #kMAX from EPRG
        # dRL = (np.power(dL_run, 2.0) + 4.0 * np.power(dD_run, 2.0)) / (8.0 * dD_run)
        # dRW = (np.power(dW_run, 2.0) + 4.0 * np.power(dD_run, 2.0)) / (8.0 * dD_run)
        # dR = np.minimum(dRW, dRL)
        # Crd = np.where(dR < 5.0*WTd, 1.0, 2.0)
        # dSEF = 1 + Crd*np.power(np.power(dD_run,1.5)*WTd/ODd,0.5)
        # gSEF = 1 + 9.0*(gD/WTd)
        # kMAX = gSEF*dSEF
        # NF = np.power(10.0,12.6007-3.0*np.log10(kMAX*(MAXStr-MINStr)))

        n_cycles = time_delta * cycles

        fails, fail_count = MonteCarlo.ls_dent_fail(NF, n_cycles, bulk=True)

        return_dict = {"FeatureID":dids,
                    "fail_count": fail_count,
                    "leak_count": fail_count,
                    "rupture_count": fail_count,
                    "iterations": np.size(fails, axis=0),
                    "NPS_Frac": dPDP,
                    "dlength": dL_measured,
                    "dwidth": dW_measured,
                    "time_delta":time_delta,}

        return_df = pd.DataFrame(return_dict)

        # ##UPDATE HERE-------------------------------------------------------------------------------
        # feat = 0
        # cols = [ODd,WTd,UTSd,gD,dW_run,dL_run,dD_run,NF,fails]
        # cols_lab = ['ODd','WTd','UTSd','gD','dW_run','dL_run','dD_run','NF','fails']

        # qc_dict = dict()
        # qc_dict = {x:np.take(y,feat,axis=1) for x,y in zip(cols_lab,cols)}
        # qc_df = pd.DataFrame(qc_dict)
        # qc_df['n_cycles']=n_cycles[0]
        # qc_df['Inst']=Inst[0]
        # qc_df['vendor']=vendor[0]
        # qc_df['tool']=tool[0]
        # qc_df['tool_D']=tool_D[0]
        # qc_df['tool_L']=tool_L[0]
        # qc_df['tool_W']=tool_W[0]
        ##-----------------------------------------------------------------------------------------
        if i==1:
            qc_dict = {'ODd': ODd.reshape(-1),
                        'WTd': WTd.reshape(-1),
                        'UTS': UTSd.reshape(-1),
                        'gD': gD.reshape(-1),
                        'dW_run': dW_run.reshape(-1),
                        'dL_run': dL_run.reshape(-1),
                        'dD_run': dD_run.reshape(-1),
                        'NF':NF.reshape(-1),
                        "fail_count": fails.reshape(-1)}

            qc_df = pd.DataFrame(qc_dict)
        else:
            qc_df = None
        
        return return_df, qc_df



if __name__ == '__main__':

    pd.set_option('display.max_columns',500)
    

    config = dict(iterations=1000_000,
                    model = 'CORR'
    )

    class PMC(MonteCarlo):
        def __init__(self, config):
            super().__init__(config)
        
        def corr_d_cgr(self, **kwargs):
            return 0.1

    corr = MonteCarlo(config=config)
    
    corr.get_data_from_csv('sample_of_inputs.csv')
    corr.run()
    print(corr.result)

    # folder_path = r"C:\\Users\\armando_borjas\\Documents\\Filed Work\\PMC\\Resident Damage Tests\\"
    #folder_path = r"C:\\Users\\armando_borjas\\Documents\\Python\\Calculators\\"

    #input_filename = "sample_of_inputs.csv"
    #output_filename = "output__res_tests.csv"

    #res = MonteCarlo('RD', config=config)
    #res.get_data_from_csv(folder_path+input_filename)
    #res.run()
    #res.merge_result(key='FeatureID').to_csv(folder_path+output_filename)

