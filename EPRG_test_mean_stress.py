import numpy as np
import pandas as pd

def nf_EPRG_current(pipe, dent, sf=1.0, units="SI"):
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

    od, wt, uts, MAX, MIN = pipe.OD, pipe.WT, pipe.UTS, pipe.MAXStr, pipe.MINStr
    dL, dW, dD, gD = dent.L, dent.W, dent.D, dent.gD

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

    dRL = (np.power(dL, 2.0) + 4.0 * np.power(dD, 2.0)) / (8.0 * dD)
    dRW = (np.power(dW, 2.0) + 4.0 * np.power(dD, 2.0)) / (8.0 * dD)
    dR = np.minimum(dRW, dRL)

    Crd = np.where(dR < 5.0*wt, 1.0, 2.0)

    dSEF = 1 + Crd*np.power(np.power(dD,1.5)*wt/od,0.5)
    gSEF = 1 + 9.0*(gD/wt)

    sigma = (1/2)*(MAX-MIN)/(1- np.power((MAX+MIN)/(2.0*uts),2.0) )

    NF = (5622.0/sf)*np.power(uts/(2.0*sigma*gSEF*dSEF),5.26)
    
    calc = results()
    calc.dRL = dRL
    calc.dRW = dRW
    calc.dR = dR
    calc.Crd = Crd
    calc.dSEF = dSEF
    calc.gSEF = gSEF
    calc.sigma = sigma
    calc.NF = NF
    
    return calc

def nf_EPRG_new(pipe, dent, sf=1.0, units="SI"):
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

    od, wt, uts, MAX, MIN = pipe.OD, pipe.WT, pipe.UTS, pipe.MAXStr, pipe.MINStr
    dL, dW, dD, gD = dent.L, dent.W, dent.D, dent.gD
    
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

    dRL = (np.power(dL, 2.0) + 4.0 * np.power(dD, 2.0)) / (8.0 * dD)
    dRW = (np.power(dW, 2.0) + 4.0 * np.power(dD, 2.0)) / (8.0 * dD)
    dR = np.minimum(dRW, dRL)

    Crd = np.where(dR < 5.0*wt, 1.0, 2.0)

    dSEF = 1 + Crd*np.power(np.power(dD,1.5)*wt/od,0.5)
    gSEF = 1 + 9.0*(gD/wt)

    sigma_a = (MAX-MIN)/2
    sigma = (-1 + np.sqrt(1+4*np.power(sigma_a/uts,2) )) / (2*sigma_a/np.power(uts,2))

    NF = (5622.0/sf)*np.power(uts/(2.0*sigma*gSEF*dSEF),5.26)

    calc = results()
    calc.dRL = dRL
    calc.dRW = dRW
    calc.dR = dR
    calc.Crd = Crd
    calc.dSEF = dSEF
    calc.gSEF = gSEF
    calc.sigma_a = sigma_a
    calc.sigma = sigma
    calc.NF = NF
    
    return calc

def nf_EPRG_1995(pipe, dent, sf=1.0, units="SI"):
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

    od, wt, uts, MAX, MIN = pipe.OD, pipe.WT, pipe.UTS, pipe.MAXStr, pipe.MINStr
    dL, dW, dD, gD = dent.L, dent.W, dent.D, dent.gD
    
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

    dSEF = 2.871*np.power(dD*wt/od,0.5)
    gSEF = 1 + 9.0*(gD/wt)

    mean_sigma = (MAX+MIN)/2.
    sigma_a = (MAX-MIN)/2

    B = (sigma_a/uts)/np.power(1- (mean_sigma/uts), 0.5)

    sigma_A_x2 = uts*(B*np.sqrt(4+np.power(B,2)) - np.power(B,2))

    NF = (1000.0)*np.power((uts-50.0)/(sigma_A_x2*dSEF),4.292)

    calc = results()
    calc.dSEF = dSEF
    calc.gSEF = gSEF
    calc.sigma_a = sigma_a
    calc.sigma_A_x2 = sigma_A_x2
    calc.NF = NF
    
    return calc

def nf_PETROBRAS(pipe, dent, sf=1.0, units="SI"):
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

    od, wt, uts, MAX, MIN = pipe.OD, pipe.WT, pipe.UTS, pipe.MAXStr, pipe.MINStr
    dL, dW, dD, gD = dent.L, dent.W, dent.D, dent.gD
    
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

    NF = np.power(sigma_A_fully_rev/C, 1/b)

    calc = results()
    calc.dLdW = dLdW
    calc.dLdD = dLdD
    calc.A = A
    calc.B = B
    calc.K = K
    calc.ka = ka
    calc.Se = Se
    calc.C = C
    calc.b = b
    calc.sigma_a = sigma_a
    calc.sigma_A_fully_rev = sigma_A_fully_rev
    calc.NF = NF
    
    return calc

class pipe:
    def __init__(self, OD, WT, S, MAXP, MINP, AESC):
            self.OD = OD
            self.WT = WT
            self.S = S
            self.MAXP = MAXP
            self.MINP = MINP
            self.MAXStr = self.MAXP*self.OD/(2000.0*self.WT)
            self.MINStr = self.MINP*self.OD/(2000.0*self.WT)
            self.AESC = AESC
            self.UTS = self.ultimate_determiner(self.S)

    def set_maxstr(self, p, mode='pct'):
        if mode == 'pct':
            self.MAXStr = self.S*p
        elif mode == 'pressure':
            self.MAXStr = p*self.OD/(2000.0*self.WT)
        elif mode == 'stress':
            self.MAXStr = p

    def set_minstr(self, p, mode='pct'):
        if mode == 'pct':
            self.MINStr = self.S*p
        elif mode == 'pressure':
            self.MINStr = p*self.OD/(2000.0*self.WT)
        elif mode == 'stress':
            self.MINStr = p

    def sigma_a(self):
        return (self.MAXStr-self.MINStr)/2

    def mean_sigma(self):
        return (1/2)*(self.MAXStr+self.MINStr)

    def current_sigma_A(self):
        return (self.MAXStr-self.MINStr)/(1- np.power((self.MAXStr+self.MINStr)/(2.0*self.UTS),2.0) )

    def gerber_sigma_A(self):
        return self.sigma_a()/(1- np.power((self.MAXStr+self.MINStr)/(2.0*self.UTS),2.0) )

    def goodman_sigma_A(self):
        return self.sigma_a()/(1- np.power((self.MAXStr+self.MINStr)/(2.0*self.UTS),1.0) )

    def soderberg_sigma_A(self):
        return self.sigma_a()/(1- np.power((self.MAXStr+self.MINStr)/(2.0*self.S),1.0) )
    
    def new_sigma_A(self):
        sigma = (-1 + np.sqrt(1+4*np.power(self.sigma_a()/self.UTS,2) )) / (2*self.sigma_a()/np.power(self.UTS,2))
        return sigma

    def swt_sigma_A(self):
        sigma_r = np.sqrt(self.MAXStr*self.sigma_a())
        return sigma_r

    def EPRG1995_sigma_A(self):
        B = (self.sigma_a()/self.UTS)/np.power(1- (self.mean_sigma()/self.UTS), 0.5)
        sigma_A_x2 = self.UTS*(B*np.sqrt(4+np.power(B,2)) - np.power(B,2))
        return sigma_A_x2
    
    def ultimate_determiner(self, grade):
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
		

class dent:
    def __init__(self, D, L, W, gD=0.00):
            self.D = D
            self.L = L
            self.W = W
            self.gD = gD


class results:
    def __init__(self):
        pass
    

p1 = pipe(323.9, 4.9, 372., 8828.34208089, 0.0, 126.5)
p1.UTS = 530.0
p1.set_maxstr(0.72)
p1.set_minstr(0.36)

d1 = dent(0.016*p1.OD, 50.0, 50.0)


current = nf_EPRG_current(p1, d1, sf=5.0)
new = nf_EPRG_new(p1, d1, sf=5.0)

##EPRG1995 = nf_EPRG_1995(p1, d1)
##petro = nf_PETROBRAS(p1,d1)

print(f"Current cycles: {current.NF:.2f}")
print(f"New cycles: {new.NF:.2f}")
##print(f"EPRG1995 cycles: {EPRG1995.NF:.2f}")
##print(f"PETROBRAS cycles: {petro.NF:.2f}")
##
##calcs = ['sigma_a',
##         'gerber_sigma_A',
##         'goodman_sigma_A',
##         'soderberg_sigma_A',
##         'new_sigma_A',
##         'swt_sigma_A',
##         'EPRG1995_sigma_A']
##
##results = [eval(f"p1.{x}()") for x in calcs]
##
##df = pd.DataFrame(results, index=calcs)
##print(df)
