import sys, os
sys.path.append('/Users/kazuki/Projects/pyHELAS')
import numpy as np
from math import sqrt, cos, sin, acos, asin, pi
import pyHELAS 
from pyHELAS import COUP

MeV = 10**-3
mH = 125.
mW = 80.369
mZ = 91.188
mt = 173
mtau = 1.777
Twidth = 1.4915
Zwidth = 2.4414
Wwidth = 2.085
Hwidth = 4*MeV

Gf = 1.16639e-5
#aMZ = 1/128
aMZ = 1/132.507
thw = 1/2 *np.arcsin(2*np.sqrt(np.sqrt(2)/2 *pi *aMZ /(Gf *mZ**2)))
sinW = np.sin(thw)
cosW = np.cos(thw)

gev2tobarn = 0.3894e-3

gA = sqrt(4*pi*aMZ)
gW = gA/sinW
gZ = gA/(sinW*cosW)

vev = 2*mZ/gZ
lam = vev**2 / mH**2

###################
# Diagram-1 
###################
# e+ e- > Z -> h Z, h -> t tb
def get_amp1(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(EE, EB, COUP['ZEE'], mZ, Zwidth)
    HC  = pyHELAS.HVVXXX(VC, Zs, COUP['HZZ'], mH, Hwidth)
    amp = pyHELAS.IOSXXX(TB,TT,HC,COUP['HTT'])
    return amp

###################
# Diagram-2
###################
# e+ e- > A -> t tb, t -> t Z
def get_amp2(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    As  = pyHELAS.JIOXXX(EE,EB, COUP['AEE'], 0, 0)
    FO_off = pyHELAS.FVOXXX(TT,VC,COUP['ZUU'],mt,Twidth)
    amp = pyHELAS.IOVXXX(TB,FO_off,As,COUP['AUU'])
    return amp

###################
# Diagram-3
###################
# e+ e- > Z -> t tb, t -> t Z
def get_amp3(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(EE,EB, COUP['ZEE'], mZ, Zwidth)
    FO_off = pyHELAS.FVOXXX(TT,VC,COUP['ZUU'],mt,Twidth)
    amp = pyHELAS.IOVXXX(TB,FO_off,Zs,COUP['ZUU'])
    return amp

###################
# Diagram-4
###################
# e+ e- > A -> t tb, tb -> tb Z
def get_amp4(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    As  = pyHELAS.JIOXXX(EE,EB, COUP['AEE'], 0, 0)
    FI_off = pyHELAS.FVIXXX(TB,VC,COUP['ZUU'],mt,Twidth)
    amp = pyHELAS.IOVXXX(FI_off,TT,As,COUP['AUU'])
    return amp

###################
# Diagram-5
###################
# e+ e- > Z -> t tb, tb -> tb Z
def get_amp5(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(EE,EB, COUP['ZEE'], mZ, Zwidth)
    FI_off = pyHELAS.FVIXXX(TB,VC,COUP['ZUU'],mt,Twidth)
    amp = pyHELAS.IOVXXX(FI_off,TT,Zs,COUP['ZUU'])
    return amp


###################
# Diagram-6
###################
# e- e+ > A- Z+, A- -> t tb
def get_amp6(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    As  = pyHELAS.JIOXXX(TB,TT, COUP['AUU'], 0, 0)
    FI_off = pyHELAS.FVIXXX(EE,As,COUP['AEE'],0,0)
    amp = pyHELAS.IOVXXX(FI_off,EB,VC,COUP['ZEE'])
    return amp

###################
# Diagram-7
###################
# e- e+ > Z- Z+, Z- -> t tb
def get_amp7(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(TB,TT, COUP['ZUU'], mZ, Zwidth)
    FI_off = pyHELAS.FVIXXX(EE,Zs,COUP['ZEE'],0,0)
    amp = pyHELAS.IOVXXX(FI_off,EB,VC,COUP['ZEE'])
    return amp

###################
# Diagram-8
###################
# e- e+ > Z- A+, A+ -> t tb
def get_amp8(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    As  = pyHELAS.JIOXXX(TB,TT, COUP['AUU'], 0, 0)
    FO_off = pyHELAS.FVOXXX(EB,As,COUP['AEE'],0,0)
    amp = pyHELAS.IOVXXX(EE,FO_off,VC,COUP['ZEE'])
    return amp

###################
# Diagram-8
###################
# e- e+ > Z- A+, A+ -> t tb
def get_amp9(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(TB,TT, COUP['ZUU'], mZ, Zwidth)
    FO_off = pyHELAS.FVOXXX(EB,Zs,COUP['ZEE'],0,0)
    amp = pyHELAS.IOVXXX(EE,FO_off,VC,COUP['ZEE'])
    return amp


######################
# get all amplitudes 
######################
def get_amphel(pp, hel):

    # Defining external momenta  
    EE = pyHELAS.IXXXXX(pp['E'], hel['E'], 'fermion') # t
    EB = pyHELAS.OXXXXX(pp['EB'], hel['EB'], 'fbar') # tbar

    TT = pyHELAS.OXXXXX(pp['T'], hel['T'], 'fermion') # t
    TB = pyHELAS.IXXXXX(pp['TB'], hel['TB'], 'fbar') # tbar
    VC = pyHELAS.VXXXXX(pp['Z'], hel['Z'], 'out') # Z
    externals = [EE, EB, TT, TB, VC]

    amp = []
    amp.append(get_amp1(externals, COUP)) 
    amp.append(get_amp2(externals, COUP)) 
    amp.append(get_amp3(externals, COUP)) 
    amp.append(get_amp4(externals, COUP)) 
    amp.append(get_amp5(externals, COUP)) 
    amp.append(get_amp6(externals, COUP)) 
    amp.append(get_amp7(externals, COUP)) 
    amp.append(get_amp8(externals, COUP)) 
    amp.append(get_amp9(externals, COUP)) 

    return np.array(amp)


