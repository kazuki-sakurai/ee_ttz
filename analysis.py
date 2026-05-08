import sys, os
sys.path.append('/Users/kazuki/Projects/pyHELAS')
import numpy as np
from math import sqrt, cos, sin, acos, asin, pi
import vector
#from decay_vec import Decay_3B
from QI_functions import *
from ee_ttz import *
from ee_ttz_func import *

def pretty_output(header, lis, mergin=1):
    lis = np.array(lis)
    tlis = lis.transpose()
    widths = []
    for i in range(len(header)):
        str_tlis = [ len(str(r)) for r in tlis[i] ] + [len(header[i])]
        w = np.max( str_tlis ) + mergin
        widths.append(w)

    print(" ".join(f"{h:>{w}}" for h, w in zip(header, widths)))
    print(" ".join("-"*(w) for w in widths))

    for ar in lis:
        print(" ".join(f"{elem:>{w}}" for elem, w in zip(ar, widths)))


######################################################

rs = 1000

pp = {}
pp['E'] = np.array([ 500,    0,    0, 500])
pp['EB'] = np.array([500,   0,   0, -500,] )

mttmax = rs - mZ
mttmin = 2*mt

entries = ['pure', 'pure1', 'pure2', 'pure3', 'c12', 'EN12', 'EN13', 'EN23', 'c1', 'c2', 'c3', 'EN1', 'EN2', 'EN3']

nx, ny = 5, 5
for pol in ['1', '2', 'unpol']:

    for mode in range(1):

        if mode == 0: # x=m12, y=th3       
            xar = np.linspace(mttmax, mttmin, nx)
            yar = np.linspace(-1, 1, ny)

        save = {}
        for ent in entries: save[ent] = []

        for xdm in xar:
            for ydm in yar:

                if mode == 0:
                    m12 = xdm
                    cth3 = ydm
                    cth1 = 0 
                    ph = 0

                p1, p2, p3 = get_momenta(rs, m12, mt, mZ, cth3, cth1, ph)
                pp['T']  = np.array( [ p1.E, p1.px, p1.py, p1.pz ] )
                pp['TB'] = np.array( [ p2.E, p2.px, p2.py, p2.pz ] )
                pp['Z']  = np.array( [ p3.E, p3.px, p3.py, p3.pz ] )

                #R_empty = np.zeros((12, 12), dtype=np.complex128)

                Rpol = []
                ##
                for lE in [1, -1]:
                    for lEB in [1, -1]:

                        if lE*lEB > 0: continue
                        if pol == '1' and lE < 0: continue  
                        if pol == '2' and lE > 0: continue 

                        ###
                        amphel = []
                        for lT in [1, -1]:
                            for lTB in [1, -1]:
                                for lZ in [1, 0, -1]:

                                    hel = {'E':lE, 'EB':lEB, 'T':lT, 'TB':lTB, 'Z':lZ}
                                    amphel.append( get_amphel(pp, hel).sum() )

                        amphel = np.array(amphel)
                        Rpol.append( np.outer(amphel, amphel.conj()) )

                if pol == 'unpol':
                    Rpol = np.array(Rpol)
                    Rmat = np.sum(Rpol, axis=0)
                else:
                    Rmat = Rpol[0]

                rho = normalise(Rmat)
                save['pure'].append( purity(rho) )

                rho_tens = rho.reshape(2,2,3,2,2,3)
                rho1 = np.einsum('xijxab->ijab', rho_tens).reshape(6, 6)
                rho2 = np.einsum('ixjaxb->ijab', rho_tens).reshape(6, 6)
                rho3 = np.einsum('ijxabx->ijab', rho_tens).reshape(4, 4)

                save['EN23'].append( log_neg_bip(rho1, [2,3]) )
                save['EN13'].append( log_neg_bip(rho2, [2,3]) )
                save['EN12'].append( log_neg_bip(rho3, [2,2]) )
                save['c12'].append( concurrence(rho3) )

                pure1 = purity(rho1)
                pure2 = purity(rho2)
                pure3 = purity(rho3)

                c1 = sqrt(2*(1-pure1))
                c2 = sqrt(2*(1-pure2))
                c3 = sqrt(2*(1-pure3))

                save['pure1'].append( pure1 )
                save['pure2'].append( pure2 )
                save['pure3'].append( pure3 )

                save['c1'].append( c1 )
                save['c2'].append( c2 )
                save['c3'].append( c3 )

                save['EN1'].append( log_negativity(rho, 'A') ) 
                save['EN2'].append( log_negativity(rho, 'B') )  
                save['EN3'].append( log_negativity(rho, 'C') ) 

                #print(rho_tt - rho_mg5)

                #print( np.trace(rho_mg5@rho_mg5), np.trace(rho_tt@rho_tt) )



