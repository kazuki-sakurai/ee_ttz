import sys, os
sys.path.append('/Users/kazuki/Projects/pyHELAS')
sys.path.append('/Users/kazuki/Packages/mg5amcnlo-madspin_valentin')
import numpy as np
from math import sqrt, cos, sin, acos, asin, pi
#import vector
#from decay_vec import Decay_3B
from QI_functions import *
from ee_ttz import *

import madgraph.various.Density_functions as dens
from madgraph.various.lhe_parser import EventFile, FourMomentum

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


def vec2ar(v): return np.array([v.E, v.px, v.py, v.pz])

def get_momenta(rs, mt, mZ):

    pe  = vector.obj(px=0, py=0, pz = -rs/2, mass = 0 )
    peb = vector.obj(px=0, py=0, pz =  rs/2, mass = 0 )

    p0 = pe + peb
    
    pt, ptb, pz = Decay_3B(p0, mt, mt, mZ)

    pp = {}

    pp['E'] = vec2ar(pe)
    pp['EB'] = vec2ar(peb)
    pp['T'] = vec2ar(pt)
    pp['TB'] = vec2ar(ptb)
    pp['Z'] = vec2ar(pz)

    return pp

######################################################

rs = 1000

for event in EventFile('unweighted_events.lhe'):

    pp = {}
    pp['E'] = np.array([ 500,    0,    0, 500])
    pp['EB'] = np.array([500,   0,   0, -500,] )

    # pp['TB'] = np.array( [ 431.25140599,  124.04042828,  259.27788317, -270.99396048] )
    # pp['Z'] = np.array( [ 364.16590127, -153.40328059, -270.33534429,  166.39647663] )

    for p in event:                    # event is a list of Particle objects
        if p.pid ==  6: pp['T']  = np.array( [ p.E, p.px, p.py, p.pz ] )
        if p.pid == -6: pp['TB'] = np.array( [ p.E, p.px, p.py, p.pz ] )
        if p.pid == 23: pp['Z']  = np.array( [ p.E, p.px, p.py, p.pz ] )


    density = event.density
    rho_instance = dens.DensityMatrixObservables(density)
    rho_mg5 = rho_instance.square_matrix()
    #print(square_density.shape)

    # print('pE:', pp['E'])
    # print('pEB:', pp['EB'])
    # print('pT:', pp['T'])
    # print('pTB:', pp['TB'])
    # print('pZ:', pp['Z'])
    # print('pT + pTB + pZ:', pp['T'] + pp['TB'] + pp['Z'])

    amp = []
    lE, lEB = 1, -1

    Rmat = np.zeros((12, 12), dtype=np.complex128)

    ##
    for lE in [1, -1]:
        for lEB in [1, -1]:
            ###
            amp_pol = []
            for lT in [1, -1]:
                for lTB in [1, -1]:
                    for lZ in [1, 0, -1]:

                        hel = {'E':lE, 'EB':lEB, 'T':lT, 'TB':lTB, 'Z':lZ}
                        amphel = get_amphel(pp, hel)
                        amp_pol.append( np.sum(amphel) )

            amp_pol = np.array(amp_pol)
            #if lE*lEB > 0: 
            #    print(amp_pol)
            Rmat += np.outer(amp_pol, amp_pol.conj())

    tr = np.trace(Rmat) 
    rho12 = Rmat/tr

    #print(rho)

    rho = rho12.reshape(2,2,3,2,2,3)

    rho_tt = np.einsum('ijklmk->ijlm', rho).reshape(4, 4)

    #print(rho_tt - rho_mg5)

    #print( np.trace(rho_mg5@rho_mg5), np.trace(rho_tt@rho_tt) )

    eigs_mg5, eigenvectors = np.linalg.eig(rho_mg5)
    eigs_tt, eigenvectors = np.linalg.eig(rho_tt)

    eigs_tt = np.sort(np.real(eigs_tt))[::-1]
    eigs_tt = np.clip(eigs_tt, 0, None)

    eigs_mg5 = np.sort(np.real(eigs_mg5))[::-1]
    eigs_mg5 = np.clip(eigs_mg5, 0, None)

    print( concurrence(rho_tt), concurrence(rho_mg5) )

    # print( eigs_tt )
    # print( eigs_mg5 )
    # print('--')
    #print(rho_ttb)

    #pretty_output( header, outlist )

