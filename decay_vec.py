#! /usr/bin/env python3

import os, sys 
import numpy as np
import vector
from math import sqrt, atan, acos, pi 

def Decay_2B(p0, m1, m2):   

    m0 = p0.mass

    m0sq = m0**2
    m1sq = m1**2
    m2sq = m2**2

    s0 = m0**2
    beta_bar0 = 1. - 2.*(m1**2 + m2**2)/s0 + (m1**2 - m2**2)**2/s0**2
    beta_bar = sqrt(beta_bar0)

    Pini_sq = ( m0sq**2 - 2.*m0sq*(m1sq + m2sq) + (m1sq - m2sq)**2 )/(4. * m0sq);
    Pini = sqrt( Pini_sq )

    p1 = vector.obj(px=0, py=0, pz = -Pini, mass = m1 )
    p2 = vector.obj(px=0, py=0, pz =  Pini, mass = m2 )

    phi = np.random.uniform(-pi, pi)  
    cth = np.random.uniform(-1., 1.)  
    theta = acos(cth)  

    p1 = p1.rotateY(theta)
    p1 = p1.rotateZ(phi)
    p2 = p2.rotateY(theta)
    p2 = p2.rotateZ(phi)

    bvec = vector.obj(px=p0.x/p0.e, py=p0.y/p0.e, pz=p0.z/p0.e)
    p1 = p1.boost(bvec)
    p2 = p2.boost(bvec)

    #print( p1.x, p1.y, p1.z )
    # print( p2.x, p2.y, p2.z )
    # print( p1.mass, p2.mass )
    # print( (p1+p2).mass )

    return p1, p2, beta_bar 


def Decay_3B(p0, m1, m2, m3):   

    if p0.mass < m1 + m2 + m3:
        print("Inconsistent Mass Hierarchy")
        print("m0, m1, m2, m3 =", p0.mass, m1, m2, m3 )
        exit()
    mXmin = m1 + m2
    mXmax = p0.mass - m3

    bool_beta = True
    while bool_beta:
        mXsq = np.random.uniform(mXmin**2, mXmax**2)
        mX = sqrt(mXsq)
        p3, pX, beta_bar1 = Decay_2B(p0, m3, mX)
        p2, p1, beta_bar2 = Decay_2B(pX, m2, m1)
        ran = np.random.uniform()
        if ran < beta_bar1 * beta_bar2: bool_beta = False

    # p1 = vector.obj(px=p1.x, py=p1.y, pz=p1.z, mass=m1)
    # p2 = vector.obj(px=p2.x, py=p2.y, pz=p2.z, mass=m2)
    # p3 = vector.obj(px=p3.x, py=p3.y, pz=p3.z, mass=m3)

    return p1, p2, p3


def Decay_2B_2B(p0, mX, m1, m2, m3):
    # sequential 2 body decays p0 -> p3 + pX, pX -> p2 + p1
    # Generate mX
    if p0.mass < m1 + m2 + m3:
        print ("Inconsistent Mass Hierarchy")
        print ("m0, m1, m2, m3 =", p0.mass, m1, m2, m3 )
        exit()

    p3, pX, beta_bar1 = Decay_2B(p0, m3, mX)
    p2, p1, beta_bar2 = Decay_2B(pX, m2, m1)

    # p1 = vector.obj(px=p1.x, py=p1.y, pz=p1.z, mass=m1)
    # p2 = vector.obj(px=p2.x, py=p2.y, pz=p2.z, mass=m2)
    # p3 = vector.obj(px=p3.x, py=p3.y, pz=p3.z, mass=m3)

    return p1, p2, p3

############################
############################

# mode = 'top'
# #mode = 'RPVg'

# n_events = 10**3

# mtop, mW = 173., 80.1
# mgl = 173.

# m1, m2, m3 = 0, 0, 0

# for iev in range(n_events):

#     #print(iev)

#     if mode == 'top':
#         p0 = vector.obj(px=0, py=0, pz=0, mass = mtop )
#         p1, p2, p3 = Decay_2B_2B(p0, mW, m1, m2, m3)
#     if mode == 'RPVg':
#         p0 = vector.obj(px=0, py=0, pz=0, mass = mgl )
#         p1, p2, p3 = Decay_3B(p0, m1, m2, m3)

#     line = '{} {} {}  '.format( p1.px, p1.py, p1.pz )
#     line += '{} {} {}  '.format( p2.px, p2.py, p2.pz )
#     line += '{} {} {}'.format( p3.px, p3.py, p3.pz )

#     #print( (p1+p2+p3).mass, (p2+p3).mass, (p1+p2).mass )

#     print(line)