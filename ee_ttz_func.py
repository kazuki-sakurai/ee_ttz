import os, sys 
import numpy as np
import vector
from math import sqrt, acos, cos, sin

def vec2ar(v): return np.array([v.E, v.px, v.py, v.pz])

def get_random_momenta(rs, mt, mZ):

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

def get_momenta(rs, m12, m1, m3, cth3, cth1, ph):
    # s -> (12) + 3 

    sth1 = sqrt(1 - cth1**2)
    sth3 = sqrt(1 - cth3**2)

    M = rs
    q3 = sqrt(M**4 - 2*(m12**2 + m3**2)*M**2 + (m12**2 - m3**2)**2)/(2*M)
    E3 = (M**2 + m3**2 - m12**2)/(2*M)  
    p3 = vector.obj(px=q3*sth3, py=0, pz = q3*cth3, mass = m3 )
    p12 = vector.obj(px=-q3*sth3, py=0, pz = -q3*cth3, mass = m12 )

    q_ = sqrt( m12**2 - 4*m1**2 )/2 

    p1_ = vector.obj(px=q_*sth1*cos(ph), py=q_*sth1*sin(ph), pz = q_*cth1, mass = m1 )
    p2_ = vector.obj(px=-q_*sth1*cos(ph), py=-q_*sth1*sin(ph), pz = -q_*cth1, mass = m1 )

    bvec = vector.obj(px=p12.x/p12.e, py=p12.y/p12.e, pz=p12.z/p12.e)
    p1 = p1_.boost(bvec)
    p2 = p2_.boost(bvec)

    return p1, p2, p3

