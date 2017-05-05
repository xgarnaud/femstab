# load freefem tools
import freefem_bodyforce as ff_body

# load libraries
from numpy import *
import matplotlib.pyplot as plt
import h5py as h5

di = '../../step'

ffdisc = ff_body.FreeFEMdisc_bodyforce(di+'/ffdata.h5')
h5f    = h5.File(di+"/results.h5","r")

Pu,nu = ffdisc.getPu([0,1])

idx = 1
while "freq_%05d"%idx in h5f:
    f = h5f["freq_%05d/forcing"%idx].value
    q = h5f["freq_%05d/flow"   %idx].value

    f2 = Pu*f

    grp = h5f["freq_%05d"%idx]
    w = grp.attrs['omega']
    G = grp.attrs['gain']
    print idx,w,G
    idx +=1
    plt.figure()
    plt.subplot(2,1,1)
    ffdisc.PlotVar(q,0)
    plt.subplot(2,1,2)
    ffdisc.PlotVar(f2,0)
    nq = vdot(q,ffdisc.Q*q)
    nf = vdot(f2,ffdisc.Q*f2)
    print nq / nf

plt.show()
