# initialize PETSC & SLEPC
import sys, petsc4py,slepc4py
slepc4py.init(sys.argv)

# load freefem tools
import freefem as ff
import freefem_bodyforce as ff_body
import freefem_boundaryforce as ff_boundary


# load libraries
from numpy import *
import matplotlib.pyplot as plt
import h5py as h5
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

# load functions for stability analysis
import parfemstab as pfs
from parfemstab import Print,PrintRed,PrintGreen

# Set MUMPS as the linear solver
opts = PETSc.Options()

# Parallel info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get the directory where ff++ data is
di = opts.getString('dir')

# Build FreeFEMdisc from .dat files
Print('Loading discretization files ... ')
if rank == 0:
    try:
        ffdisc = ff.FreeFEMdisc(di+'/ffdata.h5')
        Print('data loaded from .h5 file ... ')
    except IOError:
        ffdisc = ff.FreeFEMdisc(di+'/lin/')
        ffdisc.SaveHDF5(di+'/ffdata.h5')
        Print('data loaded from .dat file ... ')
    # Get the projection operator on velocity DOFs 
    Print('NO MASK')
    Pu = ffdisc.getPu(iu=[0,1])
    ntot,nu = Pu.shape
    Print('total # of DOF : %d, number of DOFs in the reduced vector: %d '%(ntot,nu))
    rvec = ones(nu)
    tvec = Pu*rvec
    for i in range(ffdisc.nvar):
        plt.figure()
        ffdisc.PlotVar(tvec,i,contours=linspace(-.01,1.01,10))
        plt.colorbar()

    def mask(x,y):
        if (x > 1) and (x < 10) and abs(y) > .5:
            return True
        else:
            return False

    Print('NO MASK')
    Pu = ffdisc.getPu(iu=[0,1],mask= mask)
    ntot,nu = Pu.shape
    Print('total # of DOF : %d, number of DOFs in the reduced vector: %d '%(ntot,nu))
    rvec = ones(nu)
    tvec = Pu*rvec
    for i in range(ffdisc.nvar):
        plt.figure()
        ffdisc.PlotVar(tvec,i,contours=linspace(-.01,1.01,10))
        plt.colorbar()


    plt.show()

