# initialize PETSC & SLEPC
import sys, petsc4py,slepc4py
slepc4py.init(sys.argv)

# load freefem tools
import freefem as ff

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

opts.setValue('ksp_type','preonly')
opts.setValue('pc_type','lu')
opts.setValue('pc_factor_mat_solver_package','mumps')

# Parallel info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get the directory where ff++ data is
di = opts.getString('dir')
PrintRed('Running tests in '+ di + '\n')

PrintRed("Testing time stepping routines ... \n")

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
    Pu = ffdisc.getPu(iu=[0,1])
    Qr    = Pu.transpose()*ffdisc.Q*Pu
else:
    ffdisc = ff.EmptyFreeFEMdisc()
    Pu = None
    Qr = None

PrintGreen('done \n')

# Create PETSC matrices
Print('Convert matrices to PETSC parallel format ... ')
Lmat = pfs.CSR2Mat(ffdisc.L)
Bmat = pfs.CSR2Mat(ffdisc.B)
Pumat = pfs.CSR2Mat(Pu)
Qmat  = pfs.CSR2Mat(ffdisc.Q)
Qrmat = pfs.CSR2Mat(Qr)

PrintGreen('done \n')

# Clear some space in memory
del ffdisc.L,ffdisc.B

# Compute modes using SLEPc
Print('Perform time stepping ... \n')

# Set the time step and scheme
cfl = opts.getReal('cfl',10)
cn  = opts.getBool('cn',False)
if rank == 0:
    hmin = ffdisc.GetHmin()
else:
    hmin = 1.

hmin = comm.bcast(hmin  ,root = 0)
dt = hmin * cfl

Print('Time stepping parameters ')
if cn:
    Print('scheme : CN')
else:
    Print('scheme : Euler')
Print('CFL    : %g'%cfl)
Print('dt     : %g'%dt)

localsizes,globalsizes = Lmat.getSizes()

# Set up the shell matrix and compute the factorizations
t1 = MPI.Wtime()
shell = pfs.OptimalPerturbations(Lmat,Bmat,Pumat,Qmat,Qrmat,dt,CN=cn)
localsizes,globalsizes = Qrmat.getSizes()
TG    = PETSc.Mat().create(comm)
TG.setSizes(globalsizes)
TG.setType('python')
TG.setPythonContext(shell)
TG.setUp()
t2 = MPI.Wtime()
Print(' CPU time to build TG object : %10.4g '%(t2-t1))

t1 = MPI.Wtime()

Tfs = [2]
iev = 0
for itf in range(len(Tfs)):
    shell.setTf(Tfs[itf])
    gains,optperts = pfs.OptimalPerturbationsSLEPc(TG,1)
    
    if rank == 0:
        nconv = len(gains)
        for i in range(nconv):
            Print(' gain : %g '%(sqrt(gains[i]).real))
            plt.figure()
            ffdisc.PlotVar(Pu*optperts[:,i],0)
            plt.show()
        

t2 = MPI.Wtime()
Print(' CPU time  : %10.4g '%(t2-t1))

plt.show()
