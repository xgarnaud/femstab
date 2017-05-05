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
else:
    ffdisc = ff.EmptyFreeFEMdisc()
    Pu = None

PrintGreen('done \n')

# Create PETSC matrices
Print('Convert matrices to PETSC parallel format ... ')
Lmat = pfs.CSR2Mat(ffdisc.L)
Bmat = pfs.CSR2Mat(ffdisc.B)
Pumat = pfs.CSR2Mat(Pu)

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
hmin = comm.bcast(hmin,root=0)
dt = hmin * cfl

Print('Time stepping parameters ')
if cn:
    Print('scheme : CN')
else:
    Print('scheme : Euler')
Print('CFL    : %g'%cfl)
Print('dt     : %g'%dt)

x   = Lmat.getVecRight()
Lx  = Lmat.getVecRight()
y   = Lmat.getVecRight()
LHy = Lmat.getVecRight()

lsize,gsize = x.getSizes()
localsizes,globalsizes = Lmat.getSizes()

t1 = MPI.Wtime()

# Time stepper
TS    = pfs.TimeStepping(Lmat,Bmat,Pumat,dt)

t2 = MPI.Wtime()
Print(' CPU time to build TS object : %10.4g '%(t2-t1))

# random initial condition
v = random.rand(lsize) - .5
x.setArray(v)
norm = x.norm(); x.scale(1./norm)

v = random.rand(lsize) - .5
y.setArray(v)
norm = y.norm(); y.scale(1./norm)

Tf = 0.5

TS.setTf(Tf)

t1 = MPI.Wtime()
TS.mult(None,x,Lx)
t2 = MPI.Wtime()

Print(' CPU time to advance in time of %10.4g : %10.4g '%(Tf,t2-t1))

t1 = MPI.Wtime()
TS.multTranspose(None,y,LHy)
t2 = MPI.Wtime()

Print(' CPU time to advance in time of %10.4g : %10.4g '%(Tf,t2-t1))

dot1 = y.dot(Lx)
dot2 = LHy.dot(x)

Print('Error on the adjoint: %g'%abs(dot2 - dot1.conjugate()))

xv  = pfs.Vec2DOF(x)
Lxv = pfs.Vec2DOF(Lx)

if rank == 0:
    plt.figure()
    plt.subplot(211)
    ffdisc.PlotVar(xv,0)
    plt.subplot(212)
    ffdisc.PlotVar(Lxv,0)
    plt.show()

