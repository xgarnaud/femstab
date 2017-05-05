# initialize PETSC & SLEPC
import sys, petsc4py,slepc4py
slepc4py.init(sys.argv)

# load freefem tools
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

# Set MUMPS as the linear solver
opts = PETSc.Options()
opts.setValue('st_ksp_type','preonly')
opts.setValue('st_pc_type','lu')
opts.setValue('st_pc_factor_mat_solver_package','mumps')

# Parallel info & print
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
Print = PETSc.Sys.Print

# Print in color
REDBOLD_   = "\033[1;31m"
RED_       = "\033[31m"
GREEN_     = "\033[32m"
CYAN_      = "\033[36m"
YELLOW_    = "\033[33m"
CLRFORMAT_ = "\033[0m"

def PrintRed(s):
    Print(RED_,s,CLRFORMAT_)
def PrintGreen(s):
    Print(GREEN_,s,CLRFORMAT_)

# Get the directory where ff++ data is
di = opts.getString('dir')
PrintRed('Running tests in '+ di + '\n')

Print("Testing the interface for B q' = L q with boundary forcing... \n")

# Build FreeFEMdisc from .dat files
Print('Loading dat files ... ')
if rank == 0:
    ffdisc = ff_boundary.FreeFEMdisc_boundaryforce(di+'/lin/')
else:
    ffdisc = ff_boundary.EmptyFreeFEMdisc()

PrintGreen('done \n')

# Save as HDF5
Print('Saving as .h5 file ... ')
if rank == 0:
    ffdisc.SaveHDF5(di+'/ffdata.h5')
PrintGreen('done \n')

# Loading from .h5 file
del ffdisc
Print('Loading from .h5 file ... ')
if rank == 0:
    ffdisc = ff_boundary.FreeFEMdisc_boundaryforce(di+'/ffdata.h5')
else:
    ffdisc = ff_boundary.EmptyFreeFEMdisc()

PrintGreen('done \n')

# Create PETSC matrices
Print('Convert matrices to PETSC parallel format ... ')
Lmat  = pfs.CSR2Mat(ffdisc.L)
Bmat  = pfs.CSR2Mat(ffdisc.B)
B2mat = pfs.CSR2Mat(ffdisc.B2)
Qmat  = pfs.CSR2Mat(ffdisc.Q)
Pumat = pfs.CSR2Mat(ffdisc.Pf)
Qrmat = pfs.CSR2Mat(ffdisc.Qf)
PrintGreen('done \n')

# Clear some space in memory
del ffdisc.L,ffdisc.B,ffdisc.B2,ffdisc.Q,ffdisc.Pf,ffdisc.Qf

# Compute optimal forcings
Print('Compute optimal forcings using SLEPC ... ')
omegas = [0.5]
G = zeros(len(omegas)); idx = 0
for iomega in range(len(omegas)):
    omega = omegas[iomega]
    Print('  omega = %f'%omega)
    # Set up the shell matrix and compute the factorizations
    t1 = MPI.Wtime()
    shell = pfs.OptimalForcings(Lmat,Bmat,B2mat,Pumat,Qmat,Qrmat,omega)
    localsizes,globalsizes = Qrmat.getSizes()
    FR    = PETSc.Mat().create(comm)
    FR.setSizes(globalsizes)
    FR.setType('python')
    FR.setPythonContext(shell)
    t2 = MPI.Wtime()
    Print(' CPU time to build FR object : %10.4g '%(t2-t1))

    # Compute optimal perturbations
    gains,fs,qs = pfs.OptimalForcingsSLEPc(FR,shell,1)
    G[idx] = sqrt(gains[0].real); idx +=1
    t1 = MPI.Wtime()
    Print(' CPU time to solve the EVP : %10.4g '%(t1-t2))

    # Plot example
    if rank == 0:
        f = fs[:,0]
        q = qs[:,0]
        
        plt.figure()
        plt.subplot(221)
        xx,ff = ffdisc.PlotBoundaryVar(f,0,ax='y',add=((0,0),(1,0)))
        yy = xx; xx = -10 + 0*yy
        bval = ffdisc.GetValue(q,0,xx,yy)
        plt.plot(bval,yy,'+')
        mx = abs(ff.real).max()
        plt.subplot(222)
        ffdisc.PlotVar(q,0)
        plt.colorbar()

        plt.subplot(223)
        xx,ff = ffdisc.PlotBoundaryVar(f,1,ax='y',add=((0,0),(1,0)))
        yy = xx; xx = -10 + 0*yy
        bval = ffdisc.GetValue(q,1,xx,yy)
        plt.plot(bval,yy,'+')
        plt.subplot(224)
        ffdisc.PlotVar(q,1)
        plt.colorbar()
        plt.show()

plt.show()
PrintGreen('done \n')


