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

opts.setValue('st_ksp_type','preonly')
opts.setValue('st_pc_type','lu')
opts.setValue('st_pc_factor_mat_solver_package','mumps')

# Parallel info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get the directory where ff++ data is
di = opts.getString('dir')
PrintRed('Running tests in '+ di + '\n')

PrintRed("Testing the standand interface for B q' = L q ... \n")

# Build FreeFEMdisc from .dat files
Print('Loading dat files ... ')
if rank == 0:
    ffdisc = ff.FreeFEMdisc(di+'/lin/')
else:
    ffdisc = ff.EmptyFreeFEMdisc()

PrintGreen('done \n')

# Save as HDF5
Print('Saving as .h5 file ... ')
if rank == 0:
    ffdisc.SaveHDF5(di+'/ffdata.h5')
PrintGreen('done \n')

del ffdisc

# Loading from .h5 file
Print('Loading from .h5 file ... ')
if rank == 0:
    ffdisc = ff.FreeFEMdisc(di+'/ffdata.h5')
else:
    ffdisc = ff.EmptyFreeFEMdisc()
PrintGreen('done \n')

# Create PETSC matrices
Print('Convert matrices to PETSC parallel format ... ')
Lmat = pfs.CSR2Mat(ffdisc.L)
Bmat = pfs.CSR2Mat(ffdisc.B)
PrintGreen('done \n')

# Clear some space in memory
del ffdisc.L,ffdisc.B

# Compute modes using SLEPc
Print('Compute eigenmodes ... ')

iomega = linspace(0,2.,5)
if rank == 0:
    plt.figure()
    f = open(di+'/spectrum.dat','w')

for omega0 in iomega:
    shift = -1j*omega0
    Print(' shift : (%+10.4g,%+10.4g) '%(shift.real,shift.imag))
    nev   = 20

    t1 = MPI.Wtime()
    omegas,modes,residuals = pfs.DirectModeSLEPc(Lmat,Bmat,shift,nev)
    t2 = MPI.Wtime()
    
    Print(' CPU time : %10.4g '%(t2-t1))
    if rank == 0:
        plt.scatter(omegas.real,omegas.imag)
        for om in omegas:
            f.write('%+10.4g %+10.4g\n'%(om.real,om.imag))

PrintGreen('done \n')

if rank == 0:
    plt.show()
    f.close()


