# initialize PETSC & SLEPC
import sys, petsc4py,slepc4py
slepc4py.init(sys.argv)

# load freefem tools
import freefem_bodyforce as ff_body

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

# Parallel info & print
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get the directory where ff++ data is
di = opts.getString('dir')
PrintRed('Running tests in '+ di + '\n')

Print("Testing the interface for B q' = L q + B_2 f... \n")

# Build FreeFEMdisc from .dat files
Print('Loading discretization files ... ')
if rank == 0:
    try:
        ffdisc = ff_body.FreeFEMdisc_bodyforce(di+'/ffdata.h5')
        Print('data loaded from .h5 file ... ')
    except IOError:
        ffdisc = ff_body.FreeFEMdisc_bodyforce(di+'/lin/')
        ffdisc.SaveHDF5(di+'/ffdata.h5')
        Print('data loaded from .dat file ... ')
    # Get the projection operator on velocity DOFs 
    Pu = ffdisc.getPu(iu=[0,1])
    Qr    = Pu.transpose()*ffdisc.Q*Pu
    h5f   = h5.File(di+"/results.h5","w")
else:
    ffdisc = ff_body.EmptyFreeFEMdisc()
    Pu     = None
    Qr     = None

PrintGreen('done \n')

# Create PETSC matrices
Print('Convert matrices to PETSC parallel format ... ')
Lmat  = pfs.CSR2Mat(ffdisc.L)
Bmat  = pfs.CSR2Mat(ffdisc.B)
B2mat = pfs.CSR2Mat(ffdisc.B2)
Pumat = pfs.CSR2Mat(Pu)
Qmat  = pfs.CSR2Mat(ffdisc.Q)
Qrmat = pfs.CSR2Mat(Qr)
PrintGreen('done \n')

# Clear some space in memory
del ffdisc.L,ffdisc.B,ffdisc.B2,ffdisc.Q,Qr

# Compute optimal forcings
Print('Compute optimal forcings using SLEPC ... ')
omegas = linspace(0.05,2,10)
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
    FR.setUp()
    t2 = MPI.Wtime()
    Print(' CPU time to build FR object : %10.4g '%(t2-t1))

    # Compute optimal perturbations
    gains,fs,qs = pfs.OptimalForcingsSLEPc(FR,shell,1)
    if rank == 0:
        G[idx] = gains[0].real; idx +=1
        grp    = h5f.create_group("freq_%05d"%idx)
        dset = grp.create_dataset('forcing',data=fs[:,0])
        dset = grp.create_dataset('flow'   ,data=qs[:,0])
        grp.attrs['omega'] = omega
        grp.attrs['gain']  = G[idx-1]
        h5f.flush()
                   
    t1 = MPI.Wtime()
    Print(' CPU time to solve the EVP : %10.4g '%(t1-t2))

if rank == 0:
    h5f.close()

plt.figure()
plt.plot(omegas,G)
data = array([[0.04444444444444448,179.48133558103873],
              [0.05714285714285714,205.06762710686837],
              [0.13650793650793647,566.5202447230035 ],
              [0.14603174603174598,629.5574531349081 ],
              [0.22539682539682535,1729.5858443874629],
              [0.23809523809523814,1998.2190123622177],
              [0.4554112554112554,7492.005210421082  ],
              [0.4874779541446206,7501.255814641612  ],
              [0.6095238095238096,5613.014207297659  ],
              [0.6857142857142857,4022.5514510637777 ],
              [0.8888888888888888,1385.0935270147158 ],
              [0.9904761904761905,803.793878772087   ],
              [1.0730158730158732,532.9520080701366  ],
              [1.1746031746031746,319.75927541335875 ],
              [1.276190476190476,205.06762710686837  ],
              [1.358730158730159,146.96116035346333  ],
              [1.5555555555555556,75.47680448057177  ],
              [1.6444444444444444,57.81730945850528  ],
              [1.9301587301587306,29.36605846971029  ]])
plt.plot(data[:,0],data[:,1],'o')

plt.show()
PrintGreen('done \n')


