from   numpy import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
from numpy.linalg import norm
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

from petsc4py import PETSc

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

tol_ev=1e-12
tol_fr=1e-12

def CSR2Mat(L):

    """
    Converts a sequential scipy sparse matrix (on process 0) to a PETSc
    Mat ('aij') matrix distributed on all processes
    input : L, scipy sparse matrix on proc 0
    output: PETSc matrix distributed on all procs
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Get the data from the sequential scipy matrix
    if rank == 0:
        if L.format == 'csr':
            L2 = L
        else:
            L2 = L.tocsr()
        Ai  = L2.indptr
        Aj  = L2.indices
        Av  = L2.data
        nnz = len(Aj)
        n,m = L2.shape
    else:
        n   = None
        m   = None
        nnz = None
        Ai  = None
        Aj  = None
        Av  = None

    # Broadcast sizes
    n   = comm.bcast(n  ,root = 0)
    m   = comm.bcast(m  ,root = 0)
    nnz = comm.bcast(nnz,root = 0)

    B = PETSc.Mat()
    B.create(comm)
    B.setSizes([n, m])
    B.setType('aij') 
    B.setFromOptions()

    # Create a vector to get the local sizes, so that preallocation can be done later
    V = PETSc.Vec()
    V.create(comm)
    V.setSizes(n)     
    V.setFromOptions()
    istart,iend = V.getOwnershipRange()
    V.destroy()

    nloc = iend - istart

    Istart = comm.gather(istart,root = 0)
    Iend   = comm.gather(iend  ,root = 0)

    if rank == 0:
        nnzloc = zeros(comm.size,'int')
        for i in range(comm.size):
            j0        = Ai[Istart[i]]
            j1        = Ai[Iend  [i]]
            nnzloc[i] = j1 - j0
    else:
        nnzloc = None

    nnzloc = comm.scatter(nnzloc,root = 0)
    
    ai = zeros(nloc+1   ,PETSc.IntType)
    aj = zeros(nnzloc+1 ,PETSc.IntType)
    av = zeros(nnzloc+1 ,PETSc.ScalarType)

    if rank == 0:        
        j0        = Ai[Istart[0]]
        j1        = Ai[Iend  [0]]
        ai[:nloc  ] = Ai[:nloc]
        aj[:nnzloc] = Aj[j0:j1]
        av[:nnzloc] = Av[j0:j1]
        
    for iproc in range(1,comm.size):
        if rank == 0:
            i0        = Istart[iproc]
            i1        = Iend  [iproc]
            j0        = Ai[i0]
            j1        = Ai[i1]
            comm.Send(Ai[i0:i1], dest=iproc, tag=77)
            comm.Send(Aj[j0:j1], dest=iproc, tag=78)
            comm.Send(Av[j0:j1], dest=iproc, tag=79)
        elif rank == iproc:
            comm.Recv(ai[:nloc  ], source=0, tag=77)
            comm.Recv(aj[:nnzloc], source=0, tag=78)
            comm.Recv(av[:nnzloc], source=0, tag=79)

    ai = ai- ai[0]
    ai[-1] = nnzloc+1
    
    B.setPreallocationCSR((ai,aj))
    B.setValuesCSR(ai,aj,av)
    B.assemble()
    
    return B

def DOF2Vec(v):
    """
    Converts a sequential vector of all degrees of freedom on process 0
    to a distributed PETSc Vec
    input : v, numpy array on proc 0
    output: PETSc Vec distributed on all procs
    """
     
    from petsc4py import PETSc
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
   
    n = len(v)
    
    x = PETSc.Vec()
    x.create(comm)
    x.setSizes(n)     
    x.setFromOptions()
    istart,iend = x.getOwnershipRange()

    nloc = iend - istart
    Istart = comm.gather(istart,root = 0)
    Iend   = comm.gather(iend  ,root = 0)

    vloc = zeros(nloc,PETSc.ScalarType)

    if rank == 0:        
        vloc[:nloc  ] = v[:nloc]
    
    for iproc in range(1,comm.size):        
        if rank == 0:
            i0        = Istart[iproc]
            i1        = Iend  [iproc]
            comm.Send(v[i0:i1], dest=iproc, tag=77)
        elif rank == iproc:
            comm.Recv(vloc, source=0, tag=77)

    x.setArray(vloc)
    
    return x

def Vec2DOF(x):
    
    """
    Converts a a distributed PETSc Vec to a sequential vector of all
    degrees of freedom on process 0
    input : x, PETSc Vec distributed on all procs
    output: numpy array on proc 0
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    vloc = x.getArray()
    n    = x.getSize()

    istart,iend = x.getOwnershipRange()

    nloc = iend - istart
    Istart = comm.gather(istart,root = 0)
    Iend   = comm.gather(iend  ,root = 0)

    if rank == 0:
        v = zeros(n,PETSc.ScalarType)
    else:
        v = None

    if rank == 0:        
        v[:nloc  ] = vloc
    
    for iproc in range(1,comm.size):        
        if rank == 0:
            i0        = Istart[iproc]
            i1        = Iend  [iproc]
            comm.Recv(v[i0:i1], source=iproc, tag=77)
        elif rank == iproc:
            comm.Send(vloc, dest=0, tag=77)

    return v


def DirectModeSLEPc(L,B,shift,nev):
    
    """
    Computes generalized eigenvectors and eigenvalues for the problem
    Lq = lambda Bq
    using SLEPc
    inputs : B,L, PETSc Mats
             shift, scalar (same on all procs). Shift parameter for 
               the shift-invert method
             nev, integer. Number of requested eigenvalues

    outputs: omega, complex array(nconv). Conputed eigenvalues
             modes, complex array(nconv,ndofs). Computed eigenvectors
             residual, real array(nconv). Actual residuals for each mode
    
    ALL OUTPUT ARE ONLY ON PROC 0 (=None on other procs)

    TO DO: compute left eigenvectors (adjoint problem)
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ev = L.getVecRight()
    Lq = ev.duplicate()
    Bq = ev.duplicate()

    ndof = ev.getSize()

    # Setup EPS
    Print("  - Setting up the EPS and the ST")    
    SI = SLEPc.ST().create()
    SI.setType(SLEPc.ST.Type.SINVERT)
    SI.setOperators(L,B)
    SI.setShift(shift)
    SI.setFromOptions()

    S = SLEPc.EPS();
    S.create(comm)
    S.setTarget(shift)
    S.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    S.setST(SI)
    S.setDimensions(nev = nev,ncv = 60)
    S.setTolerances(tol=tol_ev, max_it=100)
    S.setFromOptions()

    # Solve the EVP
    Print("  - Solving the EPS")
    S.solve()

    its = S.getIterationNumber()
    nconv = S.getConverged()
    
    if rank == 0:
        residual=zeros(nconv)
        omega=zeros(nconv,'complex')
        modes=zeros([ndof,nconv],'complex')
    else:
        residual = None
        omega    = None
        modes    = None


    for i in range(nconv):
        eigval = S.getEigenpair(i, ev)
        L.mult(ev,Lq)
        B.mult(ev,Bq)
        Bq.aypx(-eigval,Lq)
        res = Bq.norm()/ev.norm()
        v      = Vec2DOF(ev)

        if rank == 0:
            omega[i]  = eigval/(-1j)
            modes[:,i]= v
            residual[i] = res

    return omega,modes,residual



class OptimalPerturbations(object):

    """ 
    Shell matrix for optimal perturbations computations with PETSc
    """

    def __init__(self,L,B,Pu,Q,Qr,dt,CN=True):
        """
        Parameters of the optimal perturbations computation:
        L, PETSC Mat, discretization matrix
        B, PETSC Mat, mass matrix
        Q, PETSC Mat, norm matrix
        Pu, PETSC Mat, converting velocity only vectors to full DOFs vector
        Qr, PETSC Mat, norm matrix on the velovity-only space
           (should be invertible)
           Qr = Pu^T Q Pu
        dt, real, time step
        CN, bool, indicating whether Crank-Nicholson or backwards Euler is used

        The object contains two KSP solvers, one for implicit time
        stepping and one for the inversion of the norm matrix, that are initialized
        when the object is initialized

        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.L  = L
        self.B  = B
        self.Pu = Pu
        self.Q  = Q
        self.Qr = Qr

        if CN:
            Print('  - temporal discretization: Crank-Nicholson')
        else:
            Print('  - temporal discretization: Backwards Euler')

        Print('  - Setting the linear solvers')

        self.OP = L.duplicate(copy=True)
        if CN:
            self.OP.scale(-dt/2.)
        else:
            self.OP.scale(-dt)
        self.OP.axpy(1.,B)

        # Linear solver for time stepping
        self.ksp = PETSc.KSP()
        self.ksp.create(comm)
        self.ksp.setType('preonly')
        self.ksp.getPC().setType('lu')
        self.ksp.setOperators(self.OP)
        self.ksp.setFromOptions()
        self.ksp.getPC().setUp()

        # self.ksp.view()

        self.ksp2 = PETSc.KSP()
        self.ksp2.create(comm)
        self.ksp2.setType('preonly')
        self.ksp2.getPC().setType('lu')
        # self.ksp2.getPC().setType('cholesky')
        self.ksp2.setOperators(self.Qr)
        self.ksp2.setFromOptions()
        self.ksp2.getPC().setUp()

        # self.ksp2.view()

        self.dt  = dt
        self.nt  = 1
        self.CN  = CN

        self.tmp  = L.getVecRight()
        self.tmp2 = L.getVecRight()
        self.tmpr = Qr.getVecRight()
        if self.CN:
            self.tmp3 = L.getVecRight()
        
    def setTf(self,Tf):
        Tf = float(Tf)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.nt = int(Tf/self.dt)

        Print('  -    nt = %d'%self.nt)
        Print('  -    Tf = %f'%(self.dt*self.nt))
    
    def mult(self,A,x,y):
        self.Pu.mult(x,self.tmp2)
        for it in range(self.nt):
            self.B.mult(self.tmp2,self.tmp)
            if self.CN:# and it>0:
                self.L.mult(self.tmp2,self.tmp3)
                # Mass conservation is imposed at time n+1
                self.Pu.multTranspose(self.tmp3,self.tmpr) 
                self.Pu.mult(self.tmpr,self.tmp2) 
                self.tmp.axpy(self.dt/2.,self.tmp2)
            self.ksp.solve(self.tmp,self.tmp2)
        self.Q.mult(self.tmp2,self.tmp)
        for it in range(self.nt):
            self.tmp.conjugate()
            self.ksp.solveTranspose(self.tmp,self.tmp2)
            self.tmp2.conjugate()
            self.B.multTranspose(self.tmp2,self.tmp)
            if self.CN:# and it<(self.nt-1):
                self.Pu.multTranspose(self.tmp2,self.tmpr) 
                self.Pu.mult(self.tmpr,self.tmp3) 
                self.tmp3.conjugate()
                self.L.multTranspose(self.tmp3,self.tmp2)
                self.tmp2.conjugate()
                self.tmp.axpy(self.dt/2.,self.tmp2)

        self.Pu.multTranspose(self.tmp,self.tmpr)
        self.ksp2.solve(self.tmpr,y)

    def PropagateIC(self,x):
        self.Pu.mult(x,self.tmp2)
        for it in range(self.nt):
            self.B.mult(self.tmp2,self.tmp)
            if self.CN:# and it>0:
                self.L.mult(self.tmp2,self.tmp3)
                # Mass conservation is imposed at time n+1
                self.Pu.multTranspose(self.tmp3,self.tmpr) 
                self.Pu.mult(self.tmpr,self.tmp2) 
                self.tmp.axpy(self.dt/2.,self.tmp2)
            self.ksp.solve(self.tmp,self.tmp2)
        return self.tmp2

class TimeStepping(object):
    """ 
    Shell matrix for time stepping with PETSc
    """

    def __init__(self,L,B,Pu,dt,CN=True):
        """
        Parameters of the time stepping:
        L, PETSC Mat, discretization matrix
        B, PETSC Mat, mass matrix
        Pu, PETSC Mat, converting velocity only vectors to full DOFs vector.
           Can be None for Euler
        dt, real, time step
        CN, bool, indicating whether Crank-Nicholson or backwards Euler is used

        The object contains one KSP solver for implicit time
        stepping that is initialized when the object is initialized

        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.L = L
        self.B = B
        self.Pu = Pu

        if CN:
            Print('  - temporal discretization: Crank-Nicholson')
        else:
            Print('  - temporal discretization: Backwards Euler')

        Print('  - Setting the linear solver')

        self.OP = L.duplicate(copy=True)

        if CN:
            self.OP.scale(-dt/2.)
        else:
            self.OP.scale(-dt)
        self.OP.axpy(1.,B)

        # Linear solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm)
        self.ksp.setType('preonly')
        self.ksp.getPC().setType('lu')
        
        self.ksp.setOperators(self.OP)
        self.ksp.setFromOptions()
        self.ksp.getPC().setUp()

        self.dt = dt
        self.nt = 1
        self.CN = CN

        self.tmp  = L.getVecRight()
        if self.CN:
            self.tmp2 = L.getVecRight()
            self.tmpr = Pu.getVecRight()

    def setTf(self,Tf):
        
        Tf = float(Tf)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.nt = int(Tf/self.dt)

        Print('  - nt = %d'%self.nt)
        Print('  - Tf = %f'%(self.dt*self.nt))

    def mult(self,A,x,y):
        x.copy(y)
        for it in range(self.nt):
            self.B.mult(y,self.tmp)
            if self.CN:# and it>0:
                self.L.mult(y,self.tmp2)
                self.Pu.multTranspose(self.tmp2,self.tmpr) 
                self.Pu.mult(self.tmpr,self.tmp2) 
                self.tmp.axpy(self.dt/2.,self.tmp2)
            self.ksp.solve(self.tmp,y)

    def multTranspose(self,A,x,y):
        x.copy(y)
        for it in range(self.nt):
            y.conjugate()
            self.ksp.solveTranspose(y,self.tmp)
            self.tmp.conjugate()
            self.B.multTranspose(self.tmp,y)
            if self.CN:# and it<(self.nt-1):
                self.Pu.multTranspose(self.tmp,self.tmpr) 
                self.Pu.mult(self.tmpr,self.tmp) 
                self.tmp.conjugate()
                self.L.multTranspose(self.tmp,self.tmp2)
                self.tmp2.conjugate()
                y.axpy(self.dt/2.,self.tmp2)
                


def OptimalPerturbationsSLEPc(TG,nev):
    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ev = TG.getVecRight()

    ndof = ev.getSize()

    # Setup EPS
    Print('  - Setting the EPS')
    
    S = SLEPc.EPS();
    S.create(comm)
    S.setOperators(TG)
    S.setDimensions(nev = nev,ncv = 6)
    S.setTolerances(tol=1e-6, max_it=100)
    S.setFromOptions()

    # Solve the EVP
    Print("  - Solving the EPS")
    S.solve()

    its = S.getIterationNumber()
    nconv = S.getConverged()
    
    if rank == 0:
        eigvals=zeros(nconv,'complex')
        modes  =zeros([ndof,nconv],'complex')
    else:
        eigvals    = None
        modes    = None


    for i in range(nconv):
        eigval = S.getEigenpair(i, ev)
        v      = Vec2DOF(ev)

        if rank == 0:
            eigvals[i]  = eigval
            modes[:,i]= v

    return eigvals,modes

class OptimalForcings(object):

    """ 
    Shell matrix for optimal perturbations computations with PETSc
    """

    def __init__(self,L,B,B2,Pu,Q,Qr,omega):
        """
        Parameters of the optimal perturbations computation:
        L, PETSC Mat, discretization matrix
        B, PETSC Mat, mass matrix
        B2, PETSC Mat, 'mass' matrix corresponding to the forcing. This 
           can be used e.g. to restrict the forcing region
        Q, PETSC Mat, norm matrix
        Pu, PETSC Mat, converting velocity only vectors to full DOFs vector
        Qr, PETSC Mat, norm matrix on the velovity-only space
           (should be invertible)
           Qr = Pu^T Q Pu
        omega, real, frequency

        The object contains two KSP solvers, one for resolvent computation
        and one for the inversion of the norm matrix, that are initialized
        when the object is initialized

        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.L  = L
        self.B  = B
        self.B2 = B2
        self.Pu = Pu
        self.Q  = Q
        self.Qr = Qr

        Print('  - Setting the linear solver')

        self.OP = B.duplicate(copy=True)
        self.OP.scale(1j*omega)
        self.OP.axpy(1.,L)

        # Linear solver for time stepping
        self.ksp = PETSc.KSP()
        self.ksp.create(comm)
        self.ksp.setOptionsPrefix('OP_')
        self.ksp.setType('preonly')
        self.ksp.getPC().setType('lu')
        self.ksp.setOperators(self.OP)
        self.ksp.setFromOptions()
        self.ksp.getPC().setUp()
        
        # self.ksp.view()

        self.ksp.setOptionsPrefix('Q_')
        self.ksp2 = PETSc.KSP()
        self.ksp2.create(comm)
        self.ksp2.setType('cg')
        self.ksp2.getPC().setType('ilu')
        self.ksp2.setOperators(self.Qr)
        self.ksp2.setFromOptions()
        self.ksp2.getPC().setUp()

        # self.ksp2.view()

        self.tmp  = L.getVecRight()
        self.tmp2 = L.getVecRight()
        self.tmpr = Qr.getVecRight()
    
    def mult(self,A,x,y):
        self.Pu.mult(x,self.tmp2)
        self.B2.mult(self.tmp2,self.tmp)
        self.ksp.solve(self.tmp,self.tmp2)
        self.Q.mult(self.tmp2,self.tmp)
        self.tmp.conjugate()
        self.ksp.solveTranspose(self.tmp,self.tmp2)
        self.tmp2.conjugate()
        self.B2.multTranspose(self.tmp2,self.tmp)
        self.Pu.multTranspose(self.tmp,self.tmpr)
        self.ksp2.solve(self.tmpr,y)

def OptimalForcingsSLEPc(FR,shell,nev):
    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    f   = FR.getVecRight()
    q   = shell.L.getVecRight()
    tmp = shell.L.getVecRight()

    ndof_f = f.getSize()
    ndof_q = q.getSize()

    # Setup EPS
    Print('  - Setting the EPS')
    
    S = SLEPc.EPS();
    S.create(comm)
    S.setOperators(FR )
    S.setDimensions(nev = nev,ncv = 10)
    S.setTolerances(tol=1e-6, max_it=100)
    S.setFromOptions()

    # Solve the EVP
    Print("  - Solving the EPS")
    S.solve()

    its = S.getIterationNumber()
    nconv = S.getConverged()
    
    if rank == 0:
        eigvals = zeros(nconv,'complex')
        fs      = zeros([ndof_f,nconv],'complex')
        qs      = zeros([ndof_q,nconv],'complex')
    else:
        eigvals = None
        fs      = None
        qs      = None


    for i in range(nconv):
        eigval = S.getEigenpair(i, f)
        shell.Pu.mult(f,q)
        shell.B2.mult(q,tmp)
        shell.ksp.solve(tmp,q)
        q.scale(-1.0)

        vf      = Vec2DOF(f)
        vq      = Vec2DOF(q)
        if rank == 0:
            eigvals[i]  = eigval
            fs[:,i]     = vf
            qs[:,i]     = vq
            

    return eigvals,fs,qs
