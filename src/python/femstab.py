
from   numpy import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as splin
# import scipy.sparse.linalg.eigen.arpack as arpie
# import scipy.io as io
# import progressbar as pbar
# import matplotlib.tri as tri
# import freefem as ff
from numpy.linalg import norm
from scipy.sparse.linalg import eigs, ArpackNoConvergence

tol_ev=1e-10
tol_fr=1e-12

def Pu(ffdisc,iu):
    
    # Define the matrix of the projection on the velocity space
    # Can be made more efficient...
    ival = []
    jval = []
    dval = []
    nu    = 0
    for ivar in iu:
        for i in range(ffdisc.n[ivar]):
            ival.append(ffdisc.idof[ivar][i])
            jval.append(nu + i);
            dval.append(1.)
        nu += ffdisc.n[ivar]

    dcoo  = array(dval,'complex')
    ijcoo = [array(ival,'int'),array(jval,'int')]

    # Create COO matrix
    Pu    = sp.coo_matrix((dcoo,ijcoo),shape=(ffdisc.ndof,nu))
    # Convert to CSR format
    Pu    = Pu.tocsc()

    return Pu,nu

def FR(ffdisc,omega,Pu,nu,nev):


    # Assemble operators
    Q     = Pu.transpose()*ffdisc.Q*Pu
    Q     = Q.tocsc()
    print 'Build LU decomposition of Q'
    LUQ   = splin.splu(Q, permc_spec=3)
    OP    = ffdisc.L+1j*omega*ffdisc.B
    OP    = OP.tocsc()
    print 'Build LU decomposition of (L+iwB)'
    LU    = splin.splu(OP, permc_spec=3)
    print 'done'
    def op(x):
        y = Pu*x
        z = ffdisc.B2*y
        y = LU.solve(z)
        z = ffdisc.Q*y
        y = LU.solve(z,trans='H')
        z = ffdisc.B2.transpose()*y
        y = Pu.transpose()*z
        w = LUQ.solve(y,trans='H')
        
        return w

    SOP       = splin.LinearOperator((nu,nu),matvec=op,dtype='complex')

    try:
        w,v       = splin.eigs(SOP, k=nev, M=None, sigma=None, which='LM', ncv=20, maxiter=100, tol=tol_fr, return_eigenvectors=True)
    except ArpackNoConvergence,err:
        w = err.eigenvalues
        v = err.eigenvectors
        print 'not fully converged'

    nconv     = size(w)

    sigma = sqrt(w.real)
    f     = zeros([ffdisc.ndof,nconv],'complex')
    q     = zeros([ffdisc.ndof,nconv],'complex')

    for k in range(nconv):
        f[:,k] = Pu*v[:,k]
        z      = ffdisc.B2*f[:,k]
        q[:,k] = -LU.solve(z)

    return sigma,f,q

def DirectMode(ffdisc,shift,nev):

    OP=ffdisc.L-shift*ffdisc.B
    OP= OP.tocsc()
    print 'Build LU decomposition of (L-sB)'

    LU=splin.splu(OP, permc_spec=3)
    print 'done.'
    def op(x):
        r=ffdisc.B*x
        z=LU.solve(r)
        return z

    print 'define SOP'
    SOP=splin.LinearOperator((ffdisc.ndof,ffdisc.ndof),matvec=op,dtype='complex')
    print 'done.'

    # Compute modes
    print 'Calling eigs'
    try:
        w,v=splin.eigs(SOP, k=nev, M=None, sigma=None, which='LM', v0=None, ncv=60, maxiter=100, tol=tol_ev)
        print 'done.'
    except ArpackNoConvergence,err:
        w = err.eigenvalues
        v = err.eigenvectors
        print 'not fully converged'

    nconv=size(w)
    omega=zeros(nconv,'complex')
    modes=zeros([ffdisc.ndof,nconv],'complex')

    for i in range(nconv):
        omega[i]=(1./w[i]+shift)/(-1j)
        modes[:,i]=v[:,i]
        
    return omega,modes

def TS(ffdisc,dt,tf,q0):

    OP = ffdisc.B - dt*ffdisc.L
    OP= OP.tocsc()
    print 'Build LU decomposition of (B - L/dt)'

    LU=splin.splu(OP, permc_spec=3)
    print 'done.'

    nt = floor(tf/dt) +1
    dt = tf / nt

    q1 = q0.copy()
    for i in range(nt):
        print i,'/',nt
        t1 = -ffdisc.B*q1
        q1 = LU.solve(t1)
        
    return q1
    
def CSR2Mat(L):

    from petsc4py import PETSc

    if L.format == 'csr':
        L2 = L
    else:
        L2 = L.tocsr()


    B = PETSc.Mat();
    B.createAIJ(L2.shape,csr = (L2.indptr,
                                L2.indices,
                                L2.data))
    
    return B

def DOF2Vec(v):
     
    from petsc4py import PETSc

    n = len(v)
    x = PETSc.Vec()
    x.createSeq(n)
    x.setArray(v)

    return x

def Vec2DOF(x):
     
    v = x.getArray()

    return v


def DirectModeSLEPc(ffdisc,shift,nev):

    from petsc4py import PETSc
    from slepc4py import SLEPc

    # Operators
    print 'Set operators'
    Lmat = CSR2Mat(ffdisc.L)
    Bmat = CSR2Mat(ffdisc.B)

    # Setup EPS
    print 'Set solver'
    
    S = SLEPc.EPS();
    S.create()
    S.setTarget(shift)
    S.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    SI = SLEPc.ST().create()
    SI.setType(SLEPc.ST.Type.SINVERT)
    SI.setOperators(Lmat,Bmat)
    SI.setShift(shift)
    S.setST(SI)

    S.setDimensions(nev = nev,ncv = 60)
    S.setTolerances(tol=tol_ev, max_it=100)
    S.setFromOptions()

    # Solve the EVP
    print 'Solving EVP'
    S.solve()

    its = S.getIterationNumber()
    nconv = S.getConverged()
    
    omega=zeros(nconv,'complex')
    modes=zeros([ffdisc.ndof,nconv],'complex')

    ev = Lmat.getVecRight()

    for i in range(nconv):
        eigval = S.getEigenpair(i, ev)
        v      = Vec2DOF(ev)

        omega[i]  = eigval/(-1j)
        modes[:,i]= v
        
    return omega,modes
