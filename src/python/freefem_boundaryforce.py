from freefem import *

class FreeFEMdisc_boundaryforce(FreeFEMdisc):
    """
    object that contains all information about the FreeFem++
    discretization
    
    GENERAL INFORMATION

    ndof           : integer, nomber of degrees of freedom
    ntot           : integer, total number of discretization elements
                     (including those that are 0 due to BCs)
    newind         : integer array (ntot), contain the new index of a 
                     point when only DOFs are kept, or -1 if a dirichlet
                     BC is applied at that point
    nvar           : integer, nomber of variables (i.e. ux, uy, p,...)
    n              : integer array (nvar): number of DOFs in each variable 
    n0             : integer array (nvar): number of elements in each variable 
    np1            : integer, number of elements on the P1 mesh
    np2            : integer, number of elements on the P2 mesh

    idof    [ivar] : integer array (ndof), indivating which DOFs correspond
                     to variable 'ivar'
    idofi   [ivar] : integer array (ntot), indivating which element correspond
                     to variable 'ivar'
    itot    [ivar] : integer array (np1 or np2), indicating teh correspondance 
                     between elements and DOFs of variable ivar
    vartype [ivar] : string array, indicating if the field is discretized 
                     P1 or P2 elements
    varorder[ivar] : order or the elements of variable 'ivar' relative to the
                     corresponding mesh

    MESHES

    meshp1         : matplotlib.tri.Triangulation, P1 mesh
    meshp2         : matplotlib.tri.Triangulation, P2 mesh

    MATRICES (containing anly DOFs)

    L              : real or complex scipy.sparse CSR matrix, 
    B              : real or complex scipy.sparse CSR matrix, mass matrix
    Q              : real or complex scipy.sparse CSR matrix, inner product
                     matrix
 
    q0             : real array, base flow state vector

    BOUNDARY FORCING

    bcforcing      : bool, indicating whether forcing is applied on the boundary 
                     if true, only FR studies can be performed and the following variables are set
    forcingind     : indes of the DOFs where forcing is applied
    B2             : projection from data on the inflow  to dofs everywhere
    Qf             : norm of the forcing (1d integral)
    Pf             : projection onto the forcing space
    xf,rf          : positions associated with the forcing
    ivarf          : variable number for each forcing DOF

    """

    def __init__(self,di,dibase = None):
        """
        Initilize the object using either 
          - the text files written by FreeFem++ in folder 'di' (the 
            base flow data should be in 'di'/../base/ unless given
            in dibase)
          - an *.h5 file obtained using function SaveHDF5
        """

        if di[-3:] == '.h5':
            self.LoadHDF5(di)
            self.LoadHDF5_boundaryforce(di)
        else:
            if dibase == None:
                dibase = di + '../base/'

            self.LoadDatFiles(di,dibase)
            self.LoadDatFiles_boundaryforce(di,dibase)

    def LoadDatFiles_boundaryforce(self,di,dibase):

        BCmat,self.newind = self.LoadBC(di+'/BC.dat')

        ls = os.listdir(di)

        if 'BC_forcing.dat' in ls:
            self.forcingind = self.LoadForcingBC(di+'/BC_forcing.dat')
            self.forcingind = self.newind[self.forcingind]
            nodir           = nonzero(self.forcingind != -1)
            self.forcingind = self.forcingind[nodir]
            mat             = self.LoadMat(di+'/Qf.dat')
            self.Qf   = BCmat.transpose() * mat * BCmat
            self.Pf   = self.GetPf(self.forcingind)
            self.Qf   = self.Pf.transpose()*self.Qf*self.Pf
        else:
            raise IOError("Cannot find BC_forcing.dat in "+repr(di))

        self.B2 = BCmat.transpose() * BCmat
        self.SetBC(self.L,self.forcingind,-1.)
        self.SetBC(self.B,self.forcingind,0.)

        tmp = loadtxt(di+'/dofs.dat')
        ikeep = nonzero(self.newind != -1)[0]
        x    = tmp[:,1][ikeep]
        r    =  tmp[:,2][ikeep]
        ivar = tmp[:,0][ikeep]
        self.xf = x[self.forcingind]
        self.rf = r[self.forcingind]
        self.ivarf = ivar[self.forcingind]

    def SaveHDF5(self,fname):
        """
        Save the FreeFEMdisc object using HDF5 in file 'fname'. 
        It can be loaded when initializing an object
        """

        self.SaveHDF5_base(fname)
        self.SaveHDF5_boundaryforce(fname)

    def SaveHDF5_boundaryforce(self,fname):
        """
        Save the FreeFEMdisc object using HDF5 in file 'fname'. 
        It can be loaded when initializing an object
        """
        def savemath5(f,mat,gname):
            grp  = f.create_group(gname)
            mat  = mat.tocsr()
            dset = grp.create_dataset('shape'     ,data=mat.shape   )
            dset = grp.create_dataset('indices'   ,data=mat.indices )
            dset = grp.create_dataset('indptr'    ,data=mat.indptr  )
            dset = grp.create_dataset('data'      ,data=mat.data    )

        import h5py as h5

        file=h5.File(fname)
        grpdof = file['dof']
        dset = grpdof.create_dataset('forcingind' ,data=self.forcingind )
        dset = grpdof.create_dataset('xf'         ,data=self.xf         )
        dset = grpdof.create_dataset('rf'         ,data=self.rf         )
        dset = grpdof.create_dataset('ivarf'      ,data=self.ivarf      )

        savemath5(file,self.B2 ,'B2')
        savemath5(file,self.Qf ,'Qf')
        savemath5(file,self.Pf ,'Pf')

        file.close()

    def LoadHDF5_boundaryforce(self,fname):

        def loadmath5(f,gname):
            shape   = f[gname+'/shape'  ].value
            indices = f[gname+'/indices'].value
            indptr  = f[gname+'/indptr' ].value
            data    = f[gname+'/data'   ].value
            return sp.csr_matrix((data, indices, indptr), shape=(shape[0], shape[1]))
        import h5py as h5
        
        file=h5.File(fname,'r')

        self.forcingind =file['dof/forcingind' ].value
        self.xf         =file['dof/xf'         ].value
        self.rf         =file['dof/rf'         ].value
        self.ivarf      =file['dof/ivarf'      ].value
        
        self.B2 = loadmath5(file,'B2')
        self.Qf = loadmath5(file,'Qf')
        self.Pf = loadmath5(file,'Pf')

        file.close()

    def LoadForcingBC(self,name):
        tmp=loadtxt(name)
        n=size(tmp)
        ind=[]
        for i in range(n):
            if abs(tmp[i]) > 1e-10:
                ind.append(i)

        ind = array(ind)

        return ind

    def SetBC(self,mat,idx,val):
        
        mat = mat.tocsr()
        for i in idx:
            j1 = mat.indptr[i]
            j2 = mat.indptr[i+1]
            for j in range(j1,j2):
                if mat.indices[j] == i:
                    mat.data[j] = val
                else:
                    mat.data[j] = 0.

    def GetPf(self,idx):

        assert(len(idx) > 0)

        nf = len(idx)
        ival = idx
        jval = range(nf)
        dval = ones(nf)

        dcoo  = array(dval,'complex')
        ijcoo = [array(ival,'int'),array(jval,'int')]

        # Create COO matrix
        Pf    = sp.coo_matrix((dcoo,ijcoo),shape=(self.ndof,nf))
        # Convert to CSR format
        Pf    = Pf.tocsc()

        return Pf

    def PlotBoundaryVar(self,f,ivar,ax,add = ((None,None),(None,None))):

        i0 = nonzero(self.ivarf == ivar)[0]
        if ax == 'x':
            xx = self.xf[i0]
            xx = self.xf[i0]
        elif ax == 'y':
            xx = self.rf[i0]
            xx = self.rf[i0]
                
        isort = xx.argsort(); xx = xx[isort]; 
        
        bc0 = add[0];
        bc1 = add[1];

        if bc0[0] != None:
            xx = append(bc0[0],xx);
        if bc1[0] != None:
            xx = append(xx,bc1[0]);

        ff = f[i0]
        ff = ff[isort]

        if bc0[0] != None:
            ff = append(bc0[1],ff);
        if bc1[0] != None:
            ff = append(ff,bc1[1]);

        if ax == 'x':
            plt.plot(xx,ff)
        elif ax == 'y':
            plt.plot(ff,xx)

        
        return xx,ff
