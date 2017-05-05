from freefem import *

class FreeFEMdisc_bodyforce(FreeFEMdisc):
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
    B2             : real or complex scipy.sparse CSR matrix, forcing mass 
                     matrix
    Q              : real or complex scipy.sparse CSR matrix, inner product
                     matrix
 
    q0             : real array, base flow state vector

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
            self.LoadHDF5_bodyforce(di)
        else:
            if dibase == None:
                dibase = di + '../base/'

            self.LoadDatFiles(di,dibase)
            self.LoadDatFiles_bodyforce(di,dibase)

    def LoadDatFiles_bodyforce(self,di,dibase):

        ls = os.listdir(di)

        BCmat,self.newind = self.LoadBC(di+'/BC.dat')

        if 'B2.dat' in ls:
            mat = self.LoadMat(di+'/B2.dat')
            self.B2 = BCmat.transpose() * mat * BCmat
        else:
            raise IOError("Cannot find B2.dat in "+repr(di))

    def SaveHDF5(self,fname):
        """
        Save the FreeFEMdisc object using HDF5 in file 'fname'. 
        It can be loaded when initializing an object
        """

        self.SaveHDF5_base(fname)
        self.SaveHDF5_bodyforce(fname)

    def SaveHDF5_bodyforce(self,fname):
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

        # save matrices
        savemath5(file,self.B2,'B2')

        file.close()

    def LoadHDF5_bodyforce(self,fname):

        def loadmath5(f,gname):
            shape   = f[gname+'/shape'  ].value
            indices = f[gname+'/indices'].value
            indptr  = f[gname+'/indptr' ].value
            data    = f[gname+'/data'   ].value
            return sp.csr_matrix((data, indices, indptr), shape=(shape[0], shape[1]))
        import h5py as h5
        
        file=h5.File(fname,'r')
        self.B2 = loadmath5(file,'B2')

        file.close()
