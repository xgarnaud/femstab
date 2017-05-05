import os
from numpy import *
try:
    import readmat
except ImportError:
    os.system('f2py -m readmat -c readmat.f90 --fcompiler=gnu95')
    import readmat

import scipy.sparse as sp

import matplotlib.pyplot as plt
import matplotlib.tri as tri


class FreeFEMdisc():
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

    idof    [ivar] : integer array (ndof), indicating which DOFs correspond
                     to variable 'ivar'
    idofi   [ivar] : integer array (ntot), indicating which element correspond
                     to variable 'ivar'
    itot    [ivar] : integer array (np1 or np2), indicating the correspondance 
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
        else:
            if dibase == None:
                dibase = di + '../base/'

            self.LoadDatFiles(di,dibase)

    def LoadDatFiles(self,di,dibase):

        ls = os.listdir(di)
        
        # Find out which components of the state vector correspond to Dirichlet BCs
        if 'BC.dat' in ls:
            BCmat,self.newind = self.LoadBC(di+'/BC.dat')
            self.ndof = len(nonzero(self.newind != -1)[0])
        else:
            raise IOError("Cannot find BC.dat in "+repr(di))


        # Read Matrices
        if 'LNS.dat' in ls:
            mat    = self.LoadMat(di+'/LNS.dat')
            self.L = BCmat.transpose() * mat * BCmat
        else:
            raise IOError("Cannot find LNS.dat in "+repr(di))

        if 'B.dat' in ls:
            mat = self.LoadMat(di+'/B.dat')
            self.B = BCmat.transpose() * mat * BCmat
        else:
            raise IOError("Cannot find B.dat in "+repr(di))

        if 'Q.dat' in ls:
            mat = self.LoadMat(di+'/Q.dat')
            self.Q = BCmat.transpose() * mat * BCmat
        else:
            raise IOError("Cannot find Q.dat in "+repr(di))


        # Find the number of variables
        print ''
        tmp = loadtxt(di+'/dofs.dat')
        nvar = int(tmp[:,0].max() + 1)
        print 'Number of variables : ',nvar
        self.nvar = nvar
        self.ntot = len(tmp[:,0])
        
        self.LoadVars(tmp)

            
        # qdof[idof[i][:]] => array of dofs     corresponding to field i = qidof
        # qtot[itot[i][:]] => array of elements corresponding to field i = qitot
        # qidof = qitot[idofi[i][:]]
        

        # Meshes
        try:
            meshtri1,meshpts1 = self.LoadMesh(dibase+'/connectivity.dat',dibase+'/coordinates.dat')
        except IOError:
            raise IOError('Cannot find '+ dibase+'/connectivity.dat and '+dibase+'/coordinates.dat')

        try:
            meshtri2,meshpts2 = self.LoadMesh(dibase+'/connectivity-2.dat',dibase+'/coordinates-2.dat')
        except IOError:
            raise IOError('Cannot find '+ dibase+'/connectivity-2.dat and '+dibase+'/coordinates-2.dat')

        self.np1 = len(meshpts1[:,0])
        self.np2 = len(meshpts2[:,0])

        self.vartype=[]
        for i in range(self.nvar):
            if self.n0[i] == self.np1:
                self.vartype.append('p1')
                print '  Variable # %2d : %6d ndof. Type: %s'%(i,self.n[i],'p1')
            elif self.n0[i] == self.np2:
                self.vartype.append('p2')
                print '  Variable # %2d : %6d ndof. Type: %s'%(i,self.n[i],'p2')
            else:
                print self.n0[i], self.np1,self.np2,self.n[i]
                raise ValueError('Neither P1 nor P2')

        self.meshp1 = tri.Triangulation(meshpts1[:,0],meshpts1[:,1],meshtri1)
        xyp1 = []
        for i in range(self.np1):
            xyp1.append((meshpts1[i,0],meshpts1[i,1]))
        xyp1 = array(xyp1,dtype=[('x', 'float'), ('y', 'float')])
        self.meshp2 = tri.Triangulation(meshpts2[:,0],meshpts2[:,1],meshtri2)
        xyp2 = []
        for i in range(self.np2):
            xyp2.append((meshpts2[i,0],meshpts2[i,1]))
        xyp2 = array(xyp2,dtype=[('x', 'float'), ('y', 'float')])

        # Associate DOFs with mesh points
        self.varorder=[]
        for i in range(self.nvar):
              indi = argsort(self.xydof[i],order=['x','y'])
              if self.vartype[i] == 'p1':
                  indm = argsort(xyp1,order=['x','y'])
              elif self.vartype[i] == 'p2':
                  indm = argsort(xyp2,order=['x','y'])
              
              ii   = argsort(indi)
              iii  = ii[indm]
              iiii = argsort(indi[iii])
              self.varorder.append(indi[iiii])

        # Load base flow
        self.q0 = loadtxt(di+'/base.dat')

    def Save(self,name):

        """
        Save the FreeFEMdisc object using the pickle module in file 'name'
        It can then be re-loaded using pickle.load
        This is less efficient than using the HDF5 IO
        """
        
        import pickle

        f = open(name,'w')
        pickle.dump(self,f)
        f.close()

    def SaveHDF5(self,fname):

        """
        Save the FreeFEMdisc object using HDF5 in file 'fname'. 
        It can be loaded when initializing an object
        """

        self.SaveHDF5_base(fname)


    def SaveHDF5_base(self,fname):
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


        def savemesth5(f,msh,gname):
            grp  = f.create_group(gname)
            dset = grp.create_dataset('x'         ,data=msh.x         )
            dset = grp.create_dataset('y'         ,data=msh.y         )
            dset = grp.create_dataset('triangles' ,data=msh.triangles )
        
        import h5py as h5
        os.system('rm -f '+fname)
        file=h5.File(fname)
        # save general information
        grpgen = file.create_group('general')
        dset = grpgen.create_dataset('ndof'     ,data=self.ndof     )  
        dset = grpgen.create_dataset('ntot'     ,data=self.ntot     )
        dset = grpgen.create_dataset('newind'   ,data=self.newind   )
        dset = grpgen.create_dataset('nvar'     ,data=self.nvar     )
        dset = grpgen.create_dataset('n'        ,data=self.n        )
        dset = grpgen.create_dataset('n0'       ,data=self.n0       )
        dset = grpgen.create_dataset('np1'      ,data=self.np1      )
        dset = grpgen.create_dataset('np2'      ,data=self.np2      ) 
        
        # save dof information
        grpdof = file.create_group('dof')
        for ivar in range(self.nvar):
            grp  = grpdof.create_group('dof_%d'%ivar)
            dset = grp.create_dataset('idof'     ,data=self.idof    [ivar])
            dset = grp.create_dataset('idofi'    ,data=self.idofi   [ivar])
            dset = grp.create_dataset('itot'     ,data=self.itot    [ivar])
            dset = grp.create_dataset('vartype'  ,data=self.vartype [ivar])
            dset = grp.create_dataset('varorder' ,data=self.varorder[ivar])
            dset = grp.create_dataset('xydof'    ,data=self.xydof   [ivar])

        # save meshes
        savemesth5(file,self.meshp1,'meshp1')
        savemesth5(file,self.meshp2,'meshp2')

        # save matrices
        savemath5(file,self.L ,'L')
        savemath5(file,self.B ,'B')
        savemath5(file,self.Q ,'Q')
        # save base flow
        grpbase = file.create_group('base')
        dset = grpbase.create_dataset('q0'     ,data=self.q0)


        file.close()

    def LoadHDF5(self,fname):

        def loadmath5(f,gname):

            shape   = f[gname+'/shape'  ].value
            indices = f[gname+'/indices'].value
            indptr  = f[gname+'/indptr' ].value
            data    = f[gname+'/data'   ].value
            return sp.csr_matrix((data, indices, indptr), shape=(shape[0], shape[1]))

        def loadmesth5(f,gname):
            x         = f[gname+'/x'        ].value
            y         = f[gname+'/y'        ].value
            triangles = f[gname+'/triangles'].value
            return tri.Triangulation(x,y,triangles)

        import h5py as h5
        file=h5.File(fname,'r')
        # load general information
        self.ndof      = file['general/ndof'     ].value
        self.ntot      = file['general/ntot'     ].value
        self.newind    = file['general/newind'   ].value
        self.nvar      = file['general/nvar'     ].value
        self.n         = file['general/n'        ].value
        self.n0        = file['general/n0'       ].value
        self.np1       = file['general/np1'      ].value
        self.np2       = file['general/np2'      ].value 

        # load dof information
        self.idof     = []
        self.idofi    = []
        self.itot     = []
        self.vartype  = []
        self.varorder = []
        self.xydof    = []

        for ivar in range(self.nvar):
            self.idof     .append(file['dof/dof_%d/idof'    %ivar].value)
            self.idofi    .append(file['dof/dof_%d/idofi'   %ivar].value)
            self.itot     .append(file['dof/dof_%d/itot'    %ivar].value)
            self.vartype  .append(file['dof/dof_%d/vartype' %ivar].value)
            self.varorder .append(file['dof/dof_%d/varorder'%ivar].value)
            self.xydof    .append(file['dof/dof_%d/xydof'   %ivar].value)

        # load meshes
        self.meshp1 = loadmesth5(file,'meshp1')
        self.meshp2 = loadmesth5(file,'meshp2')

        # load matrices
        self.L  = loadmath5(file,'L')
        self.B  = loadmath5(file,'B')
        self.Q = loadmath5(file,'Q')
        # load base flow
        self.q0 = file['base/q0'].value

        file.close()

        
    def LoadMat(self,name):
        
        print 'Reading file',name
        f = open(name, 'r')
        rr = f.readlines(5)
        f.close()

        # Read the matrix size
        str=rr[3]
        w=str.split(' ')
        #remove blanks
        data=[]
        for i in range(size(w)):
            if len(w[i]) != 0:
                data.append(w[i])

        n  =int(data[0])
        m  =int(data[1])
        nnz=int(data[3])
        print "  Matrix size:",n,'*',m,', nnz',nnz

        # Determine if the matrix is real or complex
        str=rr[4]
        w=str.split(' ')
        data=[]
        for j in range(size(w)):
            if len(w[j]) != 0:
                data.append(w[j])
        tmp = data[2].split(',')
        if len(tmp) == 2:            
            print '  Type : complex'
            icoo,jcoo,dcoo = readmat.readcomplexmat(name,nnz)
        else:
            print '  Type : real'
            icoo,jcoo,dcoo = readmat.readrealmat(name,nnz)
        icoo = icoo - 1
        jcoo = jcoo - 1

        ijcoo = [icoo,jcoo]
        # Create COO matrix
        mat=sp.coo_matrix((dcoo,ijcoo),shape=(n,m))
        del dcoo,ijcoo,icoo,jcoo
        # Convert to CSR format
        mat=mat.tocsc()

        return mat

    def LoadBC(self,name):
        tmp=loadtxt(name)
        n=size(tmp)
        ind=0
        ival=[]
        jval=[]
        dval=[]
        new_ind=zeros(n,'int')
        for i in range(n):
            if abs(tmp[i]) < 1e-10:
                ival.append(i)
                jval.append(ind)
                dval.append(1.)
                new_ind[i]=ind
                ind+=1
            else:
                new_ind[i]=-1
        dcoo=array(dval,'complex')
        ijcoo=[array(ival,'int'),array(jval,'int')]

        # Create COO matrix
        mat=sp.coo_matrix((dcoo,ijcoo),shape=(n,ind))

        # Convert to CSR format
        mat=mat.tocsc()

        return mat,new_ind


    def LoadVars(self,tmp):
        
        idof  = []
        itot  = []
        idofi = []
        ind   = []
        xydof = []
        for i in range(self.nvar):
            idof .append([])
            itot .append([])
            idofi.append([])
            xydof.append([])
            ind  .append(0)

        # Fill the lists
        for i in range(self.ntot):
            ii = int(tmp[i,0])
            if self.newind[i]!=-1:
                idof[ii].append(self.newind[i])
                itot[ii].append(ind[ii])
            idofi[ii].append(i)
            ind[ii]+=1
            xydof[ii].append((tmp[i,1],tmp[i,2]))
        
        # Convert lists to arays
        self.n         = []
        self.n0        = []
        
        for i in range(self.nvar):
            idof [i] = array(idof [i],'int')
            itot [i] = array(itot [i],'int')
            idofi[i] = array(idofi[i],'int')
            xydof[i] = array(xydof[i],dtype=[('x', 'float'), ('y', 'float')])
            self.n.append (len(idof [i]))
            self.n0.append(len(idofi[i]))
            
        self.n     = array(self.n)
        self.n0    = array(self.n0)

        assert (self.ndof == sum(self.n))
        self.idof  = idof 
        self.itot  = itot 
        self.idofi = idofi
        self.xydof = xydof

    def LoadMesh(self,triname,ptsname):

        triangles  = loadtxt(triname)
        triangles -= 1
        pts        = loadtxt(ptsname)
        return triangles,pts            

    def PlotVar(self,q,i,simcolor=True,ncontours = 20,contours = None,plot = True,returnc = False,fill = True,**kwargs):
        """
        Creates a contour plot
         - 'q' is a state vector than can be real or complex (in which
           case teh real part is plotted) and can contain points where
           Dirichlet boundary conditions are applied
         - 'i'  is the index of the field
         - 'simcolor' if true, use a symmetric colorscale
         - 'contours' if provided, the contours to be drawn
         - 'ncontours' if provided, the number of contours to be drawn
         - 'plot'   if true, the data in plotted (useful to onle get the
           field)
         -  **kwargs are arguments for the contourf function
         returns a vector that contains the value of field 'i' at each
         mesh point
        """

        if self.vartype[i] == 'p1':
            v      = zeros(self.np1,dtype(q[0]))
        else:
            v      = zeros(self.np2,dtype(q[0]))

        if len(q) == self.ndof:
            qui             = q[self.idof[i]]
            v[self.itot[i]] = qui
            v               = v[self.varorder[i]]
        elif len(q) == self.ntot:
            v               = q[self.idofi[i]]
            v               = v[self.varorder[i]]
        else:
            raise ValueError("wrong size")

        if plot:
            if simcolor:
                Mx              = max(abs(v))
                mx              = -Mx
            else:
                Mx              = max(v)
                mx              = min(v)

            if contours == None:
                contours = linspace(mx,Mx,ncontours)

            if self.vartype[i] == 'p1':
                if fill:
                    c = plt.tricontourf(self.meshp1,v.real,contours,**kwargs)
                else:
                    c = plt.tricontour(self.meshp1,v.real,contours,**kwargs)
            if self.vartype[i] == 'p2':
                if fill:
                    c = plt.tricontourf(self.meshp2,v.real,contours,**kwargs)
                else:
                    c = plt.tricontour(self.meshp2,v.real,contours,**kwargs)

        if returnc:
            return v,c
        else:
            return v

    def GetValue(self,q,i,xi,yi):
        """
        Get the value of a field at given points
         - 'q' is a state vector than can be real or complex (in which
           case teh real part is plotted) and can contain points where
           Dirichlet boundary conditions are applied
         - 'i'  is the index of the field
         - 'xi' and 'yi' are the coordinates of the points
         returns the values
        """

        from scipy.interpolate import griddata

        if self.vartype[i] == 'p1':
            v      = zeros(self.np1,dtype(q[0]))
            x      = self.meshp1.x
            y      = self.meshp1.y
        else:
            v      = zeros(self.np2,dtype(q[0]))
            x      = self.meshp2.x
            y      = self.meshp2.y

        if len(q) == self.ndof:
            qui             = q[self.idof[i]]
            v[self.itot[i]] = qui
            v               = v[self.varorder[i]]
        elif len(q) == self.ntot:
            v               = q[self.idofi[i]]
            v               = v[self.varorder[i]]
        else:
            raise ValueError("wrong size")

        # zi = griddata((x, y), v, (xi[None,:], yi[:,None]), method='cubic')
        zi = griddata((x, y), v, (xi, yi), method='cubic')
        return zi
        
    def GetHmin(self):
    
        edges = self.meshp1.edges
        x     = self.meshp1.x
        y     = self.meshp1.y
        
        nedges,n = edges.shape

        hmin = 1e3
        for i in range(nedges):
            h12 = (x[edges[i,0]]-x[edges[i,1]])**2 + \
                (y[edges[i,0]]-y[edges[i,1]])**2
            hmin = min(hmin,sqrt(h12))
            
        return hmin

    def GetDt(self,cfl,iu):

        edges = self.meshp2.edges
        x     = self.meshp2.x
        y     = self.meshp2.y

        utot = zeros(self.np2)
        for i in iu:
            tmp  = self.q0[self.idofi[i]]**2
            utot += tmp[self.varorder[i]]

        utot = sqrt(utot)

        hmin = 1e3*ones(self.np2)
        nedges,n = edges.shape
        for i in range(nedges):
            p0 = edges[i,0]
            p1 = edges[i,1]
            h  = sqrt((x[p0]-x[p1])**2 + (y[p0]-y[p1])**2)
            hmin[p0] = min(hmin[p0],2*h)
            hmin[p1] = min(hmin[p1],2*h)

        utot = hmin*cfl#/utot
        print hmin.min()
        return utot

    def SaveFieldFF(self,q,fname):


        for i in range(self.nvar):
            f = open(fname+'_%d.dat'%i,'w')
            if self.vartype[i] == 'p1':
                v      = zeros(self.np1,dtype(q[0]))
                f.write("%e \n"%self.np1)
            else:
                v      = zeros(self.np2,dtype(q[0]))
                f.write("%e \n"%self.np2)
            

            if len(q) == self.ndof:
                qui             = q[self.idof[i]]
                v[self.itot[i]] = qui
            elif len(q) == self.ntot:
                v               = q[self.idofi[i]]
            else:
                raise ValueError("wrong size")
        
            for j in range(len(v)):
                f.write("(%e,%e)\n"%(v[j].real,v[j].imag))
            
            f.close()
            

    def getPu(self,iu,mask = None):
    
        """
        Computes a scipy sparse matrix (sequential) that takes as as input a 
        reduced vector defined only for some fields and perhaps for some spatial
        region and returns a vector of DOFs where the other elements are set to 
        0
        inputs : iu, integer array, containing the list of fields on which the
                     reduced vector is defined
                 mask, function of (x,y), true if the reduced vector should be
                     defined at (x,y). None if always true

        outputs: Pu, scipy sparse matrix
                 
        TO DO: PETSC // version
        """

        ival = []
        jval = []
        dval = []
        nu    = 0
        for ivar in iu:
            for i in range(self.n[ivar]):
                x,y=self.xydof[ivar][self.itot[ivar][i]]
                if mask == None or mask(x,y):
                    ival.append(self.idof[ivar][i])
                    jval.append(nu);
                    dval.append(1.)
                    nu += 1
            
        dcoo  = array(dval,'complex')
        ijcoo = [array(ival,'int'),array(jval,'int')]

        # Create COO matrix
        Pu    = sp.coo_matrix((dcoo,ijcoo),shape=(self.ndof,nu))
        # Convert to CSR format
        Pu    = Pu.tocsc()

        return Pu

class EmptyFreeFEMdisc():
    def __init__(self):
        self.L = None
        self.B = None
        self.Q = None


