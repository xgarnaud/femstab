function [ffdata] = loadvarsFF(newind,ffdata)
%
% readdofsFF.m
%
% Read FreeFem++ DOFs from .dat file.
%
% input: uxy  > vector [ndof*3] with [0,1,2] for u,v,p, x coords, y coords
%        nvar > nb of variables (e.g. 3 for u,v,p)
%        ntot > nb of DOFs
%        newind > vector with i-th entry = -1, if i-th DOF is a BC DOF, 
%                                        = non-BC DOF #, otherwise
%


disp 'loadvarsFF'
tic;

nvar = ffdata.nvar;
ntot = ffdata.ntot;
uxy  = ffdata.dofs;

idof  = zeros(nvar,ntot);   % # of non-BC DOFs
itot  = zeros(nvar,ntot);   % # of these non-BC DOFs, when counted among all DOFs
idofi = zeros(nvar,ntot);   % # of all DOFs
xydof = zeros(ntot,2,nvar); % coords of all DOFs
ind   = zeros(nvar,1);      % running counts for u,v,p DOFs
ind0  = zeros(nvar,1);      % running counts for non-BC u,v,p DOFs

for k=1:ntot
   ii = uxy(k,1)+1; % [0,1,2]+1 for u,v,p
   ind(ii) = ind(ii)+1;
   if ( newind(k)~=-1 ) % we have a BC DOF here
      ind0(ii) = ind0(ii)+1;
      idof(ii,ind0(ii)) = newind(k);
      itot(ii,ind0(ii)) = ind(ii);
   end
   idofi(ii,ind(ii)) = k;
   xydof(ind(ii),:,ii) = uxy(k,2:3); % x,y coords
end

n  = zeros(nvar,1);
n0 = zeros(nvar,1);
for k=1:nvar
    n(k)  = sum(idof(k,:)~=0);
    n0(k) = sum(idofi(k,:)~=0);
end
ndof = sum(n);

ffdata.n     = n;
ffdata.n0    = n0;
ffdata.ndof  = ndof;
ffdata.idof  = idof;
ffdata.idofi = idofi;
ffdata.itot  = itot;
ffdata.xydof = xydof;

toc

end