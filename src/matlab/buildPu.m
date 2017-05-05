function [Pu,nu] = buildPu(ffdata, iu)

% iu = [1,2] if variables = u,v,p 

% Define the matrix of the projection on the velocity space
% Can be made more efficient...

tic

idof = ffdata.idof;
ndof = ffdata.ndof;
n    = ffdata.n;

ival = [];
jval = [];
dval = [];
nu   = 0;

for ivar=1:length(iu)
    for i=1:n(ivar)
        ival = [ival, idof(ivar,i)];
        jval = [jval, nu+i];
        dval = [dval, 1];
    end
    nu = nu + n(ivar);
end

Pu = sparse(ival,jval,dval,ndof,nu);

toc

end
