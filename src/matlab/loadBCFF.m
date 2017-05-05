function [BCmat,newind] = loadBCFF(fullfilename)
%
% loadBCFF.m
%
% Read FreeFem++ BC vector from .dat file, and build BC matrix with
% entry 1 for each non-BC DOF, at (i,j)=(global DOF number, non-BC DOF number)
%
% input: fullfilename > full path (e.g. sthg like: fullfile(pathname,filename))
% outputs: BCmat > BC matrix with entry 1 for each non-BC DOF, at (i,j)=(global DOF #, non-BC DOF #)
%                  Used to extract non-BC DOFs.
%          newind > vector with i-th entry = -1, if i-th DOF is a BC DOF, 
%                                          = non-BC DOF number, otherwise
%
%

disp 'loadBCFF'
tic;

bc = load(fullfilename);
n  = length(bc);
ind = 0;

ival = [];
jval = [];
dval = [];

newind = zeros(n,1);
for i=1:n
    if ( abs(bc(i))<1e-10 )     % we have a non-BC DOF here
        ind = ind+1;
        ival = [ival, i];
        jval = [jval, ind];
        dval = [dval, 1];
        newind(i) = ind;
    else                        % we have a BC DOF here
        newind(i) = -1;
    end
end

BCmat = sparse(ival,jval,dval,n,ind);

toc
