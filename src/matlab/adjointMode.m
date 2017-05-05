function [evec,lam] = adjointMode(A,B,shift_lam,nval,startvec)
%
% Solve A'*w = lam*B*w, for lam near shift_lam
%
% Input shift_lam should be close to conj. of direct eigval of interest
%

if ~issparse(A)
    A=sparse(A);
end
if ~issparse(B)
    B=sparse(B);
end
     
OP = A'-shift_lam*B;
[L,U,p,q] = lu(OP,'vector');

h = @(x)shiftinvert(L,U,p,q,B,x);
opts.isreal = false;
opts.maxit  = 200;
% opts.tol    = 1e-30;
opts.disp   = 2;
% opts.v0     = startvec;
[evec,v] = eigs(h,length(A),nval,'lm',opts);

lam = 1./diag(v) + shift_lam;
% convert to omega
% om = 1i*( 1./diag(v) + shift_lam );

function out = shiftinvert(L,U,p,q,B,x) % return OP^(-1)*B*x
    y = B*x;
    out(q) = U\(L\y(p));
%    spparms('spumoni',0)
end

end