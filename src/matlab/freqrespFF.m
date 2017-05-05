function [sigma,f,q] = freqrespFF(ffdata,omega,Pu,nu, Lq,Uq,Pq,Qq)


%

dbg = false;

ndof = ffdata.ndof;


% Assemble operators
% PQP  = Pu'*ffdata.Q*Pu;
OP = sparse( ffdata.L + 1i*omega*ffdata.B );
B2 = ffdata.B2;


if dbg,
    figure, spy(ffdata.L), title L
    figure, spy(ffdata.B), title B
    figure, spy(OP), title OP
end


% disp 'Build LU decomposition of PQP'
% [Lq,Uq,Pq,Qq] = lu(PQP); 

disp 'Build LU decomposition of (L+iwB)'
[Lop,Uop,Pop,Qop] = lu(OP);


function w = op(x) % w = (Pu'*Q*Pu)^(-H)*Pu^H*B2^H*OP^(-H)*Q*OP^(-1)*B2*Pu*x where OP=L+i*om*B
    y = Pu*x;
    z = B2*y;
    y = Qop*(Uop\(Lop\(Pop*z)));        % solve OP*y = z
    z = ffdata.Q*y;
    y = Pop'*(Lop'\(Uop'\(Qop'*z)));    % solve OP'*y = z
    z = B2'*y;
    y = Pu'*z;
    w = Pq'*(Lq'\(Uq'\(Qq'*y)));        % solve Q'*w = y
end

disp 'eigs'
n = nu;
k = 1;
opts = struct('isreal',false, 'tol',1e-8, 'maxit',100, 'disp',2);
% defaults: tol=eps(machine prec.), maxit=300, disp=1, p=2*k (or at least 20)
[V,w] = eigs(@op,n,k,'lm',opts);
disp 'done'

if dbg, w, end

sigma = sqrt(real(w));
nconv = length(real(w));
f     = zeros(ndof,nconv);
q     = zeros(ndof,nconv);
for k=1:nconv
    f(:,k) = Pu*V(:,k);
    z      = B2*f(:,k);
    q(:,k) = -Qop*(Uop\(Lop\(Pop*z)));  % solve OP*q = -z
end

end
