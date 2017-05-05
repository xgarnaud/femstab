function v = plotvarsFF(ffdata,q,k,gr_opts)
%
% plotvarsFF.m
%
% inputs:  ffdata
%          q > vector to plot
%          k > which DOF (e.g. (u,v,p)=(1,2,3)
%          gr_opts > options such as simmcolor [true|false]
%                                 or colormap
%
%

idof     = ffdata.idof;
itot     = ffdata.itot;
vartype  = ffdata.vartype;
varorder = ffdata.varorder;

if strcmp(vartype(k),'p1')
    v = zeros(ffdata.np1,1);
else % 'p2'
    v = zeros(ffdata.np2,1);
end

if (length(q) == ffdata.ndof)
    qui                       = q( idof(k,idof(k,:)~=0) );
    v( itot(k,itot(k,:)~=0 )) = qui;
    v                         = v(varorder{k});
elseif (length(q) == ffdata.ntot)
    v                 = q(ffdata.idofi(k));
    v                 = v(varorder{k});
else
    disp 'Error(wrong size)'
end

vr = real(v);
vi = imag(v);
if gr_opts.simmcolor
    Mx = max(abs(vr));
    mx = -Mx;
else
    Mx = max(vr);
    mx = min(vr);
end

if strcmp(vartype(k),'p1')
    % tricontourf(ffdata.meshp1,real(v),linspace(mx,Mx,20))
    meshpts = ffdata.meshp1.meshpts;
    meshtri = ffdata.meshp1.meshtri;
else % 'p2' 
    % tricontourf(ffdata.meshp2,real(v),linspace(mx,Mx,20))
    meshpts = ffdata.meshp2.meshpts;
    meshtri = ffdata.meshp2.meshtri;
end

p = meshpts';
e = [];
t = double([meshtri,zeros(size(meshtri,1),1)]');
pdeplot(p,e,t,'xydata',vr)
axis equal
caxis([mx,Mx])
colormap(gr_opts.colormap);

end