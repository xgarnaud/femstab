
% addpath('D:\work\xav_femstab_freqresponse\FF\matlab');
cd('D:\work\xav_femstab_freqresponse\FF\matlab');
% addpath('/home/boujo/work/calc/xav_femstab_freqresponse/FF/matlab');
% cd('/home/boujo/work/calc/xav_femstab_freqresponse/FF/matlab');

addpath('../');

pth = '../data_cyl/';

figdir = 'figs/';
saveplo = true;



readmattype = 'mat'; % 'dat' or 'mat'


%--------------------------------------------------------------------
% build BC matrix (used to extract non-BC DOFs)
[BC,newind] = loadBCFF(fullfile(pth,'BC.dat'));


% read operators (and extract non-BC DOFs)
% !!! With the definitions in varf_cyl, Navier-Stokes read B*dx/dt = -L*x !!!
disp 'read L';    L  = readmatFF(fullfile(pth,'LNS'), readmattype);   L  = BC'*L*BC;
disp 'read B';    B  = readmatFF(fullfile(pth,'B'),   readmattype);   B  = BC'*B*BC;
disp 'read B2';   B2 = readmatFF(fullfile(pth,'B2'),  readmattype);   B2 = BC'*B2*BC;
disp 'read Q';    Q  = readmatFF(fullfile(pth,'Q'),   readmattype);   Q  = BC'*Q*BC;
% Qp2  = readmatFF(fullfile(pth,'Qp2.dat'),  'dat');
% Dxp2 = readmatFF(fullfile(pth,'Dxp2.dat'), 'dat');
% Dyp2 = readmatFF(fullfile(pth,'Dyp2.dat'), 'dat');

% ffdata = struct('BC',BC,'L',L,'B',B,'B2',B2,'Q',Q,'Qp2',Qp2,'Dxp2',Dxp2,'Dyp2',Dyp2);
ffdata = struct('BC',BC,'L',L,'B',B,'B2',B2,'Q',Q);
clear BC L B B2 Q;

% read vars and dofs
ffdata = readdofsFF(fullfile(pth,'dofs.dat'),ffdata);
ffdata = loadvarsFF(newind,ffdata);


% load meshes
ffdata = loadmeshFF(pth,ffdata);


% # # Time stepping
% 
% # q0 = zeros(ffdisc.ndof)
% # tmp = zeros(ffdisc.np1)
% # x = ffdisc.meshpts[:,0]
% # r    = ffdisc.meshpts[:,1]
% # tmp  = exp(-(x-10)**2)*exp(-(r)**2)
% # tmp2 = ffdisc.P1toP2*tmp
% 
% # q0[ffdisc.ivar[0]] = tmp2[ffdisc.ivar_wbc[0]]
% # qx = ffdisc.PlotP2Var(q0,0)
% 
% # q1 = fs.TS(ffdisc,1e-2,5,q0)
% # plt.figure()
% # qx = ffdisc.PlotP2Var(q1,0)
% 
% # plt.show()


%--------------------------------------------------------------------
% Frequency response
[Pu,nu] = buildPu(ffdata, [1,2]);

disp 'Build LU decomposition of PQP'
PQP  = Pu'*ffdata.Q*Pu;
[Lq,Uq,Pq,Qq] = lu(PQP); 

omega = [5 10 12.5 15 16 17:0.5:19 20 22.5 25:5:40];
nom = length(omega);
sigma = zeros(1,nom);

for iom=1:nom
    om = omega(iom)
    [sig,f,q] = freqrespFF(ffdata,om,Pu,nu, Lq,Uq,Pq,Qq);
    sigma(iom) = sig;
    
    gr_opts.simmcolor = true;    % symmetric color range
    gr_opts.colormap  = 'jet';
    nconv = length(sig);
    for k=1:nconv
        disp(['sigma: ' num2str(sigma(k))]);
        
        figure, DOF = 1;
        fx = plotvarsFF(ffdata, real(f(:,k)), DOF, gr_opts);
        title(['forcing at omega='  num2str(om) ', amplification=' num2str(sig) ', real(ux)'])
        if saveplo, saveas(gcf,[pth,figdir,'w' num2str(om) '-forc-ux.fig']), close, end
        
        figure, DOF = 1;
        ux = plotvarsFF(ffdata, real(q(:,k)), DOF, gr_opts);
        title(['response at omega=' num2str(om) ', amplification=' num2str(sig) ', real(ux)'])
        if saveplo, saveas(gcf,[pth,figdir,'w' num2str(om) '-resp-ux.fig']), close, end
        
        figure, DOF = 3;
        p = plotvarsFF(ffdata, real(q(:,k)), DOF, gr_opts);
        title(['response at omega=' num2str(om) ', amplification=' num2str(sig) ', real(p)'])
        if saveplo, saveas(gcf,[pth,figdir,'w' num2str(om) '-resp-p.fig']), close, end
    end
end

figure, plot(omega,sigma,'o-')
set(gca,'yscale','log')

save( fullfile(pth,'sigma.mat'), 'omega','sigma' )

% return


%--------------------------------------------------------------------
% Global modes
shift_lam = 1.5+17i;
% shift_om  = 1i*shift_lam;
nev = 3;
% [evec,om] = DirectMode(-ffdata.L,ffdata.B,shift_om,nev);
[evec,lam] = directMode(-ffdata.L, ffdata.B, shift_lam, nev);

gr_opts.simmcolor = true;    % symmetric color range
gr_opts.colormap  = 'jet';
nconv = length(lam);
for k=1:nconv
     disp(['lambda: ' num2str(lam(k))])
     
     figure, DOF = 1;
     ux = plotvarsFF(ffdata, real(evec(:,k)), DOF, gr_opts);
     title(['global mode (lambda=' num2str(lam(k)) '), real(ux)'])
     if saveplo, saveas(gcf,[pth,figdir,'glob-lam' num2str(lam(k)) '-ux-re.fig']), close, end
     
     figure, DOF = 2;
     uy = plotvarsFF(ffdata, real(evec(:,k)), DOF, gr_opts);
     title(['global mode (lambda=' num2str(lam(k)) '), real(uy)'])
     if saveplo, saveas(gcf,[pth,figdir,'glob-lam' num2str(lam(k)) '-uy-re.fig']), close, end
end


%--------------------------------------------------------------------
% Adjoint modes
nev = 3;
[evec,lam] = adjointMode(-ffdata.L, ffdata.B, shift_lam', nev);

gr_opts.simmcolor = true;    % symmetric color range
gr_opts.colormap  = 'jet';
nconv = length(lam);
for k=1:nconv
     disp(['lambda: ' num2str(lam(k))])
     
     figure, DOF = 1;
     ux = plotvarsFF(ffdata, real(evec(:,k)), DOF, gr_opts);
     title(['adjoint mode (lambda=' num2str(lam(k)) '), real(ux)'])
     if saveplo, saveas(gcf,[pth,figdir,'adj-lam' num2str(lam(k)) '-ux-re.fig']), close, end

%      figure, DOF = 1;
%      ux = plotvarsFF(ffdata, imag(evec(:,k)), DOF, gr_opts);
%      title(['adjoint mode (lambda=' num2str(lam(k)) '), imag(ux)'])
%      if saveplo, saveas(gcf,[pth,figdir,'adj-lam' num2str(lam) '-ux-im.fig']), close, end
     
     figure, DOF = 2;
     uy = plotvarsFF(ffdata, real(evec(:,k)), DOF, gr_opts);
     title(['adjoint mode (lambda=' num2str(lam(k)) '), real(uy)'])
     if saveplo, saveas(gcf,[pth,figdir,'adj-lam' num2str(lam(k)) '-uy-re.fig']), close, end
end
