function MAT = readmatFF(fullfilename,filetype)
%
% readmatFF.m
%
% Read FreeFem++ matrix from .mat or .dat file.
%
% inputs: fullfilename > full path (e.g. sthg like: fullfile(pathname,filename))
%         filetype     > 'mat' or 'dat'
%
%

% disp 'readmatFF'
tic;

%pwd0 = pwd;
%cd(fold1);

%format long;

switch filetype
    case 'mat'
        load([fullfilename '.mat']);
        
    case 'dat'
        %--- open file
        %disp(' '); disp(['read file ' filename]);
        fid = fopen([fullfilename '.dat']);
        
        %--- read header lines (matrix size, symmetry, nb of non-zero entries)
        nn    = textscan(fid,'%f',1,'HeaderLines',3);   nn       = nn{1};
        mm    = textscan(fid,'%f',1);                   mm       = mm{1};
        issym = textscan(fid,'%u',1);                   issymNSL = issym{1};
        ncoef = textscan(fid,'%f',1);                   ncoef    = ncoef{1};
        disp(['read ' num2str(ncoef) ' non-zero coeffs (among ' num2str(nn) 'x' num2str(mm) '=' num2str(nn*mm) ' DOFs...)'])
        if (issymNSL), disp('symmetric'), else disp('non-symmetric'), end

        %--- read data line by line
        ii  = zeros(ncoef,1);
        jj  = zeros(ncoef,1);
        aij = zeros(ncoef,1);
        for kk=1:ncoef
            iitmp  = textscan(fid,'%u',1);               ii(kk)  = iitmp{1};
            jjtmp  = textscan(fid,'%u',1);               jj(kk)  = jjtmp{1};
            aijtmp = textscan(fid,'%c %f %c %f %c',1);   aij(kk) = aijtmp{2} + 1i*aijtmp{4};
        end
        fclose(fid);
        
        %--- build matrix 
        MAT = sparse(ii,jj,aij,nn,mm);
        clear('aij','ii','jj');
        
        %--- save matrix
        save([fullfilename '.mat'], 'MAT');
 end

toc
