tic;


clear all

dographs=1;

eps=0.5;

crit = 1e-8;    % Numerical precision for outside loop
tol  = 1e-8;    % Numerical precision for inside loop

% Aiyagari model

alph=1/3;
bet=0.96;
gam=2;
delt=0.08;

Ps=[0.7497 0.2161 0.0322 0.002 0;
    0.2161 0.4708 0.2569 0.0542 0.002;
    0.0322 0.2569 0.4218 0.2569 0.0322;
    0.002 0.0542 0.2569 0.4708 0.2161 ;
    0 0.002 0.0322 0.2161 0.7497];

sgrid=[0.6177 0.8327 1.0000 1.2009 1.6188]';

ns=5;
nk=250;

bl=0;

%kgrid=[linspace(bl,5,nk/2) linspace(5+5/(nk/2),30,nk/2)]';

kgrid=linspace(bl,30,nk)';

kgridl=kron(kgrid,ones(ns,1));

sgridl=repmat(sgrid,nk,1);

%stationary income distribution  Labour supply
pis0=[0 0 1 0 0]';

pis=pis0'*Ps^50;

pis=pis';

N=sum(pis.*sgrid);

iter=0;


rmax=0.2;
rmin=-0.05;

K1=20;

r=1/bet-1;

while abs(rmax-rmin)>crit

    iter=iter+1;
    
% return function

K0=((((r+delt)/alph))/N^(1-alph))^(1/(alph-1));

w=(1-alph)*K0^(alph)*N^(-alph);

U= (1/(1-gam))*max(1e-24,kgridl*(1+r)+w*sgridl-kgrid').^(1-gam);

%        k1' .................... kn' 
% k1 s1
% k1 s2
%
%
% kn sn

% value function iteration

V1=zeros(nk,ns);
V0=V1+10;


while norm(V0-V1,'inf')>tol
    
    V0=V1;
    
    EV=V0*Ps';
    EVL=repmat(EV',nk,1);
    
    W=U+bet*EVL;
    
    [V1L g] =max(W,[],2);
    
    V1=reshape(V1L,ns,nk)';
    
    
end

%%% stationary distribution

PI=zeros(nk*ns,nk*ns);

%PI(sub2ind(size(PI),repmat([1:nk*ns]',1,5),(g-1)*ns+[1:5]))=Ps(repmat([1:ns]',nk,1),:); % combined policy and income transition matrix

PI(sub2ind(size(PI),repmat([1:nk*ns]',1,5),(g-1)*ns+[1:5]))=repmat(Ps,nk,1); % combined policy and income transition matrix


%lam=ones(nk*ns,1)/(nk*ns);

%lam=(lam'*PI^500)';

[lam,eigen]=eigs(PI',1);  % Ergodic Distribution as left unit eigenvector
    lam=lam./sum(lam);


K1=sum(lam.*kgrid(g));

if K1>K0
    rmax=r;
else
    rmin=r;
end

r=(rmax+rmin)/2;

if iter>200
    iter
    break
end


capgraph(iter,:)= [K1 K0 rmax rmin r];

end


lam2=reshape(lam,ns,nk)';


if dographs==1

figure(1)
clf
surf(sgrid,kgrid,lam2)
title('distribution of assets and income')


figure(2)
clf
subplot(1,2,1)
bar(kgrid,sum(lam2,2))
title ('asset dist')

subplot(1,2,2)
bar(sgrid,sum(lam2))
title('income dist')


figure(3)
clf
scatter(capgraph(:,1),capgraph(:,5))
hold on
scatter(capgraph(:,2),capgraph(:,5))
legend('capital supply','capital demand')
xlabel('K')
ylabel('r')

display('Ginis [wealth income]')

fk=sum(lam2,2);
Fk=cumsum(fk);

fy=sum(lam2)';
Fy=cumsum(fy);

muk=sum(fk.*kgrid);
muy=sum(fy.*sgrid);

Gw=0;
Gy=0;

for i=1:nk
    Gw=Gw+sum(fk(i)*fk.*abs(kgrid(i)-kgrid));
end

for i=1:ns
   
    Gy=Gy+sum(fy(i)*fy.*abs(sgrid(i)-sgrid));
    
end

Gw=Gw/(2*muk);
Gy=Gy/(2*muy);

[Gw Gy]

display('Wealth variance')

sum(lam.*(kgridl-muk).^2)

display('Income variance')

sum(lam.*(sgridl-muy).^2)

display('Wealth and income covariance')

sum(lam.*(kgridl-muk).*(sgridl-muy))


%%%% hitting time % think about lake model

% E[T]=pitrans*(E(T)+1)

et=[1 ;zeros(ns-1,1)];


display('Time from lowest to highest income state')


%%%% hitting time (assets
Pstrans=Ps(1:end-1,1:end-1);

ht=((eye(size(Pstrans))-Pstrans)^-1)*ones(ns-1,1);

ht(1)


display('Time from lowest to highest asset state')

Pstrans=PI(1:end-1,1:end-1);

ht=((eye(size(Pstrans))-Pstrans)^-1)*ones(ns*nk-1,1);

ht(1)


end 


toc;