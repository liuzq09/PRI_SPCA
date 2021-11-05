% This code was adpated from the corresponding code downloaded from https://github.com/GauriJagatap/model-copram

%% The Initialization step in CoPRAM
function [x,p,x_init] =  CoPRAM_init(y_abs,A,s,z)
%%updated 5/31/2017

%% initialize parameters
[m,n] = size(A);
%If ground truth is unknown
if nargin < 4
    z = zeros(n,1);
end
p = zeros(m,1); %phase vector
Marg = zeros(1,n); %marginals
MShat = zeros(s); %truncated correlation matrix
AShat = zeros(m,s); %truncated sensing matrix
supp = zeros(1,n); %indicator for initial support Shat
y_abs2 = y_abs.^2; %quadratic measurements
phi_sq = sum(y_abs2)/m;
phi = sqrt(phi_sq); %signal power

%% s-Truncated sensing vectors

%signal marginals
Marg = ((y_abs2)'*(A.^2))'/m; % n x 1
[Mg MgS] = sort(Marg,'descend');
S0 = MgS(1:s); %pick top s-marginals
Shat = sort(S0); %store indices in sorted order
%supp(Shat) = 1; figure; plot(supp); %support indicator
AShat = A(:,Shat); % m x s %sensing sub-matrix

%% Truncated measurements
TAF = 'on'; %consider only truncated measurements m' < m ; gives marginally better performance 
TAF = 'off'; %consider all measurements = m ; aligns with code presented in paper

switch TAF
    case 'on'
        card_Marg = ceil(m/6);
        %large measurements - amplitude flow
        for i=1:m
            M_eval(i) = y_abs(i)/norm(AShat(i,:));
        end 
        [~,MmS] = sort(M_eval,'descend');
        Io = MmS(1:card_Marg); %indices between 1 to m
    case 'off'
        card_Marg = m;
        Io = 1:card_Marg;
end

%% initialize x
%compute top singular vector according to thresholded sensing vectors and large measurements
for i = 1:card_Marg
    ii = Io(i);
    MShat = MShat + (y_abs2(ii))*AShat(ii,:)'*AShat(ii,:); % (s x s)
end

svd_opt = 'svd'; %more accurate, but slower for larger dimensions
svd_opt = 'power'; %approximate, faster for larger dimensions

switch svd_opt
    case 'svd'
        [u,sigma,v] = svd(MShat);
        v1 = u(:,1); %top singular vector of MShat, normalized - s x 1
    case 'power'
        v1 = svd_power(MShat);
end

v = zeros(n,1);
v(Shat,1) = v1;
x_init = phi*v; %ensures that the energy/norm of the initial estimate is close to actual
x = x_init;


end