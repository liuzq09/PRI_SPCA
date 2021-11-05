% This code was adpated from the corresponding code downloaded from https://github.com/GauriJagatap/model-copram

%% The initialization step of SPARTA
function [x,p,x_init] = SparTAF_init(y_abs,A,s,z)
%updated 5/31/2017

%% initialize parameters
[m, n] = size(A);
%If ground truth is unknown
if nargin < 4
    z = zeros(n,1);
end
error_hist(1,1) = 1;
error_hist(1,2) = 1;
Marg = zeros(1,n); %marginals
MShat = zeros(s); %truncated correlation matrix
AShat = zeros(m,s); %truncated sensing matrix
y_abs2 = y_abs.^2;
phi_sq = sum(y_abs2)/m;
phi = sqrt(phi_sq); %signal power
%SPARTA parameters
% mu = 1;
% gamma = 0.7;

%% s-Truncated sensing vectors

%signal marginals
Marg = ((y_abs2)'*(A.^2))'/m; % n x 1
[Mg MgS] = sort(Marg,'descend');
S0 = MgS(1:s); %pick top s-marginals
Shat = sort(S0); %store indices in sorted order
%supp(Shat) = 1; figure; plot(supp); %support indicator
AShat = A(:,Shat); % m x s %sensing sub-matrix

%% Truncated measurements
card_Marg = ceil(m/6);
%large measurements - amplitude flow
for i=1:m
    M_eval(i) = y_abs(i)/norm(AShat(i,:));
end 
[Mm MmS] = sort(M_eval,'descend');
Io = MmS(1:card_Marg); %indices between 1 to m

%% Initialize x
%compute top singular vector according to thresholded sensing vectors and large measurements
for i = 1:card_Marg
    ii = Io(i);
    MShat = MShat + (y_abs2(ii))*AShat(ii,:)'*AShat(ii,:); % (s x s)
end

svd_opt = 'svd'; %more accurate, but slower for larger dimensions
svd_opt = 'power'; %approximate, faster

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
p = sign(A*x);

%endfunction [ output_args ] = untitled5( input_args )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here


end

