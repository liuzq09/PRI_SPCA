% This code was adpated from the corresponding code downloaded from https://github.com/GauriJagatap/model-copram

%% The initialization step of ThWF
function [x,p,x_init] = Thresholded_WF_init(y_twf,A,z)
%updated 5/31/2017

%% initialize parameters
[m,n] = size(A);
%If ground truth is unknown
if nargin < 3
    z = zeros(n,1);
end
Marg = zeros(1,n); %marginals
phi_sq = sum(y_twf)/m;
phi = sqrt(phi_sq); %signal power
%Thresholded WF parameters
alpha = 0.1;
%mu = 0.23; 
thres_param = (1+alpha*sqrt(log(m*n)/m))*phi_sq ;

%% Thresholded sensing vectors

%signal marginals
Marg = (y_twf'*(A.^2))'/m; % n x 1
S0 = find(Marg>thres_param);
ss = length(S0);
Shat = sort(S0);
MShat = zeros(ss);
AShat = zeros(m,ss);
%supp(Shat) = 1; figure; plot(supp); %support indicator
AShat = A(:,Shat); % m x ss

%compute top singular vector according to thresholded sensing vectors
for i = 1:m
    MShat = MShat + (y_twf(i))*AShat(i,:)'*AShat(i,:); % (ss x ss)
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

%endfunction [ output_args ] = untitled4( input_args )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


end

