% This code was adpated from the corresponding code downloaded from https://github.com/GauriJagatap/model-copram

% To give output vectors about relative error and running time for each
% iteration in the subsequent iterative algorithm of CoPRAM
function [x,err_vec,time_vec] =  CoPRAM_GD_time(y_abs,x_init,A,s,max_iter,z)
%%updated 5/31/2017

%% initialize parameters
[m,n] = size(A);
x = x_init;

time_vec = zeros(1,max_iter);
err_vec = zeros(1,max_iter);

t0=cputime;

%% start descent
fprintf('\n#iter\t|y-Ax|\t\t|x-z|\n')
for t=1:max_iter 
    p = sign(A*x);
    x = cosamp(p.*y_abs/sqrt(m), A/sqrt(m), s,10,x); %Its = 10
    err_vec(t) = min(norm(x-z),norm(x+z))/norm(z);;
    time_vec(t) = cputime - t0;
end
end