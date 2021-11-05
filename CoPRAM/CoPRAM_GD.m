% This code was adpated from the corresponding code downloaded from https://github.com/GauriJagatap/model-copram

%% The subsequent iterative step in CoPRAM
function x =  CoPRAM_GD(y_abs,x_init,A,s,max_iter)
%%updated 5/31/2017

%% initialize parameters
[m,n] = size(A);
p = zeros(m,1); %phase vector

x = x_init;

%% start descent
fprintf('\n#iter\t|y-Ax|\t\t|x-z|\n')
for t=1:max_iter 
    p = sign(A*x);
    x = cosamp(p.*y_abs/sqrt(m), A/sqrt(m), s,10,x); %Its = 10
end


end