% This code was downloaded from https://github.com/GauriJagatap/model-copram

%% svd_power
function max_sv = svd_power(M)
% [val,idx]=max(diag(M));
% x = zeros(size(M,1),1);
% x(idx) = 1;
[m, n] = size(M);
% x = randn(n,1);
% x = ones(n,1)/sqrt(n);
[~, col_i] = max(diag(M));
x = M(:,col_i);
for it = 1:100 % number of power iterations
   y = M*x;
   y = y/norm(y);
   if norm(x-y)/norm(x) < 1e-6
        break;
   end
   x = y;
end
   max_sv = x; 
end