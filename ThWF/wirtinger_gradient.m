% This code was downloaded from https://github.com/GauriJagatap/model-copram

function gradf_x = wirtinger_gradient(x,y,A)   
    [m,n] = size(A);
    yy = A*x; 
    coeff = (yy.^2-y).*(yy);
    gradf_x = A'*coeff/m;
end