% This code was downloaded from https://github.com/GauriJagatap/model-copram

function x = soft_threshold(z,tau)

x = sign(z).*max(abs(z)-tau,0); %soft threshold
% x = z.*(abs(z)>tau); %hard threshold

end