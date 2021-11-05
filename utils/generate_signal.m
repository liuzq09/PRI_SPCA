% This code was adpated from the corresponding code downloaded from https://github.com/GauriJagatap/model-copram

function [z,z_ind] = generate_signal(n,K)
z = zeros(n,1);
z_ind = randperm(n,K);
z(z_ind) = randn(K,1); %generate K sparse signal in n-dimensional