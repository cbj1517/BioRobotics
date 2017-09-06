function [ phi ] = trainNN( data, y, centroids, sigmas )
%find the weight matrix W of the NN, aka train the NN to map the input
% data space through the hidden data space, to the output data space
clc

if (size(centroids)~=size(sigmas))
    error('size of centroids should be equal to size of sigmas')
end

numCent = size(centroids,1);

W = zeros(numCent,1);

%get phi matrix
[n, phi] = simNN(data, centroids, W, sigmas);

%get weight matrix
%W = pinv(phi)*y;

end

