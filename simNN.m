function [ n, phi ] = simNN( data, centroids, W, sigmas )
clc

%calc number of data points
numData = size(data,1);

%calc number of centroids
numCents = size(centroids, 1);

phi = ones(numData, numCents);

for i = 1:numCents
    %distance from cent to data
    r=sqrt(sum((repmat(centroids(i,:),numData,1)-data(:,:)).^2,2)); %distance from cent to data
    
    %populate phi
    phi(:,i) = exp(-r.^2/(sigmas(i).^2));
end

n=phi*W;

end

