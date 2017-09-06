function [ trainDataOut, trainYout, testDataOut, testYOut, centroidsOut, sigmasOut ] = varKmeans( numCent, iters )
clc

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%read in data from csv file
data = csvread('Hw2CancerData.csv');

%store classifier column
y = data(:,11);
%change classification from 2 & 4 to +/- 1
y = y-3;
%remove classifier column and patient # column from data
data = data(:,2:size(data,2)-1);
[rows, columns] = size(data);

%normalize data and centroids from 0 to 1 
maxCol = max(data, [], 1);      %finds max for each column
minCol = min(data, [], 1);      %finds min for each column

%randomly generate centroids based on input
%centroids = rand(numCent,9);
centroids =[0.7354    0.5450    0.5291    0.1481    0.3757    0.7513    0.3862    0.8201    0.0582;
    0.5069    0.7778    0.8889    0.6181    0.5208    0.9514    0.7014    0.6042    0.0764;
    0.6825    1.0000    1.0000    0.8889    0.8095    1.0000    0.8730    0.4921    0.5079;
    0.7569    0.8542    0.7778    0.4306    0.6111    0.2917    0.5903    0.6111    0.1181;
    0.7778    0.9365    0.7937    0.6508    0.9841    0.5714    0.5238    0.9206    0.9683;
    0.5648    0.4722    0.4537    0.7130    0.5741    0.9537    0.5926    0.6759    0.3704;
    0.5517    0.2835    0.3295    0.2720    0.3257    0.2797    0.3410    0.2912    0.1034;
    0.1854    0.0181    0.0312    0.0266    0.1156    0.0227    0.1247    0.0147    0.0091;
    0.7267    0.3604    0.4144    0.5526    0.2823    0.9610    0.4384    0.2042    0.0811];  %0.0088 misclassification
for i = 1:columns
    data(:,i) = (data(:,i)-minCol(i))/(maxCol(i)-minCol(i)); 
end
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
%split data into train and test data sets
trainData = data(1:0.5*rows, :);
trainY = y(1:0.5*rows, :);

testData = data(0.5*rows+1:end, :);
testY = y(0.5*rows+1:end, :);
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
for d = 1:iters
    
numData = size(trainData,1);
%Calc distance to centroids for each sample
for x = 1:numCent
    dist(x,:)=sqrt(sum((repmat(centroids(x,:),numData,1)-trainData(:,:)).^2,2));    %distance from center to point
end

%dist

%Cluster data based on distances
for q = 1:size(trainData,1)
    [C,I] = min(dist, [], 1); 
end

sigmas = zeros(numCent, 9);                 %pre-allocate with zeros
newCentroid = zeros(numCent, 9);            %pre-allocate with zeros
%cluster and calculate new centroids and variances
for t = 1:numCent
    [r, c] = find(I==t);
    tempCent = zeros(size(c,2),9);      %size tempCent correctly so there are no extra rows
    for p = 1:size(c,2)
        tempCent(p,:) = trainData(c(p),:);
    end
    newCentroid(t,:) = sum(tempCent,1).*(1/size(tempCent,1));
    sigmas(t,:) = var(tempCent,1);   
end
centroids = newCentroid;         %update centroids
%d                               %iteration number
end

%only output data if it is requested i.e. [x, y] = f(x)
if nargout > 0
    trainDataOut = trainData;
    trainYout = trainY;
    testDataOut = testData;
    testYOut = testY;
    centroidsOut = centroids;
    sigmasOut = sigmas;
end

end

