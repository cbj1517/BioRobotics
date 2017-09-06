function [ bestW, pbError, gbError, allGB ] = PSO( iters, pop, weights )
clc
%Define constants
K = 0.729;
c1 = 0.5;
c2 = 3.6;
maxW = 5; 
minW = -2;
%Define variables
z = 0;

%initialize matrices
Xkp1 = zeros([weights, pop]);   %init next particle position matrix
Xcurr = (maxW-minW).*rand([weights, pop]) + minW;   %init velocity matrix
%vel = (maxW-minW).*rand([weights, pop]) + minW;     %init velocity matrix
PB = zeros(weights, pop);       %init personal best weights
GB = zeros(weights, 1);         %init global best weight
pbError = ones(1,pop);          %init personal best errors      
gbError = ones(1,1);          %init global best error 


[ trainDataOut, trainYout, testData, testY, centroids, sigmas ] = varKmeans(weights,2);

while z < iters
    for i = 1:pop
        [ phi ] = trainNN(trainDataOut, trainYout, centroids, sigmas*10);
        [ percentError ] = testNN( testData, testY, centroids, Xcurr(:,i), sigmas*10);
        
        %check current error against PB error
        if percentError < pbError(1,i)      
            pbError(1,i) = percentError;    %update personal best error
            PB(:,i) = Xcurr(:,1);           %update personal best weights
        end
    end
    
    %check for new global best weights
    minError = min(pbError);                            %find smallest personal best errors
    if minError < gbError
        gbError = minError;                             %update global best error
        index = find(pbError == min(min(pbError)));     %find index of min error
        GB = PB(:,index);                               %update global best weights
    end
    allGB(1,z+1) = gbError;    
    %update particle position matrix

    for r = 1:pop
        psi1 = rand;
        psi2 = rand;
        Xkp1(:,r) = K.*(Xkp1(:,r) + (c1*psi1).*( PB(:,r) - Xcurr(:,r)) + (c2*psi2).*(GB(:,1) - Xcurr(:,r)));   
    end
    
    %vel = Xcurr;               %update velocity
    Xcurr = Xcurr+Xkp1;         %update current position
   
    z = z+1; 
end
bestW = GB;
%allGB
end

