function [ percentError ] = testNN( testData, testY, centroids, W, sigmas )
%first simNN using testData
%take in n = phi*W from simNN
%compare to yTest and report errors

[ n, phi ] = simNN( testData, centroids, W, sigmas );



% for i = 1:size(n, 1)
%     if n(i,1) == 0
%         n(i,1) = 1;
%     else
%         n(i,1) = -1;
%     end
% end

for i = 1:size(n, 1)
    if n(i,1) >= 0
        n(i,1) = 1;
    else
        n(i,1) = -1;
    end
end

err = 0; 
for i = 1:size(testY, 1)
    if n(i,1) ~= testY(i,1)
        err = err+1; 
    end
end

percentError = err/size(testY,1);

end

