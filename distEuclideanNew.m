function [ dist ] = distEuclideanNew( M, N, Weight )

dist = sqrt(sum(Weight(1:end)*(M - N) .^ 2, 1));

end

