function [C,L,U] = SpectralClustering(W, k, Type)
%SPECTRALCLUSTERING Executes spectral clustering algorithm
%   Executes the spectral clustering algorithm defined by
%   Type on the adjacency matrix W and returns the k cluster
%   indicator vectors as columns in C.
%   If L and U are also called, the (normalized) Laplacian and
%   eigenvectors will also be returned.
%
%   'W' - Adjacency matrix, needs to be square
%   'k' - Number of clusters to look for
%   'Type' - Defines the type of spectral clustering algorithm
%            that should be used. Choices are:
%      1 - Unnormalized
%      2 - Normalized according to Shi and Malik (2000)
%      3 - Normalized according to Jordan and Weiss (2002)
%
%   References:
%   - Ulrike von Luxburg, "A Tutorial on Spectral Clustering", 
%     Statistics and Computing 17 (4), 2007
%
%   Author: Ingo Buerk
%   Year  : 2011/2012
%   Bachelor Thesis

% calculate degree matrix
degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

% compute unnormalized Laplacian
L = (D - W) * 1.0;

% compute normalized Laplacian if needed
switch Type
    case 2
        % avoid dividing by zero
        degs(degs == 0) = eps;
        % calculate inverse of D
        D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
        
        % calculate normalized Laplacian
        L = D * L;
    case 3
        % avoid dividing by zero
        degs(degs == 0) = eps;
        % calculate D^(-1/2)
        D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
        
        % calculate normalized Laplacian
        L = D * L * D;
end

% compute the eigenvectors corresponding to the k smallest
% eigenvalues
diff   =  100*eps;
OPTS.tol = 1e-3;
OPTS.issym = 0; 
OPTS.isreal = 1;
OPTS.maxit = 50;
[U, ~] = eigs(L,k,diff,OPTS);
c1 = size(U,2);
UU = U;
UU(:,find(all(isnan(UU))))=[];
c2 = size(UU,2);
if c2~=c1
    [UUU, ~] = eigs(L,15,diff,OPTS);
    UUU(:,find(all(isnan(UUU))))=[];
    U = UUU(:,1:k).*1.0;  
end
% in case of the Jordan-Weiss algorithm, we need to normalize
% the eigenvectors row-wise
if Type == 3
    %rdivide: right array divide
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
end

% now use the kmeans++ algorithm to cluster U row-wise
[C,~] = kmeansnew(U',k);

end