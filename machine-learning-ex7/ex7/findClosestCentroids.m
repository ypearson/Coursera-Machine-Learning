function idx = findClosestCentroids(X, centroids)

%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K

K   = size(centroids, 1);
idx = zeros(size(X,1), 1);
num_of_examples = size(X,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
min_norm_vec = [0 0 0];

for i=1:num_of_examples

    for k=1:K
       min_norm_vec(k) = norm( X(i,:) - centroids(k,:) ) ^ 2;
    end
    [min_norm idx(i)] = min(min_norm_vec);

end
% =============================================================
end

