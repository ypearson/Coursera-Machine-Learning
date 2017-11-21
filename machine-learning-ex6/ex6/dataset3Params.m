function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
c_test     = [0.01 0.03 0.1 0.3 0.6 1 3 10 30 100];
sigma_test = [0.01 0.03 0.1 0.3 1 3 10 30 100 300];

c_test = 0.6:0.002:3;

c_test = [0.6];
sigma_test = [0.1];

test_size_c     = size(c_test,2);
test_size_sigma = size(sigma_test,2);
error_min = 1e9;
c_min = 0;
sigma_min = 0;

for i=1:test_size_c

    for j=1:test_size_sigma

        model = svmTrain(X, y, c_test(i), @(x1, x2) gaussianKernel(x1, x2, sigma_test(j)));
        predictions = svmPredict(model, Xval);

        if( mean(double(predictions ~= yval)) < error_min )
            error_min = mean(double(predictions ~= yval))
            c_min = c_test(i)
            sigma_min = sigma_test(j)
        endif
    end
end

C = c_min
sigma = sigma_min

% =========================================================================

end
