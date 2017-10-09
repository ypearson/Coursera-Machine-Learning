function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
n = size(theta);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% X * theta = [ t0*x0(1) + t1*x1(1) + tn+1*xn+1(1)]
%             [ t0*x0(2) + t1*x1(2) + tn+1*xn+1(2)]
%             ...
%             [ t0*x0(m) + t1*x1(m) + tn+1*xn+1(m)]
X*theta;

z = sigmoid( X*theta );

J = -(1/m)*sum(y.*log( z ) + (1-y).*log(1-z)) + lambda/(2*m) * sum ( theta([2:n]).^2);

theta0_index = 1;
grad(theta0_index) = ( 1/m ) * sum ( (z-y) .* X(:,theta0_index) );

for i = 2:n
    grad(i) = ( 1/m ) * sum ( (z-y) .* X(:,i) ) + (lambda/m) * theta(i);
endfor


% =============================================================

end
