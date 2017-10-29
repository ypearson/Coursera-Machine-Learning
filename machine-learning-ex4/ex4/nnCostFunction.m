function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 5000 by 401
% X =  [ones(m,1),X];

% (5000 by 401) * T(25 by 401) = 5000 by 25
zLayer2              = [ones(m,1),X]*Theta1';
activationUnitLayer2 = sigmoid( zLayer2  );

% Add bias unit 5000 by 26
activationUnitLayer2 = [ones(m,1),activationUnitLayer2];

% (5000 by 26) * T(10 by 26) = 5000 by 10
zLayer3              = activationUnitLayer2*Theta2';
activationUnitLayer3 = sigmoid( zLayer3 );

% Rename to output layer
hTheta = activationUnitLayer3;

label=1:num_labels;

% Cost Function of NN
J = -(1/m) *  sum( sum(  ( (y==label) .* log( hTheta(:,label) )) + ((1 - (y==label) ) .* log( 1 - hTheta(:,label)))  ) );

% isolate non-bias units
nonBiasUnits1 = 2:size(Theta1)(2);
nonBiasUnits2 = 2:size(Theta2)(2);

% Regularization of Cost Function
J = J + (lambda/(2*m)) * (   sum(sum( Theta1(:, nonBiasUnits1 ) .^ 2) )   +   sum(sum( Theta2(:, nonBiasUnits2 ) .^ 2) ) );

fprintf('Starting BP....\n  ');

Delta1 = zeros( hidden_layer_size, input_layer_size + 1);
Delta2 = zeros( num_labels, hidden_layer_size + 1);

for i=1:m

    % Forward propagation

    % 1 x 401   *   T(25 x 401) = 1 x 25
    a1 = [1, X(i,:)];
    z2 = a1 * Theta1';
    a2 = sigmoid ( z2 );
    % 1 x 26
    a2 = [1, a2];

    % 1 x 26    *   T(10 x 26) = 1 x 10
    z3 = a2 * Theta2';
    a3 = sigmoid ( z3 );

    % Back propagation

    % 1 x 10
    yVec       = zeros(1,num_labels);
    yVec(y(i)) = 1;
    delta3     = a3 - yVec;

    % 1x10 * 10x26 = 1x26
    % 1x26 o 1x26 o 1x26 = 1x26
    delta2 = (delta3 * Theta2) .* ( a2 .* (1 - a2));

    % delta2 = (delta3 * Theta2) .* sigmoidGradient([1,z2]);

    % remove delta2(0)
    % 1 x 25
    delta2 = delta2(2:end);

    % 25x1 x 1x401 = 25 x 401
    Delta1 = Delta1 + delta2' * a1;

    % 10x1 x  1x26 = 10x26
    Delta2 = Delta2 + delta3' * a2;

endfor

Theta1Rows = size(Theta1)(1)
Theta2Rows = size(Theta2)(1)

Theta1_grad = (1/m)*Delta1 + (lambda/m)*[ zeros(Theta1Rows,1), Theta1(:,2:end) ];
Theta2_grad = (1/m)*Delta2 + (lambda/m)*[ zeros(Theta2Rows,1), Theta2(:,2:end) ];

% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
