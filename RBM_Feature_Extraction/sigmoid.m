function x = sigmoid(x)
%SIGMOID Compute sigmoid function: y = 1 / (1 + exp(-x))
%

max0 = max(x, 0);
x = exp(x-max0) ./ (exp(x-max0) + exp(-max0));

end

