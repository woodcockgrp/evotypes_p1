function [amalgamated_weights, feature_names] = amalgamate_features(input, inputnames)
% AMALGAMATE_FEATURES Takes in weight matrices and returns amalgamated weights and feature names.
%
% INPUT:
%   input       - The weight matrices either in a cell array, matrix, or a file name as a string.
%   inputnames  - Names of the inputs
%
% OUTPUT:
%   amalgamated_weights - Resulting weights after amalgamation.
%   feature_names       - Names of the inputs in each feature given in the same order as amalgamated_weights.

    % Determine the type of input and load data accordingly
    if ischar(input)
        load(input);
        full_weights = [weight_mat{:}];
    elseif iscell(input)
        weight_mat = input;
        n_weights = size(weight_mat, 2);
        full_weights = [sparse([weight_mat{1:n_weights/2}]), sparse([weight_mat{(n_weights/2+1):n_weights}])]';
    elseif size(input, 2) > 1
        full_weights = input;
    else
        error('Input type not recognised');
    end

    n_inputs = size(full_weights, 2);
    max_weight_length = max(max(full_weights));

    % Filter out features with low influence weights
    full_weights(full_weights < 0.001 * max_weight_length) = 0;
    full_weights = full_weights .* (full_weights > 0.5 * max(full_weights, [], 1));
    full_weights(all(full_weights == 0, 2), :) = [];

    similarity_threshold = 0.5;
    mean_weights_per_input = zeros(size(full_weights, 2));

    % Compute mean weights per input
    for i = 1:n_inputs
        weights_for_input = full_weights(full_weights(:, i) > similarity_threshold * max(full_weights(:, i)), :);
        mean_weights_per_input(i, :) = mean(weights_for_input);
    end

    % Filter weights again based on threshold
    mean_weights_per_input(mean_weights_per_input < 0.1 * max_weight_length) = 0;
    mean_weights_per_input(all(mean_weights_per_input, 2), :) = [];
    mean_weights_per_input(all(mean_weights_per_input == 0, 2), :) = [];

    cosine_mat = squareform(pdist(mean_weights_per_input, 'cosine'));
    amalgamated_weights = [];

    % Amalgamate similar weights
    while(size(cosine_mat, 2) > 0)
        similar_weights = find(cosine_mat(1, :) < similarity_threshold);
        amalgamated_weights = [amalgamated_weights; mean(mean_weights_per_input(similar_weights, :), 1)];

        % Remove the weights that have been amalgamated from the analysis
        cosine_mat(:, similar_weights) = [];
        cosine_mat(similar_weights, :) = [];
        mean_weights_per_input(similar_weights, :) = [];
    end

    n_features = size(amalgamated_weights, 1);
    maxpos = zeros(1, n_features);

    % Find position of maximum weights
    for i = 1:size(amalgamated_weights, 1)
        [~, maxpos(i)] = find(amalgamated_weights(i, :) == max(amalgamated_weights(i, :), [], 2), 1, 'first');
    end

    % Sort the weights based on position
    [~, neworder] = sort(maxpos);
    amalgamated_weights = amalgamated_weights(neworder, :);
    amalgamated_weights(amalgamated_weights < 0) = 0;

    % If input names are provided, match them with features
    if nargin > 1       
        for k = 1:n_features
            input_pos = find(amalgamated_weights(k, :));
            [~, input_order] = sort(amalgamated_weights(k, input_pos), 'descend');
            feature_names{k} = inputnames(input_pos(input_order));
        end
    end
end