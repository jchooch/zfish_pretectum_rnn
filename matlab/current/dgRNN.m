% dgRNN.m
% An in silico toy data-generating RNN

function [activities, number_of_neurons] = dgRNN(weight_matrix, inputs, number_of_timesteps)
    if ~exist('inputs', 'var') % check if there are inputs (i.e. stimuli)
        disp('No inputs provided, so running without inputs.')
    end
    if length(size(weight_matrix)) ~= 2 || size(weight_matrix, 1) ~= size(weight_matrix, 2)
        disp('Input weight matrix is wrong size. It must be 2D and square.')
        return   % halt if input weight matrix is wrong size
    end
    number_of_neurons = length(weight_matrix);
    activities = nan(N, number_of_timesteps);
end

%% Pseudocode
%{
1. Input: weight_matrix, inputs
2. Forward pass: 
3. 
%}