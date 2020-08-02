close all;
clear all;
clc;

% %A Study of Recurrent neural
% networks (RNNs) in univariate and
% multivariate time series


load henondata x;

Z = x;
NumberOfInputs = 1;
LengthOfTimeSeries =5000;
PredictHorizon = 100;
lambda = 0.0;

X = []; y = []; k = 0; loop = 1;
while loop
    k = k + 1;
    X = [X; x(k+0:k+NumberOfInputs-1)];
    y = [y; x(k+NumberOfInputs)];
    if k+NumberOfInputs >= LengthOfTimeSeries; loop = 0; end
end


num_data = length(y);

train_index = 1:2:num_data;
val_index = 2:2:num_data;
train_in = X(train_index, :);
train_out = y(train_index, :);
val_in = X(val_index, :);
val_out = y(val_index, :);
num_trdata = size(train_in, 1);
num_valdata = size(val_in, 1);


hidden_units = 10;
d = 1;
q = 1;
tau = 4; % tBPTT constant for unfolding

alpha = 0.01;


Whh = rand(hidden_units,hidden_units);
Wih = rand(hidden_units,d);
Who = rand(q,hidden_units);

MAX_EPOCH = 5000;





EPOCH = 0;

while EPOCH <= 1
    o = [];
    zt = [];
    ht = zeros(hidden_units, 1);
    count = 0;
    for K = 1:100

        [o, ht, zt] = forward(train_in, Wih, Whh, Who, ht, zt, o, K); 
                   
        [dWih, dWhh, dWho, del_t] = backprop(Wih, Whh, Who, ht, zt, o, train_out, train_in, K);

    end
    EPOCH = EPOCH + 1;
end






