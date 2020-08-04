close all;
clear all;
clc;

% %A Study of Recurrent neural
% networks (RNNs) in univariate and
% multivariate time series


%x_test_in ve x_test_out halledilicek.


t = [1:0.1:100];
LengthOfTimeSeries = length(t);
NumberOfInputs = 1;
PredictHorizon = 100;
x = sin(t);
Z = x;
Zmin = min(Z); Zmax = max(Z);

x = (x - [ones(size(x, 1), 1) * Zmin]) ./ ([ones(size(x, 1), 1) * Zmax] - [ones(size(x, 1), 1) * Zmin]);


% load henondata x;
% Z = x;
% NumberOfInputs = 1;
% LengthOfTimeSeries =900;
% PredictHorizon = 50;
% lambda = 0.0;
% Z = x;
% Zmin = min(Z); Zmax = max(Z);
% data = (x - [ones(size(x, 1), 1) * Zmin]) ./ ([ones(size(x, 1), 1) * Zmax] - [ones(size(x, 1), 1) * Zmin]);
% x = data(1:LengthOfTimeSeries);
% x_test = data(LengthOfTimeSeries + 1:LengthOfTimeSeries + PredictHorizon);

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

learning_rate = 0.001;


Whh = rand(hidden_units,hidden_units);
Wih = rand(hidden_units,d);
Who = rand(q,hidden_units);

MAX_epoch = 5000;
stop_condition = 1e-4;

mse_val_prev = Inf;

ht = zeros(hidden_units, 1);
epoch = 0;
Wih_saved = [];
Whh_saved = [];
Who_saved = [];

while epoch < MAX_epoch
    o_val = [];
    zt_val = [];
    o = [];
    zt = [];
    Wih_saved = Wih;
    Whh_saved = Whh;
    Who_saved = Who;
    count = 0;
    for K = 1:num_trdata-1

        [o, ht, zt] = forward(train_in, Wih, Whh, Who, ht, zt, o, K); 
                   
        [dWih, dWhh, dWho, del_t] = bptt(Wih, Whh, Who, ht, zt, o, train_out, train_in, K);

        [Wih, Whh, Who] = gradDes(Wih, Whh, Who, dWih, dWhh, dWho, learning_rate);

        [o_val ht_val, zt_val] = forward(val_in, Wih, Whh, Who, ht, zt_val, o_val, K);
        
    end
    
    
    

    mse_train = sum((o - train_out(1:K)).^2) / K;
    disp(mse_train)
    mse_val = sum((o_val - val_out(1:K)).^2) / K;
    disp(mse_val)
    ht = ht(:, end);
    
    if mse_val > mse_val_prev
        count = 0;
        learning_rate = learning_rate/2;
        disp('1')
        Wih = Wih_saved;
        Whh = Whh_saved;
        Who = Who_saved;


    else
        count = count + 1;
        epoch = epoch + 1;
        disp('2')
    end
    
    if  epoch == MAX_epoch
        disp('3')
        epoch = MAX_epoch;
        ht = ht(:,end);
        zt_test = [];
        o_test = [];
        for Step = 1:length(x_test)
            [o_test, ht, zt_test] = forward(x_test, Wih, Whh, Who, ht, zt_test, o_test, Step);
        end
        L_test = sum((o_test - x_test_out').^2) / length(x_test_out);

    end
    mse_val_prev = mse_val;
    
end






