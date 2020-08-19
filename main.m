close all;
clear all;
clc;

% %A Study of Recurrent neural
% networks (RNNs) in univariate and
% multivariate time series

%THE PROBLEM IS THE VANISHING GRADIENT!

%x_test_in ve x_test_out halledilicek.


% t = [1:0.1:100];
% NumberOfInputs = 1;
% PredictHorizon = 50;
% LengthOfTimeSeries = length(t) - PredictHorizon;
% x = sin(t);

% Z = x;
% Zmin = min(Z); Zmax = max(Z);

% x = (x - [ones(size(x, 1), 1) * Zmin]) ./ ([ones(size(x, 1), 1) * Zmax] - [ones(size(x, 1), 1) * Zmin]);
% x_test = x(LengthOfTimeSeries + 1:LengthOfTimeSeries + PredictHorizon);

load henondata x;
NumberOfInputs = 1;
LengthOfTimeSeries =1001;
PredictHorizon = 50;
data = x(1:LengthOfTimeSeries);
lambda = 0.0;
Z = x;
Zmin = min(Z); Zmax = max(Z);
data = (x - [ones(size(x, 1), 1) * Zmin]) ./ ([ones(size(x, 1), 1) * Zmax] - [ones(size(x, 1), 1) * Zmin]);
% x_test = data(LengthOfTimeSeries + 1:LengthOfTimeSeries + PredictHorizon);

X = []; y = []; k = 0; loop = 1;
while loop
    k = k + 1;
    X = [X; data(k+0:k+NumberOfInputs-1)];
    y = [y; data(k+NumberOfInputs)];
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


hidden_units = 180;
d = 1;
q = 1;
tau = 10; % tBPTT constant for unfolding

learning_rate = 0.001;


Whh = rand(hidden_units,hidden_units);
Wih = rand(hidden_units,d);
Who = rand(q,hidden_units);

MAX_epoch = 5000;
stop_condition = 1e-7;

mse_val_prev = Inf;


epoch = 0;
Wih_saved = [];
Whh_saved = [];
Who_saved = [];
ht = zeros(hidden_units, 1);

while epoch < 5000
    
    o_val = [];
    zt_val = [];
    o = [];
    zt = [];
    Wih_saved = Wih;
    Whh_saved = Whh;
    Who_saved = Who;
    count = 0;
    
    for K = 1:num_trdata  % BU VANISHING GRADIENT E YOL ACIYOR!!!!!!!!!!!!

        [o, ht, zt] = forward(train_in, Wih, Whh, Who, ht, zt, o, K); 
        [dWih, dWhh, dWho, del_t] = bptt(Wih, Whh, Who, ht, zt, o, train_out, train_in, tau, K);
        [Wih, Whh, Who] = gradDes(Wih, Whh, Who, dWih, dWhh, dWho, learning_rate);
       
    end


    for K = 1:num_valdata
        [o_val ht_val, zt_val] = forward(val_in, Wih, Whh, Who, ht, zt_val, o_val, K);
    end
    o_best = o;
 


    mse_train = sum((o - train_out).^2) / length(o);
    mse_val = sum((o_val - val_out).^2) / length(o);
    fprintf("MSE TRAINING: %f  MSE VALIDATION: %f \n", mse_train, mse_val);
    figure(1)
    plot(o_best)
    hold on
    plot(train_out)
    hold off
    
    if mse_val > mse_val_prev || isnan(mse_val)
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
    
    if  epoch == MAX_epoch || count == 10
        disp('3')
        epoch = MAX_epoch;

        zt_test = [];
        o_test = [];
        for Step = 1:length(x_test)
            [o_test, ht, zt_test] = forward(x_test, Wih, Whh, Who, ht, zt_test, o_test, Step);
        end
        L_test = sum((o_test - x_test_out').^2) / length(x_test_out);

    end
    mse_val_prev = mse_val;
    
end






