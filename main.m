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
% PredictHorizon = 100;
% LengthOfTimeSeries = length(t) - PredictHorizon;
% x = sin(t);

% Z = x;
% Zmin = min(Z); Zmax = max(Z);

% data = (x - [ones(size(x, 1), 1) * Zmin]) ./ ([ones(size(x, 1), 1) * Zmax] - [ones(size(x, 1), 1) * Zmin]);
% x_test = data(LengthOfTimeSeries + 1:LengthOfTimeSeries + PredictHorizon);

load henondata x;
NumberOfInputs = 1;
LengthOfTimeSeries =501;
PredictHorizon = 50;
data = x(1:LengthOfTimeSeries);
lambda = 0.0;
Z = x;
Zmin = min(Z); Zmax = max(Z);
data = (x - [ones(size(x, 1), 1) * Zmin]) ./ ([ones(size(x, 1), 1) * Zmax] - [ones(size(x, 1), 1) * Zmin]);
x_test = data(LengthOfTimeSeries + 1:LengthOfTimeSeries + PredictHorizon);

X = []; y = []; k = 0; loop = 1;
while loop
    k = k + 1;
    X = [X; data(k+0:k+NumberOfInputs-1)];
    y = [y; data(k+NumberOfInputs)];
    if k+NumberOfInputs >= LengthOfTimeSeries; loop = 0; end
end
x_test_in = []; x_test_out = []; k = 0; loop = 1;
while loop
    k = k + 1;
    x_test_in = [x_test_in; x_test(k + 0:k + NumberOfInputs - 1)];
    x_test_out = [x_test_out; x_test(k + NumberOfInputs)];
    if k + NumberOfInputs >= length(x_test); loop = 0; end
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





% N = 8 ;
d = 1;
q = 1;
tau = 5; % tBPTT constant for unfolding






stop_condition = 1e-7;

fValBest = Inf;

N_max = num_trdata-1;

w_limit = sqrt(6/(2*num_trdata));

for N = 1:N_max
    disp(N)
    ht = zeros(N, 1);

    % Whh = rand(N, N);  
    % Wih = rand(N, d);
    % Who = rand(q, N);
  
    learning_rate = 0.001;
    %GLOROT UNIFROM WEIGHT INITILIZATION
    Whh = unifrnd(-w_limit, w_limit, [N, N]);
    Wih = unifrnd(-w_limit, w_limit, [N, d]);           
    Who = unifrnd(-w_limit, w_limit, [q, N]);
    Wih_old = Wih;
    Whh_old = Whh;
    Who_old = Who;
    
    
    ht = zeros(N, 1); zt = []; o = [];
    for k = 1:num_trdata
        [o, ht, zt] = forward(train_in, Wih, Whh, Who, ht, zt, o, k);
    end
    
    n = 2;
    n_max = 500;
    err = zeros(n_max,1);
    err(1) = sum((o - train_out).^2) / length(train_out);
    err_old = err(1);
    
    count = 0;
    if N < 2
        n_max = 200;
    end
    while n < n_max
        
        
        
        

        [dWih, dWhh, dWho, del_t] = bptt(Wih, Whh, Who, ht, zt, o, train_out, train_in, tau, k);
        [Wih, Whh, Who] = gradDes(Wih, Whh, Who, dWih, dWhh, dWho, learning_rate);

        
        
        
        % ht_val = zeros(N, 1); zt_val = []; o_val = [];
        % for k = 1:num_valdata
        %      [o_val, ht_val, zt_val] = forward(val_in, Wih, Whh, Who, ht_val, zt_val, o_val, k);
        % end
        

    
        o = []; zt = []; ht = zeros(N, 1);

        for K = 1:num_trdata
            [o, ht, zt] = forward(train_in, Wih, Whh, Who, ht, zt, o, K);
        end
    
        
        err(n) = sum((o - train_out).^2) / length(train_out);
        
        fprintf(" MSE : %f \n", err(n));
        
        figure(1)
        plot(o)
        hold on
        plot(train_out)
        hold off
        
     
        if (err(n) > err_old)
            
            learning_rate = learning_rate/2;
            count = count + 1;
            Wih = Wih_old;
            Whh = Whh_old;
            Who = Who_old;
            
        elseif (isnan(err(n)))
            disp('NaNIIIIII!!!!!!')
            % ht = zeros(N, 1);
            % Whh = rand(N, N);  
            % Wih = rand(N, d);
            % Who = rand(q, N);
            break
        else
            count = 0;
            err_old = err(n);
            Wih_old = Wih;
            Whh_old = Whh;
            Who_old = Who;
            
            
        end
        if count == 100
            n = n_max;
        end
        n = n + 1;
        

    end
    
    
    
    
    count = 0;
    o = []; zt = []; ht = zeros(N, 1);
    for k = 1:num_trdata
        [o, ht, zt] = forward(train_in, Wih, Whh, Who, ht, zt, o, k);
    end

    o_val = []; zt_val = []; ht_val = zeros(N, 1);
    for k = 1:num_valdata
        [o_val, ht_val, zt_val] = forward(val_in, Wih, Whh, Who, ht_val, zt_val, o_val, k);
    end
    err_train(N) = sum((o - train_out).^2) / length(train_out);
    err_val(N) = sum((o_val - val_out).^2) / length(o_val);

    if (err_val(N) < fValBest)
        count = count + 1;
        N_best = N;
        fValBest = err_val(N);
        Wih_best = Wih;
        Whh_best = Whh;
        Who_best = Who;


    elseif count == 10
        break
    end

    if err_train < stop_condition
        break
    end




   
    
    fprintf("MSE TRAINING: %f  MSE VALIDATION: %f \n", err_train(N), err_val(N));
end


% for k = 1:length(x_test_in)

%     [o, ht, zt] = forward(x_test_in, Wih, Whh, Who, ht, zt, o, k);

% end
% figure(2)
% plot(o)
% hold on
% plot(x_test_out)
% hold off

