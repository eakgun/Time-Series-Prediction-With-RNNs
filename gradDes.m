function [Theta] = gradDes(learning_rate, dTheta, Theta)
    
    % Wih = Wih - learning_rate.*dWih;
    % Whh = Whh - learning_rate.*dWhh;
    % Who = Who - learning_rate.*dWho;
    Theta = Theta - learning_rate.*dTheta;

end