function [o, ht, zt] = forward(x, Wih, Whh, Who, ht, zt, o, K)
    
    


    zt = [zt, Wih*x(K) + Whh*ht(:,K)];

    ht = [ht tanh(zt(:,K))];

    o = [o; Who*ht(:,K+1)];
    



end

% Lt = (o-y).^2