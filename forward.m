function [o, ht, zt, zt_saved, ht_prev, ht_saved] = forward(x, Wih, Whh, ht_prev, o,ht_saved,zt_saved,Who, K)
    
    


    zt = Wih*x(K) + Whh*ht_prev;

    zt_saved = [zt_saved zt];

    ht = tanh(zt);

    ht_saved = [ht_saved ht];
    
    ht_prev = ht;

    o = [o; Who*ht];
    



end

% Lt = (o-y).^2