function output = backprop(o, K, Whh, Who,zt_saved,ht_saved)
    
    e_t = 2 * (o - train_out(1:K));
    del_t = [];


    t = K;  
    while t > 1
       

        if t == K 
           del_t = [del_t, Who'* et(t) .* dtanh(zt_saved(:,t))];
           del_next = del_t;
        else
            del_t = (Who'*e(t)+Whh'*del_next) .* dtanh(zt_saved(:,t));
            del_next = del_t;
        end
        t = t - 1;

    end


end