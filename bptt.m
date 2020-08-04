function [dWih, dWhh, dWho,del_t] = bptt(Wih, Whh, Who, ht, zt, o, train_out, train_in, K)
    
    e_t = 2 * (o - train_out(1:K));
    del_t = []; %columns are t backwards through time.
    ind = 0;
    dWih = 0;
    dWhh = 0; 
    dWho = 0;
    t = K;  
    while t > 0
       

        if t == K 
            del_t = [del_t, Who'* e_t(t) .* dtanh(zt(:,t))];
            
        else
            del_t = [del_t, (Who'*e_t(t)+Whh'*del_t(:,ind)) .* dtanh(zt(:,t))];
           
        end
        ind = ind + 1;
        t = t - 1;

    end
    
    del_rev = fliplr(del_t);

    for i = 1:K
        dWih = dWih + del_rev(:, i) * train_in(i);

        dWhh = dWhh + del_rev(:,i)*ht(:,i)';

        dWho = dWho + e_t(i) * ht(:, i + 1)';
    end
     
    
end


