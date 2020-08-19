function [dWih, dWhh, dWho,del_t] = bptt(Wih, Whh, Who, ht, zt, o, train_out, train_in, tau,K)
    
    % e_t = 2*(o - train_out(1:K));
    e_t = 2*((o-train_out(1:K)).*dsigmoid(o));
    del_t = []; %columns are t backwards through time.
    ind = 0;
    dWih = 0;
    dWhh = 0; 
    dWho = 0;
    t = K;  
    if K < tau
        stop_cond = 0;
    else
        stop_cond = K - tau;
    end
    while t > stop_cond
       

        if t == K 
            del_t = [del_t, Who'* e_t(t) .* dtanh(zt(:,t))];
            
        else
            del_t = [del_t, (Who'*e_t(t)+Whh'*del_t(:,ind)) .* dtanh(zt(:,t))];
           
        end
        
        ind = ind + 1;
        t = t - 1;
        

    end

    del_rev = fliplr(del_t);
    % size(del_rev)
    indx = 1;
    for i = t+1:K

        dWih = dWih + del_rev(:,indx) * train_in(i);

        dWhh = dWhh + del_rev(:,indx)*ht(:,i)';

        dWho = dWho + e_t(i) * ht(:, i+1)';
        indx = indx + 1;
    end
     
    
end


