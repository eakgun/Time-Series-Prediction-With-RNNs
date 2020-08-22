function [dWih, dWhh, dWho, dbh, dz_t] = bptt(Wih, Whh, Who, ht, zt, o, train_out, train_in, tau, T)
    
    do_t = 2*(o - train_out(1:T)); % dL/do_t cikisin amac fonk göre türevi.
    % do_t = 2*((o-train_out(1:K)).*dsigmoid(o));
    dz_t = []; %columns are t backwards through time.
    ind = 0;
    dWih = 0;
    dWhh = 0;
    dWho = 0;
    dbh = 0;
    dbo = 0;
    t = T;  
    % if K < tau
    %     stop_cond = 0;
    % else
    %     stop_cond = K - tau;
    % end
    while t > 0
       

        if t == T 
            dz_t = [dz_t, Who'* do_t(t) .* dtanh(zt(:,t))]; % Who'*do_t = dht sadece T adiminda bu yapilicak
            
        else
            dz_t = [dz_t, (Who'*do_t(t)+Whh'*dz_t(:,ind)) .* dtanh(zt(:,t))]; % T --> t_1 giderken 
           % yani zamanda geriye giderken arkadakilerde hesaba katilir, ex: dz_T-1 hesabinda dz_T vs.
        end
        ind = ind + 1;
        
        % T'den t_1 e kadar olan tum gradientler toplanir.
        dWih = dWih + dz_t(:, ind) * train_in(t);
        dWhh = dWhh + dz_t(:, ind) * ht(:, t)';
        dWho = dWho + do_t(t) * ht(:, t+1)';
        dbh = dbh + dz_t(:,ind);


        
        t = t - 1;
        

    end

    % del_rev = fliplr(dz_t);
    % % size(del_rev)
    % indx = 1;
    % for i = t+1:K

    %     dWih = dWih + del_rev(:,indx) * train_in(i);

    %     dWhh = dWhh + del_rev(:,indx)*ht(:,i)';

    %     dWho = dWho + do_t(i) * ht(:, i+1)';
    %     indx = indx + 1;
    % end
     
    
end


