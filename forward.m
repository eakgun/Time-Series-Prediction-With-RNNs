function [o, ht, zt] = forward(x, Wih, Whh, Who, bh, ht, zt, o, K)
    
    

    %ht_0 0 lardan olusan bir vector bu sebepten dolayi
    %ht 1 sutun fazla basladi. Yani kodlarken literaturde "ht-1 olan bu kodda ht" 
    %ve dolayisiyla "ht ise ht+1" oldu. Sadece index kolayligi olmasi icin.

    zt = [zt, Wih*x(K) + Whh*ht(:,K) + bh];

    ht = [ht tanh(zt(:,K))];
    % ht = [ht sigmoid(zt(:,K))];

    o = [o; Who * ht(:, K + 1)];
    % o = [o; sigmoid(Who * ht(:, K + 1))];



end

% Lt = (o-y).^2