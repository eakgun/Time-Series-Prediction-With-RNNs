function [Wih, Whh, Who] = gradDes(Wih, Whh, Who, dWih, dWhh, dWho, learning_rate)
    
    Wih = Wih - learning_rate.*dWih;
    Whh = Whh - learning_rate.*dWhh;
    Who = Who - learning_rate.*dWho;


end