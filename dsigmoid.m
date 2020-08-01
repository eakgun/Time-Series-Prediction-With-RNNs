function output = dsigmoid(x)
    
    output = sigmoid(x).*(1-sigmoid(x));

end