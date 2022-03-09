% control.m

%dale_transform(J, 0)

weights = normrnd(0,1,[100,150]);
[activities, number_of_neurons] = dgRNN(weights)

%{
for filenumber = 1:100
    joes_main_rnn_code(filenumber, false)
end
%}