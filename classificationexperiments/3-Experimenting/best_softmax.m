


x = (0.00001:0.00001:0.0009);

for i=1:length(x)
    fprintf('%d of %d\n', i, length(x));
    evals{i} = SoftMax(Train_CNN_features, Test_CNN_features, Train_Labels, Test_Labels, x(i), 0);
end


for i=1:length(x)
    acc(i) = evals{i}(1);
    sen(i) = evals{i}(2);
    spe(i) = evals{i}(3);
end