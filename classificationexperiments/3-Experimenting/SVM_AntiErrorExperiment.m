



function SVM_AntiErrorExperiment(Train_feature, Test_feature, Train_Labels, Test_Labels)

    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
    
    for i=1:Num_of_Classes
        fprintf('Class Number = %d\n', i);
        model{i} = fitcsvm(Train_feature', Train_Labels(i,:)');
    end

    for i=1:Num_of_Classes
        [estimate(i,:), score] = predict(model{i}, Test_feature');
    end

    evals = Evaluate(Test_Labels(:),estimate(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
end