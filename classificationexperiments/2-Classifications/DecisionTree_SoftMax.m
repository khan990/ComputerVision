
% Accuracy		0.769000
% Sensitivity		0.536051
% Specificity		0.847833
% Precision		0.543827
% Recall			0.536051
% F_Measure		0.539911
% Gmean			0.674153


function DecisionTree_SoftMax(Train_features, Test_features, Train_Labels, Test_Labels)
    
    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
    Train_Label_Learning = trainSoftmaxLayer(Train_Labels, Train_Labels, 'ShowProgressWindow', false);
    Train_Label_Estimation = Train_Label_Learning(Train_features);
    
    disp('Decision Tree...');
    for i=1:Num_of_Classes
        DecisionTree{i} = fitctree(Train_Label_Estimation', Train_Labels(i, :)');
    end
    
    Test_Label_Learning = trainSoftmaxLayer(Test_Labels, Test_Labels, 'ShowProgressWindow', false);
    Test_Label_Estimation = Test_Label_Learning(Test_features);
    
    disp('Estimation...');
    for i=1:Num_of_Classes
        Estimation(i, :) = predict(DecisionTree{i}, Test_Label_Estimation');
    end
    
    evals = Evaluate(Test_Labels(:),Estimation(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

end