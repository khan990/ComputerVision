
% 
% Accuracy		0.747154
% Sensitivity		0.000000
% Specificity		1.000000
% Precision		NaN
% Recall			0.000000
% F_Measure		NaN
% Gmean			0.000000

function TreeBaggerClassification( NumberOfTrees, Train_features, Test_features, Train_Labels, Test_Labels)

    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
    for i = 1:Num_of_Classes
        i
        Trees{i} = TreeBagger(NumberOfTrees, Train_features', Train_Labels(i,:)');
    end
    
    disp('Training Done...');
    disp('Predict starting...');
    
    tic
    for i=1:Num_of_Classes
        i
        Estimated(i,:) = predict(Trees{i}, Test_features');
    end
    toc
    
    estimated = cell2mat(Estimated);
    
    
    evals = Evaluate(Test_Labels(:), estimated(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

end