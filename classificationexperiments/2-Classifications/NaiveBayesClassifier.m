
% Accuracy		0.837769
% Sensitivity		0.638272
% Specificity		0.905282
% Precision		0.695162
% Recall			0.638272
% F_Measure		0.665504
% Gmean			0.760142


function NaiveBayesClassifier(Train_features, Test_features, Train_Labels, Test_Labels)

%     Removing Underpass Labels
%     Test_Labels(21, :) = [];
%     Train_Labels(21, :) = [];
    
    Num_of_classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
    for i=1:Num_of_classes
        i
        NBc{i} = fitcnb(Train_features', Train_Labels(i, :)', 'Distribution', 'kernel');
    end
    
    tic
    for i=1:Num_of_classes
        i
        Estimated(i, :) = predict(NBc{i}, Test_features');
    end
    toc
    
    evals = Evaluate(Test_Labels(:),Estimated(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));


end