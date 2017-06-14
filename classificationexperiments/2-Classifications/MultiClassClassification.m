% fitcecoc
% see also fitensemble

function MultiClassClassification(Train_features, Test_features, Train_Labels, Test_Labels)
    
%     Preprocessing
    Num_of_Classes = 26;
    Num_of_Images = 500;
    
    for i=1:Num_of_Classes
        i
        x{i} = fitcecoc(Train_features', Train_Labels(i,:)', 'Learners', 'tree', 'Verbose', 1);
    end

    
    tic
    for i=1:Num_of_Classes
        estimate(i, :) = predict(x{i}, Test_features');
    end
%     estimate = predict(x, Test_features');
    toc
    
    evals = Evaluate(Test_Labels, estimate);
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

% 

end

% % Default
% Accuracy		0.851154
% Sensitivity		0.664740
% Specificity		0.914239
% Precision		0.723989
% Recall			0.664740
% F_Measure		0.693101
% Gmean			0.779571

% x{i} = fitcecoc(Train_features', Train_Labels(i,:)', 'Coding', 'allpairs');
% Elapsed time is 0.843213 seconds.
% Accuracy		0.851154
% Sensitivity		0.664740
% Specificity		0.914239
% Precision		0.723989
% Recall			0.664740
% F_Measure		0.693101
% Gmean			0.779571

% x{i} = fitcecoc(Train_features', Train_Labels(i,:)', 'Learners', 'discriminant');
% Elapsed time is 1.481172 seconds.
% Accuracy		0.817000
% Sensitivity		0.644357
% Specificity		0.875425
% Precision		0.636418
% Recall			0.644357
% F_Measure		0.640363
% Gmean			0.751056

% x{i} = fitcecoc(Train_features', Train_Labels(i,:)', 'Learners', 'knn');
% Elapsed time is 2.522434 seconds.
% Accuracy		0.837308
% Sensitivity		0.643444
% Specificity		0.902914
% Precision		0.691629
% Recall			0.643444
% F_Measure		0.666667
% Gmean			0.762217

% x{i} = fitcecoc(Train_features', Train_Labels(i,:)', 'Learners', 'naivebayes');
% Elapsed time is 2.005714 seconds.
% Accuracy		0.840308
% Sensitivity		0.682689
% Specificity		0.893648
% Precision		0.684773
% Recall			0.682689
% F_Measure		0.683729
% Gmean			0.781079

% x{i} = fitcecoc(Train_features', Train_Labels(i,:)', 'Learners', 'svm');
% Elapsed time is 0.813183 seconds.
% Accuracy		0.851154
% Sensitivity		0.664740
% Specificity		0.914239
% Precision		0.723989
% Recall			0.664740
% F_Measure		0.693101
% Gmean			0.779571

% x{i} = fitcecoc(Train_features', Train_Labels(i,:)', 'Learners', 'tree');
% Elapsed time is 0.654153 seconds.
% Accuracy		0.817385
% Sensitivity		0.630666
% Specificity		0.880572
% Precision		0.641200
% Recall			0.630666
% F_Measure		0.635890
% Gmean			0.745216


