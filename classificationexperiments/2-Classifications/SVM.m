

% SVM Algorithm over Feature set
% see also fitcecoc
% 
% Accuracy		0.851154
% Sensitivity		0.664740
% Specificity		0.914239
% Precision		0.723989
% Recall			0.664740
% F_Measure		0.693101
% Gmean			0.779571

function SVM(Train_features, Test_features, Train_labels, Test_labels)
% Relabeling data as 1s and -1s
    Num_of_Classes = size(Train_labels, 1);
    Num_of_Images = size(Train_labels, 2);
    
%     Training SVM Tables
%     SVM_Array = repmat( struct , Num_of_Classes, 1);
    SVM_Array = cell(Num_of_Classes, 1);
    
    for i=1:Num_of_Classes
        SVM_Array{i} = fitcsvm(Train_features', Train_labels(i,:)');
%         SVM_Array(i) = Temp;
    end

% Testing SVM Tables
    Estimated = zeros(Num_of_Classes, Num_of_Images);
    
    tic
    for i=1:Num_of_Classes
        Estimated(i,:) = predict(SVM_Array{i}, Test_features');
    end
    toc
% Evaluate
    
    evals = Evaluate(Test_labels(:),Estimated(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

end