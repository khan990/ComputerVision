
% 
% Accuracy		0.816462
% Sensitivity		0.622756
% Specificity		0.882014
% Precision		0.641090
% Recall			0.622756
% F_Measure		0.631790
% Gmean			0.741134

function DecisionTree(Train_features, Test_features, Train_labels, Test_labels)
% Relabeling data as 1s and -1s
    Num_of_Classes = size(Train_labels, 1);
    Num_of_Images = size(Train_labels, 2);
    
%     Train_labels(Train_labels == 0) = -1;
%     Test_labels(Test_labels == 0) = -1;
    
    Tree_Array = cell(Num_of_Classes, 1);
    
    for i=1:Num_of_Classes
        fprintf('Training Class Number = %d\n', i);
        Tree_Array{i} = fitctree(Train_features', Train_labels(i,:)');
    end

    Estimated = zeros(Num_of_Classes, Num_of_Images);
    tic
    for i=1:Num_of_Classes
        Estimated(i,:) = predict(Tree_Array{i}, Test_features');
    end
    toc

    
    evals = Evaluate(Test_labels(:),Estimated(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));


end

