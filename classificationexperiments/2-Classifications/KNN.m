% KNN 

% Example
% KNN(Training_CNN_features, Test_CNN_features, train_labels, test_labels, 'Distance', 'cosine');

function KNN(Training_Features, Test_Features, Train_Labels, Test_Labels, Method, MethodType)

    IDX = knnsearch(Training_Features',Test_Features', Method, MethodType);
    
    estimated_objects_of_test = zeros(size(Train_Labels, 1), size(Train_Labels, 2));

    tic
    for i=1:length(IDX)
        estimated_objects_of_test(:,i) = Train_Labels(:, IDX(i));
    end
    toc
    
    evals = Evaluate(Test_Labels(:), estimated_objects_of_test(:));
%     fprintf('Accuracy\t\tSensitivity\t\tSpecificity\t\tPrecision\t\tRecall\t\tF_Measure\t\tGmean\n');
%     evals
%     fprintf('%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\n', evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
end


% KNN(Train_CNN_features, Test_CNN_features, Train_Labels, Test_Labels, 'Distance', 'cosine')
% Accuracy		0.834769
% Sensitivity		0.678126
% Specificity		0.887779
% Precision		0.671588
% Recall			0.678126
% F_Measure		0.674841
% Gmean			0.775903