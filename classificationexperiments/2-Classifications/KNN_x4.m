


function  KNN_x4(Train_features, Test_features, Train_Labels, Test_Labels)


%     Distance_vars = {'euclidean' 'seuclidean' 'cityblock' 'chebychev' 'minkowski' 'cosine' 'correlation' 'spearman' 'hamming' 'jaccard'};
Distance_vars = {'cosine'};
    Distance_name = {'Distance'};
    
    for j=1:length(Distance_vars)
        K = 5;
        IDX = knnsearch(Train_features',Test_features', 'Distance', Distance_vars{j}, 'k', K);
%         estimated_objects_of_test = zeros(size(Train_Labels, 1), size(Train_Labels, 2), K);
%         
%         for k=1:K
%             tic
%             for i=1:length(IDX)
%                 estimated_objects_of_test(:,i, k) = Train_Labels(:, IDX(i,k));
%             end
%             toc
%         end
        estimated_objects_of_test = zeros(size(Train_Labels, 1), size(Train_Labels, 2));
        tic
       for k=1:K
            for i=1:length(IDX)
                estimated_objects_of_test(:,i) = estimated_objects_of_test(:,i) + Train_Labels(:, IDX(i,k));
            end
       end
       toc
       if K ~= 1
           estimated_objects_of_test(estimated_objects_of_test < ((K-1)/2)) = 0;
           estimated_objects_of_test(estimated_objects_of_test >  0       ) = 1;
       end
%        result = estimated_objects_of_test;
        evals = Evaluate(Test_Labels(:), estimated_objects_of_test(:));
        
        fprintf('\nDistance = %s\n', Distance_vars{j});
        Print_Evaluations(evals);
    end

%         for j=1:20
%             IDX = knnsearch(Train_features',Test_features', 'Distance', 'minkowski', 'P', j);
%             tic
%             for i=1:length(IDX)
%                 estimated_objects_of_test(:,i) = Train_Labels(:, IDX(i));
%             end
%             toc
%             evals = Evaluate(Test_Labels(:), estimated_objects_of_test(:));
% 
%             fprintf('P = %d\n', j);
%             Print_Evaluations(evals);
%         end
    
end