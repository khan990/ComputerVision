


function KNN_x(Train_features, Test_features, Train_Labels, Test_Labels)


    Distance_vars = {'euclidean' 'seuclidean' 'cityblock' 'chebychev' 'minkowski' 'cosine' 'correlation' 'spearman' 'hamming' 'jaccard'};
    Distance_name = {'Distance'};
    
    for j=1:length(Distance_vars)
        
        IDX = knnsearch(Train_features',Test_features', 'Distance', Distance_vars{j});
        tic
        for i=1:length(IDX)
            estimated_objects_of_test(:,i) = Train_Labels(:, IDX(i));
        end
        toc
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