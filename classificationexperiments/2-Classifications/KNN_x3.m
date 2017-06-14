


function KNN_x3(Train_features, Test_features, Train_Labels, Test_Labels)


%     Distance_vars = {'euclidean' 'seuclidean' 'cityblock' 'chebychev' 'minkowski' 'cosine' 'correlation' 'spearman' 'hamming' 'jaccard'};
    Distance_vars = {'euclidean' 'cityblock' 'chebychev' 'minkowski'};
    Distance_name = {'Distance'};
    
    for j=1:length(Distance_vars)
        
        Mdl = KDTreeSearcher(Train_features', 'Distance', Distance_vars{j});
        IDX = knnsearch(Mdl,Test_features', 'k', 3);
        vote_table = zeros(size(IDX,1), size(IDX, 2));
        tic
        for i=1:size(IDX,1)
            for j=1:size(IDX,2)
                vote_table(i,:) = Train_Labels(:, IDX(i,j))
            end
            estimated_objects_of_test(:,i) = Train_Labels(:, IDX(i,:));
        end
        toc
        evals = Evaluate(Test_Labels(:), estimated_objects_of_test(:));
        
        fprintf('\nDistance = %s\n', Distance_vars{j});
        Print_Evaluations(evals);
    end
% 
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