


function correlationAmongLabels(Train_features, Test_features, Train_Labels, Test_Labels, classes)
  
    
    
    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
    x = (1:Num_of_Classes);
    x = x';
    x_combinations = combnk(x, 2);
    
    x_interRelations1s = zeros(Num_of_Classes, size(x_combinations, 1));
    x_interRelations0s = zeros(Num_of_Classes, size(x_combinations, 1));
    
    for i=1:size(x_combinations, 1)
        for j=1:Num_of_Images
            if (Train_Labels(x_combinations(i,1), j) == 1 && Train_Labels(x_combinations(i,2), j) == 1)
                x_interRelations1s(:, i) = x_interRelations1s(:, i) + Train_Labels(:,j);
            end
        end
    end
    
    for i=1:size(x_combinations, 1)
        for j=1:Num_of_Images
            if (Train_Labels(x_combinations(i,1), j) == 0 && Train_Labels(x_combinations(i,2), j) == 0)
                temp = Train_Labels(:,j);
                temp(temp == 1) = -1;
                temp(temp == 0) = 1;
                temp(temp == -1) = 0;
                x_interRelations0s(:, i) = x_interRelations0s(:, i) + temp;
            end
        end
    end
    
    softnet = trainSoftmaxLayer(Train_features,Train_Labels, 'ShowProgressWindow', false);
    estimated = softnet(Test_features);
    
    post_estimated = estimated;
    post_estimated_temp = estimated;
    
    post_estimated_temp(post_estimated_temp < 0.0001) = 0;
    post_estimated_temp(post_estimated_temp > 0) = 1;
    
  
    disp('SoftMax Values...');
    evals = Evaluate(Test_Labels(:),post_estimated_temp(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

    disp('Post calculation Values...');
    
    chil = (1:0.1:10);
    chil = chil';
    
    for t=1:size(chil)
        chil(t)
        post_estimated = estimated;
        for i=1:size(x_combinations, 1)
            for j=1:500
                if ( post_estimated_temp(x_combinations(i,1), j) == 1 && post_estimated_temp(x_combinations(i,2), j) == 1 )
                    max_value = max(x_interRelations1s(:, i));
                    indexes = find(x_interRelations1s(:, i) > max_value/2);
                    mean_value = mean(x_interRelations1s(:, i));
    %                 fprintf('1s... Max = %d, Max/2 = %d\n', max_value, max_value/2);
                    for k=1:size(indexes)
    %                     multiple = (x_interRelations1s(k, i)/mean_value);
                        post_estimated(k, i) = post_estimated(k, i) * (chil(t));
                    end
                end
            end
        end

%         post_estimated(post_estimated > 0.0001) = 1;
        
        for i=1:size(x_combinations, 1)
            for j=1:500
                if ( post_estimated_temp(x_combinations(i,1), j) == 0 && post_estimated_temp(x_combinations(i,2), j) == 0 )
                    max_value = max(x_interRelations0s(:, i));
                    indexes = find(x_interRelations0s(:, i) > max_value);
    %                 fprintf('0s... Max = %d, Max/2 = %d\n', max_value, max_value/2);
                    for k=1:size(indexes)
%                         fprintf('%e to ',post_estimated(k, i));
                        post_estimated(k, i) = post_estimated(k, i) / chil(t);
%                         fprintf('%e\n',post_estimated(k, i));
                    end
                end
            end
        end
        

        post_estimated(post_estimated < 0.0001) = 0;
        post_estimated(post_estimated > 0) = 1;

        evals = Evaluate(Test_Labels(:),post_estimated(:));
        fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

    end
    
end