
%previous
%[accuracy sensitivity specificity precision recall f_measure gmean]
%   0.8126    0.8147    0.8119    0.5945    0.8147    0.6874   0.8133
%For max/2 for missing objects
% Accuracy		0.806538
% Sensitivity		0.848798
% Specificity		0.792237
% Precision		0.580283
% Recall			0.848798
% F_Measure		0.689314
% Gmean			0.820030

% Example
% Increase0s_Mohamad(Training_CNN_features, Test_CNN_features, train_labels, test_labels, 0.0001, 0);

function Increase0s_Mohamad(Train_features, Test_features, Train_Labels, Test_Labels, upperbound, lowerbound)
% SoftMax Part
    softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);

    estimated = softnet(double(Test_features));

    estimated( estimated < upperbound)=0;

    estimated(estimated > lowerbound)=1;
    
    Post_estimated_2 = estimated;
% Relational Table Part
    Num_of_Classes = size(Train_Labels, 1); %Number of classes
    Num_of_training_samples = size(Train_Labels, 2); %number of training samples

    RelationTable = zeros(Num_of_Classes, Num_of_Classes);

    for i=1:Num_of_Classes
        for j=1:Num_of_Classes
            if (i~=j)
                for m=1:Num_of_training_samples
                    if (Train_Labels(i,m)==0) && (Train_Labels(j,m)==0)
                        RelationTable(i,j) = RelationTable(i,j) + 1;
                    end
                end
            else
                RelationTable(i,j) = 0;
            end

        end
    end
% Increasing 1s part

    for i=1:Num_of_training_samples
%         i
        x = find(Post_estimated_2(:,i) == 0);
        x_combinations = combnk(x,2);
        NewCol = zeros(size(x_combinations(:,1)));
        x_combinations = [x_combinations NewCol];
        for j=1:size(x_combinations)
            x_combinations(j, 3) = RelationTable(x_combinations(j, 1), x_combinations(j, 2)); 
        end
        [MaxValue, index] = max(x_combinations(:, 3));
        indexes_to_remove = find(x_combinations(:, 3) < MaxValue/2);
        x_combinations(indexes_to_remove, :) = [];

        for k=1:Num_of_Classes
            if (Post_estimated_2(k,i) == 0)
                if(any(x_combinations(:,1) == k) || any(x_combinations(:,2) == k))
                    if( any(x_combinations(:,1) == k) )
                        position = find(x_combinations(:,1) == k, 1);
                        Post_estimated_2(x_combinations(position,2),i) = 0;
                    elseif ( any(x_combinations(:,2) == k) )
                        position = find(x_combinations(:,2) == k, 1);
                        Post_estimated_2(x_combinations(position,1),i) = 0;
                    end
                else
                    Post_estimated_2(k,i) = 1;
                end
            end
        end
    end

    evals = Evaluate(Test_Labels, Post_estimated_2);
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));


end
