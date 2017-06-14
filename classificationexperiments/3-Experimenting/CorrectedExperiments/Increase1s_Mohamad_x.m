


function Increase1s_Mohamad_x(Train_features, Test_features, Train_Labels, Test_Labels)

% SoftMax Part
    softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);

    estimated = softnet(double(Test_features));

    Post_estimated = estimated;
    
% Relational Table Part
    Num_of_Classes = size(Train_Labels, 1); %Number of classes
    Num_of_training_samples = size(Train_Labels, 2); %number of training samples

    RelationTable = zeros(Num_of_Classes, Num_of_Classes);

    for i=1:Num_of_Classes
        for j=1:Num_of_Classes
            if (i~=j)
                for m=1:Num_of_training_samples
                    if (Train_Labels(i,m)==1) && (Train_Labels(j,m)==1)
                        RelationTable(i,j) = RelationTable(i,j) + 1;
                    end
                end
            else
                RelationTable(i,j) = 0;
            end

        end
    end
    
    
    disp('Finding Threshold');
    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_features, Train_Labels, Kfolds);

    threshold = (0:0.0001:1);
    
    acc = zeros(length(threshold), Kfolds);
    sen = zeros(length(threshold), Kfolds);
    spe = zeros(length(threshold), Kfolds);
    
    for y=1:Kfolds
        fprintf('Kfold = %d\n', y);
        ft_softnet = trainSoftmaxLayer(double(Tr_features{y}), Tr_Labels{y}, 'ShowProgressWindow', false);
        ft_estimated = ft_softnet(double(Te_features{y}));
        
        for z=1:length(threshold)
            ft_Post_estimated = ft_estimated;

            ft_Post_estimated(ft_Post_estimated < threshold(z)) = 0;
            ft_Post_estimated(ft_Post_estimated > 0           ) = 1;



            % Increasing 1s part
            Num_of_training_samples = size(Te_features{y}, 2); %number of training samples
            for i=1:Num_of_training_samples
%                 i
                x = find(ft_Post_estimated(:,i));
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
                    if (ft_Post_estimated(k,i) == 1)
                        if(any(x_combinations(:,1) == k) || any(x_combinations(:,2) == k))
                            if( any(x_combinations(:,1) == k) )
                                position = find(x_combinations(:,1) == k, 1);
                                ft_Post_estimated(x_combinations(position, 2), i) = 1;
                            elseif ( any(x_combinations(:,2) == k) )
                                position = find(x_combinations(:,2) == k, 1);
                                ft_Post_estimated(x_combinations(position, 1), i) = 1;
                            end
        %                     Post_estimated(k,i) = 1;
                        else
                            ft_Post_estimated(k,i) = 0;
                        end
                    end
                end
            end
            evals = Evaluate(Te_Labels{y}(:),ft_Post_estimated(:));

            acc(z, y) = evals(1);
            sen(z, y) = evals(2);
            spe(z, y) = evals(3);

        end
        
        
    end
    
    final_acc = mean(acc');
    final_sen = mean(sen');
    final_spe = mean(spe');
    
    plot(threshold, final_acc, threshold, final_sen, threshold, final_spe);
    hold on;
    legend('Accuracy', 'Sensitivity', 'Specificity');
    hold off;
    

    
    evals = Evaluate(Test_Labels(:), Post_estimated(:));
    Print_Evaluations(evals);
end
