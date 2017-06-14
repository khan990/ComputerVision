


function GPRegression_x(Train_feature, Test_feature, Train_Label, Test_Label)

    Num_of_Classes = size(Train_Label, 1);
    Num_of_Images = size(Train_feature, 2);
    
    
    for i=1:Num_of_Classes
        fprintf('Class Number = %d\n', i);
        model{i} = fitrgp(Train_feature', Train_Label(i, :)');
    end
    tic
    for i=1:Num_of_Classes
        estimate(i, :) = predict(model{i}, Test_feature');
    end
    toc
    
    
    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_feature, Train_Label, Kfolds);

    disp('Finding Threshold');
    threshold = (0:0.0001:1);
    
    acc = zeros(length(threshold), Kfolds);
    sen = zeros(length(threshold), Kfolds);
    spe = zeros(length(threshold), Kfolds);
    
    for x=1:Kfolds
        fprintf('Kfold = %d\n', x);
        for i=1:Num_of_Classes
            ft_model{i} = fitrgp(Tr_features{x}', Tr_Labels{x}(i, :)');
        end

        for i=1:Num_of_Classes
            ft_final_estimate(i, :) = predict(ft_model{i}, Te_features{x}');
        end

        for i=1:length(threshold)
            ft_estimated = ft_final_estimate;
        
            ft_estimated(ft_estimated < threshold(i)) = 0;
            ft_estimated(ft_estimated > 0           ) = 1;

            evals = Evaluate(Te_Labels{x}(:),ft_estimated(:));
            
            acc(i, x) = evals(1);
            sen(i, x) = evals(2);
            spe(i, x) = evals(3);
            
        end
    end
    
    final_acc = mean(acc');
    final_sen = mean(sen');
    final_spe = mean(spe');
    
    plot(threshold, final_acc, threshold, final_sen, threshold, final_spe);
    hold on;
    legend('Accuracy', 'Sensitivity', 'Specificity');
    hold off;
    
    index = find(final_acc == max(final_acc), 1);
    
    
    
    
    
    
    
    post_estimate = estimate;
    post_estimate(post_estimate < threshold(index) ) = 0;
    post_estimate(post_estimate > 0 ) = 1;
    evals = Evaluate(Test_Label(:),post_estimate(:));
    Print_Evaluations(evals);
end