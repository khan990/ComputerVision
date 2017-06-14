



function GPRfeatureTraining_SoftmaxErrorTraining_x(Train_features, Test_features, Train_Labels, Test_Labels)

    Num_of_classes = size(Train_Labels, 1);
    Num_of_images = size(Train_Labels, 2);
    
    for i=1:Num_of_classes
        GPR_model{i} = fitrgp(Train_features', Train_Labels(i,:)');
    end
    
    tic
    for i=1:Num_of_classes
        estimated(i,:) = predict(GPR_model{i}, Test_features');
    end
    toc
    
    for i=1:Num_of_classes
        error_estimate(i,:) = predict(GPR_model{i}, Train_features');
    end
    
    error_estimate = Train_Labels - error_estimate;
    
    
    softmax_errorModel = trainSoftmaxLayer(Train_features, error_estimate, 'ShowProgressWindow', false);
    tic
    residualTestError = softmax_errorModel(Test_features);
    
    estimated = estimated + residualTestError;
    toc
    
    
    
    
    
    
    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_features, Train_Labels, Kfolds);

    disp('Finding Threshold');
    threshold = (0:0.0001:1);
    
    acc = zeros(length(threshold), Kfolds);
    sen = zeros(length(threshold), Kfolds);
    spe = zeros(length(threshold), Kfolds);
    
    for x=1:Kfolds
        fprintf('Kfold = %d\n', x);
        
        for i=1:Num_of_classes
            ft_GPR_model{i} = fitrgp(Tr_features{x}', Tr_Labels{x}(i,:)');
        end

        
        for i=1:Num_of_classes
            ft_estimated(i,:) = predict(ft_GPR_model{i}, Te_features{x}');
        end
        

        for i=1:Num_of_classes
            ft_error_estimate(i,:) = predict(ft_GPR_model{i}, Tr_features{x}');
        end

        ft_error_estimate = Tr_Labels{x} - ft_error_estimate;

        
        ft_softmax_errorModel = trainSoftmaxLayer(Tr_features{x}, ft_error_estimate, 'ShowProgressWindow', false);
        ft_residualTestError = ft_softmax_errorModel(Te_features{x});

        ft_final_estimate = ft_estimated + ft_residualTestError;
        
        
        
        for i=1:length(threshold)
            ft_estimated_x = ft_final_estimate;
        
            ft_estimated_x(ft_estimated_x < threshold(i)) = 0;
            ft_estimated_x(ft_estimated_x > 0           ) = 1;

            evals = Evaluate(Te_Labels{x}(:),ft_estimated_x(:));
            
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
    
    final_estimate_temp = estimated;
    final_estimate_temp(final_estimate_temp < threshold(index)) = 0;
    final_estimate_temp(final_estimate_temp > 0               ) = 1;

    evals = Evaluate(Test_Labels, final_estimate_temp);
    Print_Evaluations(evals);
    

end