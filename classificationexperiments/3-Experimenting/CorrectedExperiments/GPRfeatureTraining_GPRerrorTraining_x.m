



function GPRfeatureTraining_GPRerrorTraining_x(Train_features, Test_features, Train_Labels, Test_Labels)

    Num_of_classes = size(Train_Labels, 1);
    Num_of_images = size(Train_Labels, 2);
    GPR_model = cell(Num_of_classes, 1);
%     --------------------------------------
    for i=1:Num_of_classes
        GPR_model{i} = fitrgp(Train_features', Train_Labels(i,:)');
    end
    
    
    for i=1:Num_of_classes
        errorEstimate(i,:) = predict(GPR_model{i}, Train_features');
    end
    
    tic
    for i=1:Num_of_classes
        estimated(i, :) = predict(GPR_model{i}, Test_features');
    end
    toc
    errorEstimate = Train_Labels - errorEstimate;
    
    for i=1:Num_of_classes
        GPR_model{i} = fitrgp(Train_features', errorEstimate(i,:)');
    end
    tic
    for i=1:Num_of_classes
        errorEstimate(i,:) = predict(GPR_model{i}, Test_features');
    end
    
    estimated = estimated + errorEstimate;
    toc
%     -----------------------------------------------------
    disp('Finding Threshold');
    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_features, Train_Labels, Kfolds);

    threshold = (0:0.0001:1);
    
    acc = zeros(length(threshold), Kfolds);
%     sen = zeros(length(threshold), Kfolds);
%     spe = zeros(length(threshold), Kfolds);
    
    for x=1:Kfolds
        fprintf('Kfold = %d\n', x);
        for i=1:Num_of_classes
            GPR_model{i} = fitrgp(Tr_features{x}', Tr_Labels{x}(i,:)');
        end
        
        for i=1:Num_of_classes
            ft_estimated(i, :) = predict(GPR_model{i}, Te_features{x}');
            ft_errorEstimate(i,:) = predict(GPR_model{i}, Tr_features{x}');
        end
        
        ft_errorEstimate = Tr_Labels{x} - ft_errorEstimate;
    
        for i=1:Num_of_classes
            GPR_model{i} = fitrgp(Tr_features{x}', ft_errorEstimate(i,:)');
        end

        for i=1:Num_of_classes
            ft_errorEstimate_ret(i,:) = predict(GPR_model{i}, Te_features{x}');
        end

        ft_estimated = ft_estimated + ft_errorEstimate_ret;
        
        
        
        for j=1:length(threshold)
            ft_final_estimate = ft_estimated;
        
            ft_final_estimate(ft_final_estimate < threshold(j)) = 0;
            ft_final_estimate(ft_final_estimate > 0           ) = 1;

            evals = Evaluate(Te_Labels{x}(:),ft_final_estimate(:));
            
            acc(j, x) = evals(1);
%             sen(j, x) = evals(2);
%             spe(j, x) = evals(3);
            
        end
    end
    
    final_acc = mean(acc');
%     final_sen = mean(sen');
%     final_spe = mean(spe');
    
%     plot(threshold, final_acc, threshold, final_sen, threshold, final_spe);
%     hold on;
%     legend('Accuracy', 'Sensitivity', 'Specificity');
%     hold off;
    
    
    
    
    index = find(final_acc == max(final_acc), 1);
    
    final_estimate_temp = estimated;
    final_estimate_temp(final_estimate_temp < threshold(index)) = 0;
    final_estimate_temp(final_estimate_temp > 0               ) = 1;

    evals = Evaluate(Test_Labels, final_estimate_temp);
    Print_Evaluations(evals);
end