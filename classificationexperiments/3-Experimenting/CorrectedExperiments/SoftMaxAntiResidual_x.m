





function SoftMaxAntiResidual_x(Train_features, Test_features, Train_Labels, Test_Labels)

    SoftMax_TrainFeatures_TrainLabels = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);
    tic
    estimated_TestFeatures_on_Tef_Tl = SoftMax_TrainFeatures_TrainLabels(double(Test_features));
    toc
    estimated_TrainLabelErrors_on_Tf_Tl = SoftMax_TrainFeatures_TrainLabels(double(Train_features));
    
    ErrorResidual = Train_Labels - estimated_TrainLabelErrors_on_Tf_Tl;
    
    SoftMax_TrainFeatures_ErrorResidual = trainSoftmaxLayer(double(Train_features),ErrorResidual, 'ShowProgressWindow', false);
    tic
    estimate_Residual_on_Tef = SoftMax_TrainFeatures_ErrorResidual(Test_features);
    toc
    final_estimate = estimate_Residual_on_Tef + estimated_TestFeatures_on_Tef_Tl;
    
    
    
    
    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_features, Train_Labels, Kfolds);

    disp('Finding Threshold');
    threshold = (0:0.0001:1);
    
    acc = zeros(length(threshold), Kfolds);
%     sen = zeros(length(threshold), Kfolds);
%     spe = zeros(length(threshold), Kfolds);
    
    for x=1:Kfolds
        fprintf('Kfold = %d\n', x);
        ft_SoftMax_TrainFeatures_TrainLabels = trainSoftmaxLayer(double(Tr_features{x}),Tr_Labels{x}, 'ShowProgressWindow', false);
        ft_estimated_TestFeatures_on_Tef_Tl = ft_SoftMax_TrainFeatures_TrainLabels(double(Te_features{x}));
        
        ft_estimated_TrainLabelErrors_on_Tf_Tl = ft_SoftMax_TrainFeatures_TrainLabels(double(Tr_features{x}));
        ft_ErrorResidual = Tr_Labels{x} - ft_estimated_TrainLabelErrors_on_Tf_Tl;
        
        ft_SoftMax_TrainFeatures_ErrorResidual = trainSoftmaxLayer(double(Tr_features{x}),ft_ErrorResidual, 'ShowProgressWindow', false);
        ft_estimate_Residual_on_Tef = ft_SoftMax_TrainFeatures_ErrorResidual(Te_features{x});
        
        ft_final_estimate = ft_estimate_Residual_on_Tef + ft_estimated_TestFeatures_on_Tef_Tl;
        for i=1:length(threshold)
            ft_estimated = ft_final_estimate;
        
            ft_estimated(ft_estimated < threshold(i)) = 0;
            ft_estimated(ft_estimated > 0           ) = 1;

            evals = Evaluate(Te_Labels{x}(:),ft_estimated(:));
            
            acc(i, x) = evals(1);
%             sen(i, x) = evals(2);
%             spe(i, x) = evals(3);
            
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
%     index = input('Put value: ');
    estimated = final_estimate;
        
    estimated(estimated < threshold(index)) = 0;
    estimated(estimated > 0               ) = 1;
 
    evals = Evaluate(Test_Labels(:),estimated(:));
    Print_Evaluations(evals);
end