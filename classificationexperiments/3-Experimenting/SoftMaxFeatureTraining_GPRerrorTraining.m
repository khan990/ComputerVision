


function SoftMaxFeatureTraining_GPRerrorTraining(Train_features, Test_features, Train_Labels, Test_Labels)

    Num_of_classes = size(Train_Labels, 1);
    Num_of_images = size(Train_Labels, 2);
    
    softmax_train_model = trainSoftmaxLayer(Train_features, Train_Labels, 'ShowProgressWindow', false);
    tic
    estimated_error = softmax_train_model(Train_features);
    estimated = softmax_train_model(Test_features);
    toc
    
    estimated_error = Train_Labels - estimated_error;
    
    for i=1:Num_of_classes
        GPR_error_model{i} = fitrgp(Train_features', estimated_error(i,:)');
    end
    
    tic
    for i=1:Num_of_classes
        error_estimated(i,:) = predict(GPR_error_model{i}, Test_features');
    end
    toc
    
    final_estimate = estimated + error_estimated;
    
    threshold = (-1:0.0001:1.5);
    
    acc = zeros(length(threshold), 1);
    sen = zeros(length(threshold), 1);
    spe = zeros(length(threshold), 1);
    
    for i=1:length(threshold)
        final_estimate_temp = final_estimate;
        final_estimate_temp(final_estimate_temp < threshold(i)) = 0;
        final_estimate_temp(final_estimate_temp > 0           ) = 1;
        
        evals = Evaluate(Test_Labels, final_estimate_temp);
        
        acc(i) = evals(1);
        sen(i) = evals(2);
        spe(i) = evals(3);
    end
    
    index = find(acc == max(acc), 1);
    
    final_estimate_temp = final_estimate;
    final_estimate_temp(final_estimate_temp < threshold(index)) = 0;
    final_estimate_temp(final_estimate_temp > 0               ) = 1;

    evals = Evaluate(Test_Labels, final_estimate_temp);
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

end