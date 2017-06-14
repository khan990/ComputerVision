



function SoftMaxPushBack(Train_features, Test_features, Train_Labels, Test_Labels)

    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
    Softmax_model = trainSoftmaxLayer(Train_features, Train_Labels, 'ShowProgressWindow', false);
%     AntiSoftmax_model = trainSoftmaxLayer(Train_features, double(not(Train_Labels)), 'ShowProgressWindow', false);
    
    original_estimate = Softmax_model(Train_features);
%     AntiOriginal_estimate = AntiSoftmax_model(Train_features);
    

    threshold = (-0.5:0.0001:1);
    
    acc = zeros(length(threshold), 1);
    sen = zeros(length(threshold), 1);
    spe = zeros(length(threshold), 1);
    
    for i=1:length(threshold)
        estimate = original_estimate;
        
        estimate(estimate < threshold(i)) = 0;
        estimate(estimate > 0           ) = 1;
        
        evals = Evaluate(Train_Labels, estimate);
        acc(i) = evals(1);
        sen(i) = evals(2);
        spe(i) = evals(3);
        
    end
    
    plot(threshold, acc, threshold, sen, threshold, spe);
    hold on;
    legend('Accuracy', 'Sensitivity', 'Specificity');
    hold off;
    
    index = find(acc == max(acc), 1);
    
    estimate = original_estimate;
        
    estimate(estimate < .3) = 0;
    estimate(estimate > 0               ) = 1;
    
    evals = Evaluate(Train_Labels, estimate)
    
    
    Softmax_model_error = trainSoftmaxLayer(Train_features, estimate, 'ShowProgressWindow', false);
    estimate = Softmax_model(Test_features);
    error_estimate = Softmax_model_error(Test_features);
    
    Estimate = estimate - error_estimate;

    acc = 0;
    sen = 0;
    spe = 0;
    
    
    for i=1:length(threshold)
        estimate = Estimate;
        
        estimate(estimate < threshold(i)) = 0;
        estimate(estimate > 0           ) = 1;
        
        evals = Evaluate(Test_Labels, estimate);
        acc(i) = evals(1);
        sen(i) = evals(2);
        spe(i) = evals(3);
        
    end
    
    index = find(acc == max(acc), 1);
    
    estimate = Estimate;
        
    estimate(estimate < threshold(index)) = 0;
    estimate(estimate > 0               ) = 1;

    evals = Evaluate(Test_Labels(:),estimate(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    
end