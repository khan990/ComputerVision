


function SoftMaxAntiResidual(Train_features, Test_features, Train_Labels, Test_Labels)

    SoftMax_TrainFeatures_TrainLabels = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);
    estimated_TestFeatures_on_Tef_Tl = SoftMax_TrainFeatures_TrainLabels(double(Test_features));
    
    estimated_TrainLabelErrors_on_Tf_Tl = SoftMax_TrainFeatures_TrainLabels(double(Train_features));
    
    ErrorResidual = Train_Labels - estimated_TrainLabelErrors_on_Tf_Tl;
    
    SoftMax_TrainFeatures_ErrorResidual = trainSoftmaxLayer(double(Train_features),ErrorResidual, 'ShowProgressWindow', false);
    
    estimate_Residual_on_Tef = SoftMax_TrainFeatures_ErrorResidual(Test_features);
    
    final_estimate = estimate_Residual_on_Tef + estimated_TestFeatures_on_Tef_Tl;
    
    % following lines are for OTSU method
%     threshold_Value = graythresh(final_estimate);
%     estimated = final_estimate;
%     estimated(estimated < threshold_Value) = 0;
%     estimated(estimated > 0              ) = 1;

    % following lines are for OTSU method with im2bw() method (similar
    % results to manual method.
%     threshold_Value = graythresh(final_estimate);
%     estimated = final_estimate;
%     estimated = im2bw(estimated, threshold_Value);
    
    % following lines are for Kittler and Illingworth method
%     threshold_Value = kittler(final_estimate);
%     estimated = final_estimate;
%     estimated(estimated < threshold_Value) = 0;
%     estimated(estimated > 0              ) = 1;

    % following lines are for multithresh()
%     threshold_Value = multithresh(final_estimate);
%     estimated = final_estimate;
%     estimated = im2bw(estimated, threshold_Value);    
    
    % following lines are for adaptthresh()
%     threshold_Value = adaptthresh(final_estimate);
%     estimated = final_estimate;
%     estimated = imbinarize(estimated, threshold_Value);
    
%     This area is for manual threshold finding.
%     disp('Finding Threshold');
%     threshold = (0:0.00001:0.1);
%     
%     for i=1:length(threshold)
%         estimated = final_estimate;
%         
%         estimated(estimated < threshold(i)) = 0;
%         estimated(estimated > 0           ) = 1;
%         
%         evals = Evaluate(Test_Labels(:),estimated(:));
%         
%         acc(i) = evals(1);
%         sen(i) = evals(2);
%         spe(i) = evals(3);
%     end
%     index = find(acc == max(acc), 1);
% %     plot(threshold, acc, threshold, sen, threshold, spe);
% %     index = input('Put value: ');
%     estimated = final_estimate; 
%     estimated(estimated < threshold(index)) = 0;
%     estimated(estimated > 0               ) = 1;
%  
    evals = Evaluate(Test_Labels(:),estimated(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
end