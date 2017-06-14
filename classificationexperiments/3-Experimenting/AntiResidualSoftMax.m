


% Train_Estimation = Estimate Training_Labels over Training_Features
% Test_Estimate = Estimate Test_Labels over Training_Features and
% Training_Labels
% ResidualError = Train_Labels - Train_Estimation
% Residual_Estimate = Estimate Test_Labels over Training_Features and ResidualError
% Residual_train_est = Estimate Train_Labels over Train_Features and ResidualError




function AntiResidualSoftMax(Train_features, Test_features, Train_Labels, Test_Labels)

    softnet = trainSoftmaxLayer(Train_features,Train_Labels, 'ShowProgressWindow', false);
    Train_Estimation = softnet(Train_features);
    
    Test_Estimate = softnet(Test_features);
    ResidualError = Train_Labels - Train_Estimation;
    
    
    SoftNet_Residual = trainSoftmaxLayer(Train_features, ResidualError, 'ShowProgressWindow', false);
    Residual_train_est = SoftNet_Residual(Train_features);
	
	Residual_Estimate = SoftNet_Residual(Test_features);
    
%  Final_Estimate with Test_Labels   
    Final_Estimate = Test_Estimate + Residual_Estimate;
% Just to compute threshold
	Final_Estimate_2 = Train_Estimation + Residual_train_est;

    i = 1;
    threshold = (0:0.0001:1);
    for j = threshold
        Final_Estimate_2 = Final_Estimate;
        Final_Estimate_2(Final_Estimate_2 < j) = 0;
        Final_Estimate_2(Final_Estimate_2 > 0) = 1;
        evals = Evaluate(Train_Labels(:),Final_Estimate_2(:));
        acc(i) = evals(1);
        sen(i) = evals(2);
        spe(i) = evals(3);
        i = i + 1;
%         fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n\n\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    end

%     i = 1;
%     threshold = (0:0.0001:0.15);
% %     threshold = (0.001:0.002:0.03);
% %     threshold = (0:0.0001:0.9);
%     Kfold_x = 5;
%     NumOfColumns = size(Train_features, 2) / Kfold_x;
%     
%     for k=1:Kfold_x
%         fprintf('Fold = %d\n', k);
%         
%         switch (k)
%             case 1
%                 Training_feature = Train_features;
%                 Training_feature(:, 1:100) = [];
% 
%                 Training_Label = Train_Labels;
%                 Training_Label(:, (1):100) = [];
% 
%                 ResidualingError = ResidualError;
%                 ResidualingError(:, (1):100) = [];
% 
%                 TrainingEstimation = Train_Estimation;
%                 TrainingEstimation(:, (1):100) = [];                
%             case 2
%                 Training_feature = Train_features;
%                 Training_feature(:, 101:200) = [];
% 
%                 Training_Label = Train_Labels;
%                 Training_Label(:, (101):200) = [];
% 
%                 ResidualingError = ResidualError;
%                 ResidualingError(:, (101):200) = [];
% 
%                 TrainingEstimation = Train_Estimation;
%                 TrainingEstimation(:, (101):200) = [];                
%             case 3
%                 Training_feature = Train_features;
%                 Training_feature(:, 201:300) = [];
% 
%                 Training_Label = Train_Labels;
%                 Training_Label(:, (201):300) = [];
% 
%                 ResidualingError = ResidualError;
%                 ResidualingError(:, (201):300) = [];
% 
%                 TrainingEstimation = Train_Estimation;
%                 TrainingEstimation(:, (201):300) = [];                
%             case 4
%                 Training_feature = Train_features;
%                 Training_feature(:, 301:400) = [];
% 
%                 Training_Label = Train_Labels;
%                 Training_Label(:, (301):400) = [];
% 
%                 ResidualingError = ResidualError;
%                 ResidualingError(:, (301):400) = [];
% 
%                 TrainingEstimation = Train_Estimation;
%                 TrainingEstimation(:, (301):400) = [];                
%             case 5
%                 Training_feature = Train_features;
%                 Training_feature(:, 401:500) = [];
% 
%                 Training_Label = Train_Labels;
%                 Training_Label(:, (401):500) = [];
% 
%                 ResidualingError = ResidualError;
%                 ResidualingError(:, (401):500) = [];
% 
%                 TrainingEstimation = Train_Estimation;
%                 TrainingEstimation(:, (401):500) = [];                
%         end
%         
% %         if k == 1
% %             Training_feature = Train_features;
% %             Training_feature(:, (1)*NumOfColumns:(1)*NumOfColumns + NumOfColumns + 1) = [];
% % 
% %             Training_Label = Train_Labels;
% %             Training_Label(:, (1)*NumOfColumns:(1)*NumOfColumns + NumOfColumns + 1) = [];
% % 
% %     %         Testing_feature = Train_features(:, (k-1)*NumOfColumns:(k-1)*NumOfColumns + NumOfColumns);
% %     %         Testing_Label = Train_Labels(:, (k-1)*NumOfColumns:(k-1)*NumOfColumns + NumOfColumns);
% % 
% %             ResidualingError = ResidualError;
% %             ResidualingError(:, (1)*NumOfColumns:(1)*NumOfColumns + NumOfColumns + 1) = [];
% %             
% %             TrainingEstimation = Train_Estimation;
% %             TrainingEstimation(:, (1)*NumOfColumns:(1)*NumOfColumns + NumOfColumns + 1) = [];
% %         else
% %             Training_feature = Train_features;
% %             Training_feature(:, (k-1)*NumOfColumns:(k-1)*NumOfColumns + NumOfColumns + 1) = [];
% % 
% %             Training_Label = Train_Labels;
% %             Training_Label(:, (k-1)*NumOfColumns:(k-1)*NumOfColumns + NumOfColumns + 1) = [];
% % 
% %     %         Testing_feature = Train_features(:, (k-1)*NumOfColumns:(k-1)*NumOfColumns + NumOfColumns);
% %     %         Testing_Label = Train_Labels(:, (k-1)*NumOfColumns:(k-1)*NumOfColumns + NumOfColumns);
% % 
% %             ResidualingError = ResidualError;
% %             ResidualingError(:, (k-1)*NumOfColumns:(k-1)*NumOfColumns + NumOfColumns + 1) = [];
% %             
% %             TrainingEstimation = Train_Estimation;
% %             TrainingEstimation(:, (k-1)*NumOfColumns:(k-1)*NumOfColumns + NumOfColumns + 1) = [];
% %         end
%         SoftNet_Residual = trainSoftmaxLayer(Training_feature, ResidualingError, 'ShowProgressWindow', false);
%         Residual_train_est = SoftNet_Residual(Training_feature);
%         Final_Estimate_2 = TrainingEstimation + Residual_train_est;
%         i=1;
%         for j = threshold
%             Final_Estimate_2 = TrainingEstimation + Residual_train_est;
%             Final_Estimate_2(Final_Estimate_2 < j) = 0;
%             Final_Estimate_2(Final_Estimate_2 > 0) = 1;
%             evals = Evaluate(Training_Label(:),Final_Estimate_2(:));
%             switch (k)
%                 case 1
% %                     disp('case 1');
%                     acc1(i) = evals(1);
%                     sen1(i) = evals(2);
%                     spe1(i) = evals(3);
%                 case 2
% %                     disp('case 2');
%                     acc2(i) = evals(1);
%                     sen2(i) = evals(2);
%                     spe2(i) = evals(3);
%                 case 3
% %                     disp('case 3');
%                     acc3(i) = evals(1);
%                     sen3(i) = evals(2);
%                     spe3(i) = evals(3);
%                 case 4
% %                     disp('case 4');
%                     acc4(i) = evals(1);
%                     sen4(i) = evals(2);
%                     spe4(i) = evals(3);
%                 case 5
% %                     disp('case 5');
%                     acc5(i) = evals(1);
%                     sen5(i) = evals(2);
%                     spe5(i) = evals(3);
%             end
%             i = i + 1;
%     %         fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n\n\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
%         end
%     end
%     
%     for i=1:length(threshold)
%         accavg(i) = ( acc1(i) + acc2(i) + acc3(i) + acc4(i) + acc5(i) )/5;
%         senavg(i) = ( sen1(i) + sen2(i) + sen3(i) + sen4(i) + sen5(i) )/5;
%         speavg(i) = ( spe1(i) + spe2(i) + spe3(i) + spe4(i) + spe5(i) )/5;
%     end
%     
%     figure;
%     hold on;
%     title('Accuracy Average');
%     plot(threshold, accavg);
%     hold off;
%     
%     figure;
%     hold on;
%     title('Sensitivity Average');
%     plot(threshold, senavg);
%     hold off;
%     
%     figure;
%     hold on;
%     title('Specificity Average');
%     plot(threshold, speavg);
%     hold off;
%     
%     

% compute threshold

    plot(threshold, acc, threshold, sen, threshold, spe);
    
    index = input('Threshold value: ');
    Final_Estimate_3 = Final_Estimate;
    Final_Estimate_3(Final_Estimate_3 < index) = 0;
    Final_Estimate_3(Final_Estimate_3 > 0) = 1;
%     threshold(max_index)
    evals = Evaluate(Test_Labels(:),Final_Estimate_3(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

end



