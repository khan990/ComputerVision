



function NewAntiResidualExperiment(Train_features, Test_features, Train_Labels, Test_Labels)

    SoftMax_Basic = trainSoftmaxLayer(Train_features,Train_Labels, 'ShowProgressWindow', false);
    EstimateOnTraining = SoftMax_Basic(Train_features);
    EstimateOnTesting = SoftMax_Basic(Test_features);
    
    ErrorOnTraining = Train_Labels - EstimateOnTraining;

    SoftMaxOnTrainingError = trainSoftmaxLayer(Train_features, ErrorOnTraining, 'ShowProgressWindow', false);
    EstimateOnTrainingError = SoftMaxOnTrainingError(Train_features);
    EstimateOnTestingError = SoftMaxOnTrainingError(Test_features);
    
%     Compare with Test_Labels
    PostResidual_EstimateOnTesting = EstimateOnTesting + EstimateOnTestingError;
    
%     Compute threshold out of it.
    PostResidual_EstimateOnTraining = EstimateOnTraining + EstimateOnTrainingError;
    
%     Original SoftMax
    fprintf('Simple SoftMax\n----------------------\n');
    Final_Test_Estimate = EstimateOnTesting;
    Final_Test_Estimate(Final_Test_Estimate < 0.0001) = 0;
    Final_Test_Estimate(Final_Test_Estimate > 0     ) = 1;
    evals = Evaluate(Test_Labels(:), Final_Test_Estimate(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n\n\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    fprintf('\n');
%     ------------------
%     SoftMax after Error Correction
    fprintf('After Error Correction\n----------------------\n');
    Post_PostResidual_EstimateOnTesting = PostResidual_EstimateOnTesting;
    Post_PostResidual_EstimateOnTesting(Post_PostResidual_EstimateOnTesting < 0.0001) = 0;
    Post_PostResidual_EstimateOnTesting(Post_PostResidual_EstimateOnTesting > 0     ) = 1;
    evals = Evaluate(Test_Labels(:), Post_PostResidual_EstimateOnTesting(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n\n\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    fprintf('\n');
%     -------------------
%     Finding Best Threshold Point...

%     threshold = (0:0.0001:0.15);
%     threshold = (0.001:0.002:0.03);
    threshold = (0:0.0001:0.9);
    Kfold_x = 5;
    
    acc1 = zeros(length(threshold), 1);
    sen1 = zeros(length(threshold), 1);
    spe1 = zeros(length(threshold), 1);

    acc2 = zeros(length(threshold), 1);
    sen2 = zeros(length(threshold), 1);
    spe2 = zeros(length(threshold), 1);

    acc3 = zeros(length(threshold), 1);
    sen3 = zeros(length(threshold), 1);
    spe3 = zeros(length(threshold), 1);

    acc4 = zeros(length(threshold), 1);
    sen4 = zeros(length(threshold), 1);
    spe4 = zeros(length(threshold), 1);

    acc5 = zeros(length(threshold), 1);
    sen5 = zeros(length(threshold), 1);
    spe5 = zeros(length(threshold), 1);

    for k=1:Kfold_x
        fprintf('Fold = %d\n', k);
        
        switch (k)
            case 1
                Training_feature_temp = Train_features;
                Training_feature_temp(:, 1:100) = [];

                Training_Label_temp = Train_Labels;
                Training_Label_temp(:, (1):100) = [];
                
                SoftMax_ErrorOnTraining_temp = trainSoftmaxLayer(Training_feature_temp, Training_Label_temp, 'ShowProgressWindow', false);
                EstimateOnTraining_temp = SoftMax_ErrorOnTraining_temp(Training_feature_temp);
                
                ErrorOnTraining_temp = Training_Label_temp - EstimateOnTraining_temp;

            case 2
                Training_feature_temp = Train_features;
                Training_feature_temp(:, 101:200) = [];

                Training_Label_temp = Train_Labels;
                Training_Label_temp(:, (101):200) = [];

                SoftMax_ErrorOnTraining_temp = trainSoftmaxLayer(Training_feature_temp, Training_Label_temp, 'ShowProgressWindow', false);
                EstimateOnTraining_temp = SoftMax_ErrorOnTraining_temp(Training_feature_temp);
                
                ErrorOnTraining_temp = Training_Label_temp - EstimateOnTraining_temp;
            case 3
                Training_feature_temp = Train_features;
                Training_feature_temp(:, 201:300) = [];

                Training_Label_temp = Train_Labels;
                Training_Label_temp(:, (201):300) = [];

                SoftMax_ErrorOnTraining_temp = trainSoftmaxLayer(Training_feature_temp, Training_Label_temp, 'ShowProgressWindow', false);
                EstimateOnTraining_temp = SoftMax_ErrorOnTraining_temp(Training_feature_temp);
                
                ErrorOnTraining_temp = Training_Label_temp - EstimateOnTraining_temp;
                
            case 4
                Training_feature_temp = Train_features;
                Training_feature_temp(:, 301:400) = [];

                Training_Label_temp = Train_Labels;
                Training_Label_temp(:, (301):400) = [];

                SoftMax_ErrorOnTraining_temp = trainSoftmaxLayer(Training_feature_temp, Training_Label_temp, 'ShowProgressWindow', false);
                EstimateOnTraining_temp = SoftMax_ErrorOnTraining_temp(Training_feature_temp);
                
                ErrorOnTraining_temp = Training_Label_temp - EstimateOnTraining_temp;
                
            case 5
                Training_feature_temp = Train_features;
                Training_feature_temp(:, 401:500) = [];

                Training_Label_temp = Train_Labels;
                Training_Label_temp(:, (401):500) = [];

                SoftMax_ErrorOnTraining_temp = trainSoftmaxLayer(Training_feature_temp, Training_Label_temp, 'ShowProgressWindow', false);
                EstimateOnTraining_temp = SoftMax_ErrorOnTraining_temp(Training_feature_temp);
                
                ErrorOnTraining_temp = Training_Label_temp - EstimateOnTraining_temp;
                
        end
        
        SoftMax_ErrorOnTraining = trainSoftmaxLayer(Training_feature_temp, ErrorOnTraining_temp, 'ShowProgressWindow', false);
        EstimateOnErrorOnTraining = SoftMax_ErrorOnTraining(Training_feature_temp);
        
        i=1;
        for j = threshold
            Final_EstimateOnTraining_temp = EstimateOnTraining_temp + EstimateOnErrorOnTraining;
            Final_EstimateOnTraining_temp(Final_EstimateOnTraining_temp < j) = 0;
            Final_EstimateOnTraining_temp(Final_EstimateOnTraining_temp > 0) = 1;
            evals = Evaluate(Training_Label_temp(:),Final_EstimateOnTraining_temp(:));
            switch (k)
                case 1
%                     disp('case 1');
                    acc1(i) = evals(1);
                    sen1(i) = evals(2);
                    spe1(i) = evals(3);
                case 2
%                     disp('case 2');
                    acc2(i) = evals(1);
                    sen2(i) = evals(2);
                    spe2(i) = evals(3);
                case 3
%                     disp('case 3');
                    acc3(i) = evals(1);
                    sen3(i) = evals(2);
                    spe3(i) = evals(3);
                case 4
%                     disp('case 4');
                    acc4(i) = evals(1);
                    sen4(i) = evals(2);
                    spe4(i) = evals(3);
                case 5
%                     disp('case 5');
                    acc5(i) = evals(1);
                    sen5(i) = evals(2);
                    spe5(i) = evals(3);
            end
            i = i + 1;
        end
    end
    
    accavg = zeros(length(threshold), 1);
    senavg = zeros(length(threshold), 1);
    speavg = zeros(length(threshold), 1);
    
    for i=1:length(threshold)
        accavg(i) = ( acc1(i) + acc2(i) + acc3(i) + acc4(i) + acc5(i) )/5;
        senavg(i) = ( sen1(i) + sen2(i) + sen3(i) + sen4(i) + sen5(i) )/5;
        speavg(i) = ( spe1(i) + spe2(i) + spe3(i) + spe4(i) + spe5(i) )/5;
    end
    
    disp('End');
    figure;
    hold on;
    title('Accuracy Average');
    plot(threshold, accavg);
    hold off;
    
    figure;
    hold on;
    title('Sensitivity Average');
    plot(threshold, senavg);
    hold off;
    
    figure;
    hold on;
    title('Specificity Average');
    plot(threshold, speavg);
    hold off;
    
    
end


%                 ErrorOnTraining_temp = ErrorOnTraining;
%                 ErrorOnTraining_temp(:, (1):100) = [];
                
                
%                 EstimateOnTraining_temp = EstimateOnTraining;
%                 EstimateOnTraining_temp(:, (1):100) = [];                
