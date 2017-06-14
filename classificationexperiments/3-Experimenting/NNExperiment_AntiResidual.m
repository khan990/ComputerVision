




% [NN_on_Train_features_Train_Labels, NN_on_ErrorResidual_on_Training] = NNExperiment_AntiResidual(Train_CNN_features, Test_CNN_features, Train_Labels_New, Test_Labels_New)
function [NN_on_Train_features_Train_Labels, NN_on_ErrorResidual_on_Training] = NNExperiment_AntiResidual(Train_features, Test_features, Train_Labels, Test_Labels)
%     diary('NN_on_ErrorResidual_on_Training2.txt');
    Train_features = single(Train_features);
    Test_features = single(Test_features);
    Train_Labels = single(Train_Labels);
    Test_Labels = single(Test_Labels);
    
    AllActivationFunctions = {'radbasn'};
%     AllActivationFunctions = {'compet' 'elliotsig' 'hardlim' 'hardlims' 'hardlims' 'logsig' 'poslin' 'purelin' 'radbas' 'radbasn' 'satlin' 'satlins' 'softmax' 'tansig' 'tribas'};
%     AllTrainFunctions = {'trainlm' 'trainbr' 'trainbfg' 'traincgb' 'traincgf' 'traincgp' 'traingd' 'traingda' 'traingdm' 'traingdx' 'trainoss' 'trainrp' 'trainscg' 'trainb' 'trainc' 'trainr' 'trains'};
    
    for j=1:length(AllActivationFunctions)
%         for k=1:length(AllTrainFunctions)
            try
            %     NN_on_Train_features_Train_Labels_temp = fitnet(10);
                NN_on_Train_features_Train_Labels_temp = feedforwardnet(10);
                NN_on_Train_features_Train_Labels_temp.layers{1}.transferFcn = AllActivationFunctions{j};
%                 NN_on_Train_features_Train_Labels_temp.trainFcn = AllTrainFunctions{k};
            %     NN_on_Train_features_Train_Labels = train(NN_on_Train_features_Train_Labels_temp, Train_features, Train_Labels, 'useParallel','yes','useGPU','yes','showResources','yes');
                NN_on_Train_features_Train_Labels_temp = configure(NN_on_Train_features_Train_Labels_temp, Train_features, Train_Labels);
                NN_on_Train_features_Train_Labels_temp = init(NN_on_Train_features_Train_Labels_temp);

                Train_features_GPU = nndata2gpu(Train_features);
                Train_Labels_GPU = nndata2gpu(Train_Labels);
                NN_on_Train_features_Train_Labels{j} = train(NN_on_Train_features_Train_Labels_temp, Train_features_GPU, Train_Labels_GPU,'useGPU','yes','showResources','yes');

                Test_features_GPU = nndata2gpu(Test_features);
                ErrorEstimate_on_Train_features = NN_on_Train_features_Train_Labels{j}(Test_features_GPU);
                ErrorEstimate_on_Train_features = gpu2nndata(ErrorEstimate_on_Train_features);


                ErrorResidual_on_Training = Train_Labels - ErrorEstimate_on_Train_features;

            %     NN_on_ErrorResidual_on_Training_temp = fitnet(10);
                NN_on_ErrorResidual_on_Training_temp = feedforwardnet(10);
                NN_on_ErrorResidual_on_Training_temp.layers{1}.transferFcn = AllActivationFunctions{j};
            %     NN_on_ErrorResidual_on_Training = train(NN_on_ErrorResidual_on_Training_temp, Train_features, ErrorResidual_on_Training, 'useParallel','yes','useGPU','yes','showResources','yes');
                NN_on_ErrorResidual_on_Training_temp = configure(NN_on_ErrorResidual_on_Training_temp, Train_features, Train_Labels);
                ErrorResidual_on_Training_GPU = nndata2gpu(ErrorResidual_on_Training);
                NN_on_ErrorResidual_on_Training{j} = train(NN_on_ErrorResidual_on_Training_temp, Train_features_GPU, ErrorResidual_on_Training_GPU,'useGPU','yes','showResources','yes');

                ErrorResidual_for_Test_features_on_Trained_NN = NN_on_ErrorResidual_on_Training{j}(Test_features_GPU);

                estimated = NN_on_Train_features_Train_Labels{j}(Test_features_GPU);

                ErrorResidual_for_Test_features_on_Trained_NN = gpu2nndata(ErrorResidual_for_Test_features_on_Trained_NN);
                estimated = gpu2nndata(estimated);
                Purified_estimate = estimated + ErrorResidual_for_Test_features_on_Trained_NN;

                threshold = (-2:0.001:2);
                acc = zeros(length(threshold), 1);
                sen = zeros(length(threshold), 1);
                spe = zeros(length(threshold), 1);

                for i=1:length(threshold)
                    NonCorrected_estimated = ErrorEstimate_on_Train_features;
                    NonCorrected_estimated(NonCorrected_estimated < threshold(i)) = 0;
                    NonCorrected_estimated(NonCorrected_estimated > 0           ) = 1;
                    evals = Evaluate(Test_Labels, NonCorrected_estimated);
                    acc(i) = evals(1);
                    sen(i) = evals(2);
                    spe(i) = evals(3);
                end
%                 plot(threshold, acc, threshold, sen, threshold, spe);
%                 hold on;
%                 title('Non Corrected...');
%                 legend('Accuracy', 'Sensitivity', 'Specificity');
%                 hold off;

%                 final_threshold = input('input final threshold value: ');
                index = find(acc == max(acc), 1);

                NonCorrected_estimated = ErrorEstimate_on_Train_features;
                NonCorrected_estimated(NonCorrected_estimated < threshold(index)) = 0;
                NonCorrected_estimated(NonCorrected_estimated > 0              ) = 1;

                fprintf('\nFinal Threshold Value for NonCorrected is %f\n', threshold(index));
                evals = Evaluate(Test_Labels, NonCorrected_estimated);
                fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n',evals(1), evals(2), evals(3));

                acc = 0;
                sen = 0;
                spe = 0;

                for i=1:length(threshold)
                    post_estimated = Purified_estimate;
                    post_estimated(post_estimated < threshold(i)) = 0;
                    post_estimated(post_estimated > 0           ) = 1;
                    evals = Evaluate(Test_Labels, post_estimated);
                    acc(i) = evals(1);
                    sen(i) = evals(2);
                    spe(i) = evals(3);
                end

%                 plot(threshold, acc, threshold, sen, threshold, spe);
%                 hold on;
%                 title('Corrected...');
%                 legend('Accuracy', 'Sensitivity', 'Specificity');
%                 hold off;

%                 final_threshold = input('input final threshold value: ');
                index = find(acc == max(acc), 1);

                post_estimated = Purified_estimate;
                post_estimated(post_estimated < threshold(index)) = 0;
                post_estimated(post_estimated > 0              ) = 1;

                fprintf('Combination(TrainFunction = %s, LayerTransferFunction = %s)', 'trainscg', AllActivationFunctions{j});
                fprintf('\nFinal Threshold Value for CORRECTED is %f\n', threshold(index));
                evals = Evaluate(Test_Labels, post_estimated);
                fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n',evals(1), evals(2), evals(3));
            catch
                fprintf('Combination(TrainFunction = %s, LayerTransferFunction = %s)\n', 'trainscg', AllActivationFunctions{j});
                warning('Ignore Last combination...');
            end
%         end
    end
%     diary off;
end