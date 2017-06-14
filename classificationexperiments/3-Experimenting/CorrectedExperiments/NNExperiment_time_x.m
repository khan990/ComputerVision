




% [NN_on_Train_features_Train_Labels, NN_on_ErrorResidual_on_Training] = NNExperiment_AntiResidual(Train_CNN_features, Test_CNN_features, Train_Labels_New, Test_Labels_New)
function [NN_on_Train_features_Train_Labels, NN_on_ErrorResidual_on_Training] = NNExperiment_time_x(Train_features, Test_features, Train_Labels, Test_Labels)
    diary('NN_time.txt');
    Train_features = single(Train_features);
    Test_features = single(Test_features);
    Train_Labels = single(Train_Labels);
    Test_Labels = single(Test_Labels);
    
%     AllActivationFunctions = {'elliotsig'};
    AllActivationFunctions = {'compet' 'elliotsig' 'hardlim' 'hardlims' 'hardlims' 'logsig' 'poslin' 'purelin' 'radbas' 'radbasn' 'satlin' 'satlins' 'softmax' 'tansig' 'tribas'};
    AllTrainFunctions = {'trainlm' 'trainbr' 'trainbfg' 'traincgb' 'traincgf' 'traincgp' 'traingd' 'traingda' 'traingdm' 'traingdx' 'trainoss' 'trainrp' 'trainscg' 'trainb' 'trainc' 'trainr' 'trains'};
    
    for j=1:length(AllActivationFunctions)
        for k=1:length(AllTrainFunctions)
            try
            %     NN_on_Train_features_Train_Labels_temp = fitnet(10);
                NN_on_Train_features_Train_Labels_temp = feedforwardnet(10);
                NN_on_Train_features_Train_Labels_temp.layers{1}.transferFcn = AllActivationFunctions{j};
                NN_on_Train_features_Train_Labels_temp.trainFcn = AllTrainFunctions{k};
            %     NN_on_Train_features_Train_Labels = train(NN_on_Train_features_Train_Labels_temp, Train_features, Train_Labels, 'useParallel','yes','useGPU','yes','showResources','yes');
                NN_on_Train_features_Train_Labels_temp = configure(NN_on_Train_features_Train_Labels_temp, Train_features, Train_Labels);
                NN_on_Train_features_Train_Labels_temp = init(NN_on_Train_features_Train_Labels_temp);

                Train_features_GPU = nndata2gpu(Train_features);
                Train_Labels_GPU = nndata2gpu(Train_Labels);
                NN_on_Train_features_Train_Labels{j} = train(NN_on_Train_features_Train_Labels_temp, Train_features_GPU, Train_Labels_GPU,'useGPU','yes');

%                 Test_features_GPU = nndata2gpu(Test_features);
%                 ErrorEstimate_on_Train_features = NN_on_Train_features_Train_Labels{j}(Train_features_GPU);
%                 ErrorEstimate_on_Train_features = gpu2nndata(ErrorEstimate_on_Train_features);


%                 ErrorResidual_on_Training = Train_Labels - ErrorEstimate_on_Train_features;

            %     NN_on_ErrorResidual_on_Training_temp = fitnet(10);
%                 NN_on_ErrorResidual_on_Training_temp = feedforwardnet(10);
%                 NN_on_ErrorResidual_on_Training_temp.layers{1}.transferFcn = AllActivationFunctions{j};
%                 NN_on_ErrorResidual_on_Training_temp.trainFcn = AllTrainFunctions{k};
            %     NN_on_ErrorResidual_on_Training = train(NN_on_ErrorResidual_on_Training_temp, Train_features, ErrorResidual_on_Training, 'useParallel','yes','useGPU','yes','showResources','yes');
%                 NN_on_ErrorResidual_on_Training_temp = configure(NN_on_ErrorResidual_on_Training_temp, Train_features, ErrorResidual_on_Training);
%                 ErrorResidual_on_Training_GPU = nndata2gpu(ErrorResidual_on_Training);
%                 NN_on_ErrorResidual_on_Training{j} = train(NN_on_ErrorResidual_on_Training_temp, Train_features_GPU, ErrorResidual_on_Training_GPU,'useGPU','yes');

%                 ErrorResidual_for_Test_features_on_Trained_NN = NN_on_ErrorResidual_on_Training{j}(Test_features_GPU);
                tic
                estimated = NN_on_Train_features_Train_Labels{j}(Test_features);
                toc

%                 ErrorResidual_for_Test_features_on_Trained_NN = gpu2nndata(ErrorResidual_for_Test_features_on_Trained_NN);
%                 estimated = gpu2nndata(estimated);
                
%                 Purified_estimate = estimated + ErrorResidual_for_Test_features_on_Trained_NN;

                
                
                
                
                
                
                
                Kfolds = 5;
                [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_features, Train_Labels, Kfolds);

                disp('Finding Threshold');
                threshold = (0:0.0001:1);

                acc = zeros(length(threshold), Kfolds);
                sen = zeros(length(threshold), Kfolds);
                spe = zeros(length(threshold), Kfolds);
                
                acc2 = zeros(length(threshold), Kfolds);
                sen2 = zeros(length(threshold), Kfolds);
                spe2 = zeros(length(threshold), Kfolds);

                for x=1:Kfolds
                    fprintf('Kfold = %d\n', x);
                    
                    ft_NN_on_Train_features_Train_Labels_temp = feedforwardnet(10);
                    ft_NN_on_Train_features_Train_Labels_temp.layers{1}.transferFcn = AllActivationFunctions{j};
                    ft_NN_on_Train_features_Train_Labels_temp.trainFcn = AllTrainFunctions{k};
                %     NN_on_Train_features_Train_Labels = train(NN_on_Train_features_Train_Labels_temp, Train_features, Train_Labels, 'useParallel','yes','useGPU','yes','showResources','yes');
                    ft_NN_on_Train_features_Train_Labels_temp = configure(ft_NN_on_Train_features_Train_Labels_temp, Tr_features{x}, Tr_Labels{x});
                    ft_NN_on_Train_features_Train_Labels_temp = init(ft_NN_on_Train_features_Train_Labels_temp);

                    ft_Train_features_GPU = nndata2gpu(Tr_features{x});
                    ft_Train_Labels_GPU = nndata2gpu(Tr_Labels{x});
                    ft_NN_on_Train_features_Train_Labels{j} = train(ft_NN_on_Train_features_Train_Labels_temp, ft_Train_features_GPU, ft_Train_Labels_GPU,'useGPU','yes');

%                     ft_Test_features_GPU = nndata2gpu(Te_features{x});
%                     ft_ErrorEstimate_on_Train_features = ft_NN_on_Train_features_Train_Labels{j}(ft_Train_features_GPU);
%                     ft_ErrorEstimate_on_Train_features = gpu2nndata(ft_ErrorEstimate_on_Train_features);


%                     ft_ErrorResidual_on_Training = Tr_Labels{x} - ft_ErrorEstimate_on_Train_features;

                %     NN_on_ErrorResidual_on_Training_temp = fitnet(10);
%                     ft_NN_on_ErrorResidual_on_Training_temp = feedforwardnet(10);
%                     ft_NN_on_ErrorResidual_on_Training_temp.layers{1}.transferFcn = AllActivationFunctions{j};
%                     ft_NN_on_ErrorResidual_on_Training_temp.trainFcn = AllTrainFunctions{k};
                %     NN_on_ErrorResidual_on_Training = train(NN_on_ErrorResidual_on_Training_temp, Train_features, ErrorResidual_on_Training, 'useParallel','yes','useGPU','yes','showResources','yes');
%                     ft_NN_on_ErrorResidual_on_Training_temp = configure(ft_NN_on_ErrorResidual_on_Training_temp, Tr_features{x}, ft_ErrorResidual_on_Training);
%                     ft_ErrorResidual_on_Training_GPU = nndata2gpu(ft_ErrorResidual_on_Training);
%                     ft_NN_on_ErrorResidual_on_Training{j} = train(ft_NN_on_ErrorResidual_on_Training_temp, ft_Train_features_GPU, ft_ErrorResidual_on_Training_GPU,'useGPU','yes');

%                     ft_ErrorResidual_for_Test_features_on_Trained_NN = ft_NN_on_ErrorResidual_on_Training{j}(ft_Test_features_GPU);
                    
                    ft_estimated = ft_NN_on_Train_features_Train_Labels{j}(Te_features{x});

%                     ft_ErrorResidual_for_Test_features_on_Trained_NN = gpu2nndata(ft_ErrorResidual_for_Test_features_on_Trained_NN);
%                     ft_estimated = gpu2nndata(ft_estimated);

%                     ft_Purified_estimate = ft_estimated + ft_ErrorResidual_for_Test_features_on_Trained_NN;
                    
                    
                    
                    
                    for i=1:length(threshold)
%                         Non corrected one
                        ft_estimated_nonCorrected = ft_estimated;

                        ft_estimated_nonCorrected(ft_estimated_nonCorrected < threshold(i)) = 0;
                        ft_estimated_nonCorrected(ft_estimated_nonCorrected > 0           ) = 1;

                        evals = Evaluate(Te_Labels{x}(:),ft_estimated_nonCorrected(:));

                        acc(i, x) = evals(1);
                        sen(i, x) = evals(2);
                        spe(i, x) = evals(3);
% %                          Corrected one
%                         ft_estimated_Corrected = ft_Purified_estimate;
% 
%                         ft_estimated_Corrected(ft_estimated_Corrected < threshold(i)) = 0;
%                         ft_estimated_Corrected(ft_estimated_Corrected > 0           ) = 1;
% 
%                         evals = Evaluate(Te_Labels{x}(:),ft_estimated_Corrected(:));
% 
%                         acc2(i, x) = evals(1);
%                         sen2(i, x) = evals(2);
%                         spe2(i, x) = evals(3);
                    end
                end
                disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
                fprintf('Combination(TrainFunction = %s, LayerTransferFunction = %s)\n', AllTrainFunctions{k}, AllActivationFunctions{j});

                final_acc = mean(acc');
                final_sen = mean(sen');
                final_spe = mean(spe');
                
%                 figure;
%                 h1 = plot(threshold, final_acc, threshold, final_sen, threshold, final_spe);
%                 hold on;
%                 legend('Accuracy', 'Sensitivity', 'Specificity');
%                 hold off;
%                 saveas(h1, strcat(AllActivationFunctions{j}, '_nonCorrected.jpg'));
                
                index = find(final_acc == max(final_acc), 1);

                estimated_nonCorrected = estimated;

                estimated_nonCorrected(estimated_nonCorrected < threshold(index)) = 0;
                estimated_nonCorrected(estimated_nonCorrected > 0               ) = 1;
                disp('Non Corrected estimate...');
                evals = Evaluate(Test_Labels(:),estimated_nonCorrected(:));
                Print_Evaluations(evals);
                
                
                
                
                
%                 final_acc2 = mean(acc2');
%                 final_sen2 = mean(sen2');
%                 final_spe2 = mean(spe2');
                
%                 figure;
%                 h2 = plot(threshold, final_acc2, threshold, final_sen2, threshold, final_spe2);
%                 hold on;
%                 legend('Accuracy', 'Sensitivity', 'Specificity');
%                 hold off;
%                 saveas(h2, strcat(AllActivationFunctions{j}, '_Corrected.jpg'));
%                 
%                 index = find(final_acc2 == max(final_acc2), 1);
% 
%                 estimated_Corrected = Purified_estimate;
% 
%                 estimated_Corrected(estimated_Corrected < threshold(index)) = 0;
%                 estimated_Corrected(estimated_Corrected > 0               ) = 1;
%                 disp('Corrected estimate...');
%                 evals = Evaluate(Test_Labels(:),estimated_Corrected(:));
%                 Print_Evaluations(evals);
%                 
%                 disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
% %                 pause;
                
            catch
                fprintf('Combination(TrainFunction = %s, LayerTransferFunction = %s)\n', AllTrainFunctions{k}, AllActivationFunctions{j});
                warning('Ignore Last combination...');
            end
        end
    end
    diary off;
end