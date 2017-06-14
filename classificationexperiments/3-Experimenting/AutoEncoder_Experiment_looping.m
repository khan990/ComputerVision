


function AutoEncoder_Experiment_looping(Train_Images, Test_Images, Train_Labels, Test_Labels)
    diary('AutoEncoder_Experiment_looping3.txt');
    disp('Time of Image size Conversion. 1000 Images...');

    for i=1:500
        Train_Images_bw{i} = rgb2gray(Train_Images{i});
        Test_Images_bw{i} = rgb2gray(Test_Images{i});
    end
    
    
    Train_Images_autoEncoder_this = Train_Images_bw;
    Test_Images_autoEncoder_this = Test_Images_bw;
    
%     hiddensize = {10 100 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000};
%     hiddensize_1 = 100;
%     hiddensize_2 = 100;
%     hiddensize_3 = 100;
%     hiddensize_4 = 100;
%     hiddensize_5 = 100; 
%     
    hiddensize_1 = 1500;
    hiddensize_2 = 1250;
    hiddensize_3 = 1000;
    hiddensize_4 = 500;
    hiddensize_5 = 100; 
    
    EncoderTransferFunction_string = 'EncoderTransferFunction';
    EncoderTransferFunction = {'logsig' 'satlin'};
    
    DecoderTransferFunction_string = 'DecoderTransferFunction';
    DecoderTransferFunction = {'logsig' 'satlin' 'purelin'};
    
    L2WeightRegularization_string = 'L2WeightRegularization';
    L2WeightRegularization = 0.1:0.2:1;
    
    LossFunction_string = 'LossFunction';
    LossFunction = {'msesparse'};
    
    SparsityProportion_string = 'SparsityProportion';
    SparsityProportion = 0:0.2:1; 
    
    SparsityRegularization_string = 'SparsityRegularization';
    SparsityRegularization = 0:2:10;
    
    TrainingAlgorithm_string = 'TrainingAlgorithm';
    TrainingAlgorithm = {'trainscg'};
%     TrainingAlgorithm = {'trainlm' 'trainbfg' 'trainrp' 'trainscg' 'traincgb' 'traincgf' 'traincgp' 'trainoss' 'traingdx'};

    for I = 1:length(EncoderTransferFunction)
        for J = 1:length(DecoderTransferFunction)
            for K = 1:length(L2WeightRegularization)
                for L = 1:length(LossFunction)
                    for M = 1:length(SparsityProportion)
                        for N = 1:length(SparsityRegularization)
                            for O = 1:length(TrainingAlgorithm)
                                % AutoEncoder 1 
                                disp('AutoEncoder 1');
                                AutoEncoder1 = trainAutoencoder(Train_Images_autoEncoder_this, hiddensize_1, 'UseGPU', true, 'ShowProgressWindow', true, ...
                                EncoderTransferFunction_string, EncoderTransferFunction{I}, DecoderTransferFunction_string, DecoderTransferFunction{J}, ...
                                L2WeightRegularization_string, L2WeightRegularization(K), LossFunction_string, LossFunction{L}, SparsityProportion_string, ...
                                SparsityProportion(M), SparsityRegularization_string, SparsityRegularization(N), TrainingAlgorithm_string, TrainingAlgorithm{O});

                                AutoEncoder1_encode = encode(AutoEncoder1, Train_Images_autoEncoder_this);

                                % AutoEncoder 2
                                disp('AutoEncoder 2');
                                AutoEncoder2 = trainAutoencoder(AutoEncoder1_encode, hiddensize_2, 'UseGPU', true, 'ShowProgressWindow', true, ...
                                EncoderTransferFunction_string, EncoderTransferFunction{I}, DecoderTransferFunction_string, DecoderTransferFunction{J}, ...
                                L2WeightRegularization_string, L2WeightRegularization(K), LossFunction_string, LossFunction{L}, SparsityProportion_string, ...
                                SparsityProportion(M), SparsityRegularization_string, SparsityRegularization(N), TrainingAlgorithm_string, TrainingAlgorithm{O});
                                AutoEncoder2_encode = encode(AutoEncoder2, AutoEncoder1_encode);

                                % AutoEncoder 3
                                disp('AutoEncoder 3');
                                AutoEncoder3 = trainAutoencoder(AutoEncoder2_encode, hiddensize_3, 'UseGPU', true, 'ShowProgressWindow', true, ...
                                EncoderTransferFunction_string, EncoderTransferFunction{I}, DecoderTransferFunction_string, DecoderTransferFunction{J}, ...
                                L2WeightRegularization_string, L2WeightRegularization(K), LossFunction_string, LossFunction{L}, SparsityProportion_string, ...
                                SparsityProportion(M), SparsityRegularization_string, SparsityRegularization(N), TrainingAlgorithm_string, TrainingAlgorithm{O});
                                AutoEncoder3_encode = encode(AutoEncoder3, AutoEncoder2_encode);

                                % AutoEncoder 4
                                disp('AutoEncoder 4');
                                AutoEncoder4 = trainAutoencoder(AutoEncoder3_encode, hiddensize_4, 'UseGPU', true, 'ShowProgressWindow', true, ...
                                EncoderTransferFunction_string, EncoderTransferFunction{I}, DecoderTransferFunction_string, DecoderTransferFunction{J}, ...
                                L2WeightRegularization_string, L2WeightRegularization(K), LossFunction_string, LossFunction{L}, SparsityProportion_string, ...
                                SparsityProportion(M), SparsityRegularization_string, SparsityRegularization(N), TrainingAlgorithm_string, TrainingAlgorithm{O});
                                AutoEncoder4_encode = encode(AutoEncoder4, AutoEncoder3_encode);

                                % AutoEncoder 5
                                disp('AutoEncoder 5');
                                AutoEncoder5 = trainAutoencoder(AutoEncoder4_encode, hiddensize_5, 'UseGPU', true, 'ShowProgressWindow', true, ...
                                EncoderTransferFunction_string, EncoderTransferFunction{I}, DecoderTransferFunction_string, DecoderTransferFunction{J}, ...
                                L2WeightRegularization_string, L2WeightRegularization(K), LossFunction_string, LossFunction{L}, SparsityProportion_string, ...
                                SparsityProportion(M), SparsityRegularization_string, SparsityRegularization(N), TrainingAlgorithm_string, TrainingAlgorithm{O});
                                AutoEncoder5_encode = encode(AutoEncoder5, AutoEncoder4_encode);

                                disp('softmax');
                                softnet_x = trainSoftmaxLayer(double(AutoEncoder5_encode), double(Train_Labels));

                            %     5 stacks
                                deep = stack(AutoEncoder1, AutoEncoder2, AutoEncoder3, AutoEncoder4, AutoEncoder5, softnet_x);

                            %     4 stacks
                            %     softnet_x = trainSoftmaxLayer(double(AutoEncoder4_encode), double(Train_Labels));
                            %     deep = stack(AutoEncoder1, AutoEncoder2, AutoEncoder3, AutoEncoder4, softnet_x);

                            %     3 stacks
                            %     softnet_x = trainSoftmaxLayer(double(AutoEncoder3_encode), double(Train_Labels));
                            %     deep = stack(AutoEncoder1, AutoEncoder2, AutoEncoder3, softnet_x);

                            %     2 stacks 
                            %     softnet_x = trainSoftmaxLayer(double(AutoEncoder2_encode), double(Train_Labels));
                            %     deep = stack(AutoEncoder1, AutoEncoder2, softnet_x);

                            %     1 stack
                            %     softnet_x = trainSoftmaxLayer(double(AutoEncoder1_encode), double(Train_Labels));
                            %     deep = stack(AutoEncoder1, softnet_x);
                            % 
                            %     

                                xTrain = zeros(64*48 ,numel(Train_Images_autoEncoder_this));
                                for i = 1:numel(Train_Images_autoEncoder_this)
                                    xTrain(:,i) = Train_Images_autoEncoder_this{i}(:);
                                end

                                xTest = zeros(64*48 ,numel(Test_Images_autoEncoder_this));
                                for i = 1:numel(Test_Images_autoEncoder_this)
                                    xTest(:,i) = Test_Images_autoEncoder_this{i}(:);
                                end


                                deepnet = train(deep, xTrain, Train_Labels, 'useGPU','yes');
                                disp('Time of estimated values for 500 images...');
                                tic
                                estimated_classes = deepnet(xTest);
                                toc




                            %     This area is for manual threshold finding.
                                disp('Finding Threshold');
                                threshold = (0:0.00001:0.1);

                                for i=1:length(threshold)
                                    estimated = estimated_classes;

                                    estimated(estimated < threshold(i)) = 0;
                                    estimated(estimated > 0           ) = 1;

                                    evals = Evaluate(Test_Labels(:),estimated(:));

                                    acc(i) = evals(1);
                                    sen(i) = evals(2);
                                    spe(i) = evals(3);
                                end
                                index = find(acc == max(acc), 1);
                                 plot(threshold, acc, threshold, sen, threshold, spe);
                            %     index = input('Put value: ');
                                estimated = estimated_classes; 
                                estimated(estimated < threshold(index)) = 0;
                                estimated(estimated > 0               ) = 1;

                                evals = Evaluate(Test_Labels(:),estimated(:));
                            %     fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
                                disp('------------');
                                average = (evals(2) + evals(3))/2;
                                fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
                                disp('------------');
                                disp('Loop Values');
                                fprintf('%s\t%s\n%s\t%s\n%s\t%f\n%s\t%s\n%s\t%f\n%s\t%f\n%s\t%s\n', ...
                                EncoderTransferFunction_string, EncoderTransferFunction{I}, DecoderTransferFunction_string, DecoderTransferFunction{J}, ...
                                L2WeightRegularization_string, L2WeightRegularization(K), LossFunction_string, LossFunction{L}, SparsityProportion_string, ...
                                SparsityProportion(M), SparsityRegularization_string, SparsityRegularization(N), TrainingAlgorithm_string, TrainingAlgorithm{O});
                            end
                        end
                    end
                end
            end
        end
    end
    diary off;
end