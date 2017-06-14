


function AutoEncoder_Experiment_FineTuned(Train_Images, Test_Images, Train_Labels, Test_Labels)

    for i=1:500
        Train_Images_bw{i} = rgb2gray(Train_Images{i});
        Test_Images_bw{i} = rgb2gray(Test_Images{i});
    end
    
    
    Train_Images_autoEncoder_this = Train_Images_bw;
    Test_Images_autoEncoder_this = Test_Images_bw;
    
% %     hiddensize = {10 100 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000};
    hiddensize_1 = 100;
    hiddensize_2 = 100;
    hiddensize_3 = 100;
    hiddensize_4 = 100;
    hiddensize_5 = 100; 
%     hiddensize_1 = 1536;
%     hiddensize_2 = 1024;
%     hiddensize_3 = 768;
%     hiddensize_4 = 614;
%     hiddensize_5 = 512; 
    

    % AutoEncoder 1 
    disp('AutoEncoder 1');
    AutoEncoder1 = trainAutoencoder(Train_Images_autoEncoder_this, hiddensize_1, 'UseGPU', true, 'ShowProgressWindow', true);
    AutoEncoder1_encode = encode(AutoEncoder1, Train_Images_autoEncoder_this);

    % AutoEncoder 2
    disp('AutoEncoder 2');
    AutoEncoder2 = trainAutoencoder(AutoEncoder1_encode, hiddensize_2, 'UseGPU', true, 'ShowProgressWindow', true);
    AutoEncoder2_encode = encode(AutoEncoder2, AutoEncoder1_encode);

    % AutoEncoder 3
    disp('AutoEncoder 3');
    AutoEncoder3 = trainAutoencoder(AutoEncoder2_encode, hiddensize_3, 'UseGPU', true, 'ShowProgressWindow', true);
    AutoEncoder3_encode = encode(AutoEncoder3, AutoEncoder2_encode);

    % AutoEncoder 4
    disp('AutoEncoder 4');
    AutoEncoder4 = trainAutoencoder(AutoEncoder3_encode, hiddensize_4, 'UseGPU', true, 'ShowProgressWindow', true);
    AutoEncoder4_encode = encode(AutoEncoder4, AutoEncoder3_encode);

    % AutoEncoder 5
    disp('AutoEncoder 5');
    AutoEncoder5 = trainAutoencoder(AutoEncoder4_encode, hiddensize_5, 'UseGPU', true, 'ShowProgressWindow', true);
    AutoEncoder5_encode = encode(AutoEncoder5, AutoEncoder4_encode);

    disp('softmax');
    softnet_x = trainSoftmaxLayer(double(AutoEncoder5_encode), double(Train_Labels));

%     5 stacks
    deep = stack(AutoEncoder1, AutoEncoder2, AutoEncoder3, AutoEncoder4, AutoEncoder5, softnet_x);


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




    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds_autoEncoder(Train_Images_autoEncoder_this, Train_Labels, Kfolds);

    disp('Finding Threshold');
    threshold = (0:0.000001:1);
    
    acc = zeros(length(threshold), Kfolds);
    sen = zeros(length(threshold), Kfolds);
    spe = zeros(length(threshold), Kfolds);
    
    for x=1:Kfolds
        x
        % AutoEncoder 1 
        disp('k-fold AutoEncoder 1');
        AutoEncoder1_cv = trainAutoencoder(Tr_features{x}, hiddensize_1, 'UseGPU', true, 'ShowProgressWindow', true);

        AutoEncoder1_encode_cv = encode(AutoEncoder1_cv, Tr_features{x});

        % AutoEncoder 2
        disp('k-fold AutoEncoder 2');
        AutoEncoder2_cv = trainAutoencoder(AutoEncoder1_encode_cv, hiddensize_2, 'UseGPU', true, 'ShowProgressWindow', true);
        AutoEncoder2_encode_cv = encode(AutoEncoder2_cv, AutoEncoder1_encode_cv);

        % AutoEncoder 3
        disp('k-fold AutoEncoder 3');
        AutoEncoder3_cv = trainAutoencoder(AutoEncoder2_encode_cv, hiddensize_3, 'UseGPU', true, 'ShowProgressWindow', true);
        AutoEncoder3_encode_cv = encode(AutoEncoder3_cv, AutoEncoder2_encode_cv);

        % AutoEncoder 4
        disp('k-fold AutoEncoder 4');
        AutoEncoder4_cv = trainAutoencoder(AutoEncoder3_encode_cv, hiddensize_4, 'UseGPU', true, 'ShowProgressWindow', true);
        AutoEncoder4_encode_cv = encode(AutoEncoder4_cv, AutoEncoder3_encode_cv);

        % AutoEncoder 5
        disp('k-fold AutoEncoder 5');
        AutoEncoder5_cv = trainAutoencoder(AutoEncoder4_encode_cv, hiddensize_5, 'UseGPU', true, 'ShowProgressWindow', true);
        AutoEncoder5_encode_cv = encode(AutoEncoder5_cv, AutoEncoder4_encode_cv);

        disp('k-fold softmax');
        softnet_x_cv = trainSoftmaxLayer(double(AutoEncoder5_encode_cv), double(Tr_Labels{x}));

    %     5 stacks
        deep_cv = stack(AutoEncoder1_cv, AutoEncoder2_cv, AutoEncoder3_cv, AutoEncoder4_cv, AutoEncoder5_cv, softnet_x_cv);


        xTrain_cv = zeros(64*48 ,numel(Tr_features{x}));
        for i = 1:numel(Tr_features{x})
            xTrain_cv(:,i) = Tr_features{x}{i}(:);
        end

        xTest_cv = zeros(64*48 ,numel(Te_features{x}));
        for i = 1:numel(Te_features{x})
            xTest_cv(:,i) = Te_features{x}{i}(:);
        end

        disp('k-fold deep');
        deepnet_cv = train(deep_cv, xTrain_cv, Tr_Labels{x}, 'useGPU','yes');
        disp('Time of estimated values for 500 images...');
        ft_estimated = deepnet_cv(xTest_cv);
        
        for i=1:length(threshold)
            ft_final_estimated = ft_estimated;
        
            ft_final_estimated(ft_final_estimated < threshold(i)) = 0;
            ft_final_estimated(ft_final_estimated > 0           ) = 1;

            evals = Evaluate(Te_Labels{x}(:),ft_final_estimated(:));
            
            acc(i, x) = evals(1);
%             sen(i, x) = evals(2);
%             spe(i, x) = evals(3);
            
        end
    end
    estimated = estimated_classes;
    
    final_acc = mean(acc');

    index = find(final_acc == max(final_acc), 1);

    estimated(estimated < threshold(index)) = 0;
    estimated(estimated > 0               ) = 1;
 
    evals = Evaluate(Test_Labels(:),estimated(:));
    fprintf('Threshold = %f \n\n', threshold(index));
    average = (evals(2) + evals(3))/2;
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);

end