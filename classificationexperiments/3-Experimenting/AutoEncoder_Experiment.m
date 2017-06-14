


function AutoEncoder_Experiment(Train_Images, Test_Images, Train_Labels, Test_Labels)

    
    for i=1:500
        Train_Images_bw{i} = rgb2gray(Train_Images{i});
        Test_Images_bw{i} = rgb2gray(Test_Images{i});
    end
    Train_Images_autoEncoder_this = Train_Images_bw;
    Test_Images_autoEncoder_this = Test_Images_bw;
    

    hiddensize_1 = 1536;
    hiddensize_2 = 1024;
    hiddensize_3 = 768;
    hiddensize_4 = 614;
    hiddensize_5 = 512;    
%     hiddensize_1 = 1000;
%     hiddensize_2 = 1000;
%     hiddensize_3 = 1000;
%     hiddensize_4 = 1000;
%     hiddensize_5 = 1000; 
    
    % AutoEncoder 1 
    disp('AutoEncoder 1');
    AutoEncoder1 = trainAutoencoder(Train_Images_autoEncoder_this, hiddensize_1, 'UseGPU', true);

    AutoEncoder1_encode = encode(AutoEncoder1, Train_Images_autoEncoder_this);
    
    % AutoEncoder 2
    disp('AutoEncoder 2');
    AutoEncoder2 = trainAutoencoder(AutoEncoder1_encode, hiddensize_2, 'UseGPU', true);
    AutoEncoder2_encode = encode(AutoEncoder2, AutoEncoder1_encode);
    
    % AutoEncoder 3
    disp('AutoEncoder 3');
    AutoEncoder3 = trainAutoencoder(AutoEncoder2_encode, hiddensize_3, 'UseGPU', true);
    AutoEncoder3_encode = encode(AutoEncoder3, AutoEncoder2_encode);
    
    % AutoEncoder 4
    disp('AutoEncoder 4');
    AutoEncoder4 = trainAutoencoder(AutoEncoder3_encode, hiddensize_4, 'UseGPU', true);
    AutoEncoder4_encode = encode(AutoEncoder4, AutoEncoder3_encode);
    
    % AutoEncoder 5
    disp('AutoEncoder 5');
    AutoEncoder5 = trainAutoencoder(AutoEncoder4_encode, hiddensize_5, 'UseGPU', true);
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
    
    disp('deep net training');
    deepnet = train(deep, xTrain, Train_Labels, 'useGPU','yes');
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
    
end