% GoogleNet
% Extract features from modified GoogleNet mat file.
% Modified = last layer removed.

% Example
% [Train_features_f_check, Test_features_f_check] = GoogleNet_Features('googlenetMod.mat', Trainall, Testall);

function [Training_Features, Test_Features] = GoogleNet_Features (GoogleNetPath, Training_images, Test_images)
% GoogleNet_Features  Extract features using GoogleNetmodified
%   [Training_Features, Test_Features] = GoogleNet_Features (GoogleNetPath, Training_Labels, Test_Labels)
%   

    loadNet = load(GoogleNetPath); 
    net = dagnn.DagNN.loadobj(loadNet.googlenetMod);

    Training_Features = zeros(1024, length(Training_images));
    Test_Features = zeros(1024, length(Test_images));

    for i=1:length(Training_images)
        fprintf('Training %d...\n', i);
        Training_Features(:,i)=CNN_features(Training_images{i} ,net);
    end

    for i=1:length(Test_images)
        fprintf('Testing %d...\n', i);
        Test_Features(:,i)=CNN_features(Test_images{i},net);
    end
end