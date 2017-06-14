


function SoftMax_OTSU_x(Train_features, Test_features, Train_Labels, Test_Labels)

    softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);
%     softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false, 'LossFunction', 'mse');
%     softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);

    tic
    estimated = softnet(double(Test_features));
    toc
    

    level = graythresh(estimated);
    estimated = im2bw(estimated, level);

    evals = Evaluate(Test_Labels(:),estimated(:));

    average = (evals(2) + evals(3))/2;
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
    
end