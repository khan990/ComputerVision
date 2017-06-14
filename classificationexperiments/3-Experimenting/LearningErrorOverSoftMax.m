




function LearningErrorOverSoftMax(Train_features, Test_features, Train_Labels, Test_Labels)

    Num_of_Images = size(Train_Labels, 2);
    Num_of_Classes = size(Train_Labels, 1);

    x_softmax = trainSoftmaxLayer(Train_features, Train_Labels, 'ShowProgressWindow', false);
    x_estimate = x_softmax(Train_features);

    x_estimate_2 = x_estimate;
    x_estimate_2(x_estimate_2 < 0.001) = 0;
    x_estimate_2(x_estimate_2 > 0) = 1;

    x_Train_CNN_features = [Train_features ; Train_Labels];
    x_Learn_Train_Error = trainSoftmaxLayer(x_Train_CNN_features, x_estimate_2, 'ShowProgressWindow', false);

    x_Dummy_Test_Labels = zeros(Num_of_Classes, Num_of_Images);
    x_Dummy_Test_Labels(x_Dummy_Test_Labels == 0) = 0.5;

    x_Test_CNN_features_Dummy = [Test_features ; x_Dummy_Test_Labels];

    x_Learnt_Test_Error = x_Learn_Train_Error(x_Test_CNN_features_Dummy);

    x_Learnt_Test_Error_2 = x_Learnt_Test_Error;
    x_Learnt_Test_Error_2(x_Learnt_Test_Error_2 < 0.001) = 0;
    x_Learnt_Test_Error_2(x_Learnt_Test_Error_2 > 0) = 1;

    estimate_test = x_softmax(Test_features);

    estimate_test(estimate_test < 0.0001) = 0;
    estimate_test(estimate_test > 0     ) = 1;

    Cleaned_Test_Labels = estimate_test + x_Learnt_Test_Error_2;
    Cleaned_Test_Labels( Cleaned_Test_Labels == 2) = 1;

    evals = Evaluate(Test_Labels(:),Cleaned_Test_Labels(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

end