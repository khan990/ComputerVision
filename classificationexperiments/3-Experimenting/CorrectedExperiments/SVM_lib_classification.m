

function SVM_lib_classification(Train_features, Test_features, Train_Labels, Test_Labels)

% -s 0 -t 0 -d 0 -g 0 -r 0 -c 10 -n 10 -e 5.000000e-04 
% -s 0 -t 1 -d 0 -g 0 -r 5 -c 5 -n 5 -e 3.000000e-04 
% -s 0 -t 1 -d 5 -g 0 -r 0 -c 10 -n 10 -e 4.000000e-04
% -s 0 -t 1 -d 5 -g 0 -r 10 -c 10 -n 10 -e 4.000000e-04 

    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);

    CommandString = sprintf('%s ', '-s 0 -t 1 -d 0 -g 0 -r 5 -c 5 -n 5 -e 3.000000e-04 ');
    for i=1:Num_of_Classes
        model{i} = svmtrain(double(Train_Labels(i,:)'), double(Train_features'), CommandString);
    end

    estimated = zeros(Num_of_Classes, Num_of_Images);
    tic
    for i=1:Num_of_Classes
        estimated(i,:) = svmpredict(double(Test_features(i,:)'), double(Test_features'), model{i});
    end
    toc

    evals = Evaluate(Test_Labels(:),estimated(:));
%     fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    disp('------------');
    average = (evals(2) + evals(3))/2;
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
    
end