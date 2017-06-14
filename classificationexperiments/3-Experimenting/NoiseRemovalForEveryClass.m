


function NoiseRemovalForEveryClass(Train_features, Test_features, Train_Labels, Test_Labels)
    
    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
    TableOfTrainLabels = cell(Num_of_Classes, 1);
    SoftLayer_Models = cell(Num_of_Classes, 1);
    EstimatedNoises = cell(Num_of_Classes, 1);
    
    EstimatedNoises2 = cell(Num_of_Classes, 1);
    EstimatedNoises3 = cell(Num_of_Classes, 1);
    SoftLayer_Models2 = cell(Num_of_Classes, 1);
    
    for i=1:Num_of_Classes
        i
        TableOfTrainLabels{i} = zeros(Num_of_Classes, Num_of_Images);
        TableOfTrainLabels{i}(i,:) = Train_Labels(i,:);
        
        SoftLayer_Models{i} = trainSoftmaxLayer(Train_features, TableOfTrainLabels{i}, 'ShowProgressWindow', false);
        EstimatedNoises{i} = SoftLayer_Models{i}(Train_features);
        EstimatedNoises2{i} = EstimatedNoises{i};
        EstimatedNoises{i}(i,:) = 0;
        
        EstimatedNoises2{i}(i,:) = Train_Labels(i,:);
        SoftLayer_Models2{i} = trainSoftmaxLayer(Train_features, EstimatedNoises2{i}, 'ShowProgressWindow', false);
        EstimatedNoises3{i} = SoftLayer_Models2{i}(Train_features);
    end
    
    SoftmaxTraining = trainSoftmaxLayer(Train_features, Train_Labels, 'ShowProgressWindow', false);
    Estimated = SoftmaxTraining(Test_features);
    
    for i=1:Num_of_Classes
        i
        Estimated = Estimated - EstimatedNoises{i};
    end
    
    threshold = (-2:0.0001:2);
    
    acc = zeros(length(threshold), 1);
    sen = zeros(length(threshold), 1);
    spe = zeros(length(threshold), 1);
    
    for i=1:length(threshold)
        post_estimated = Estimated;
        post_estimated(post_estimated < threshold(i)) = 0;
        post_estimated(post_estimated > 0           ) = 1;
        
        evals = Evaluate(Test_Labels(:),post_estimated(:));
        
        acc(i) = evals(1);
        sen(i) = evals(2);
        spe(i) = evals(3);
    end
    
    index = find(acc == max(acc), 1);
%     plot(threshold, acc, threshold, sen, threshold, spe);
%     index = input('Put value: ');
    estimated = Estimated;
        
    estimated(estimated < threshold(index)) = 0;
    estimated(estimated > 0               ) = 1;
  
    evals = Evaluate(Test_Labels(:),estimated(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));

end