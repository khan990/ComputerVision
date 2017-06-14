


function [estimate1, estimate2, estimate3, estimate4, estimate5, estimate6, estimate7] = Ensemble(Train_features, Test_features, Train_Labels, Test_Labels)

% -----------------------------------SOFTMAX------------------------------------
    disp('SOFTMAX');
    softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);

    tic
    estimated = softnet(double(Test_features));
    toc
    
   
    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_features, Train_Labels, Kfolds);

    disp('Finding Threshold');
    threshold = (0:0.000001:1);
    
    acc = zeros(length(threshold), Kfolds);
    sen = zeros(length(threshold), Kfolds);
    spe = zeros(length(threshold), Kfolds);
    
    for x=1:Kfolds
        fprintf('Kfold = %d\n', x);
        ft_softnet = trainSoftmaxLayer(double(Tr_features{x}),Tr_Labels{x}, 'ShowProgressWindow', false);
        ft_estimated = ft_softnet(double(Te_features{x}));
        
        for i=1:length(threshold)
            ft_final_estimated = ft_estimated;
        
            ft_final_estimated(ft_final_estimated < threshold(i)) = 0;
            ft_final_estimated(ft_final_estimated > 0           ) = 1;

            evals = Evaluate(Te_Labels{x}(:),ft_final_estimated(:));
            
            acc(i, x) = evals(1);            
        end
    end
    
    final_acc = mean(acc');
    index = find(final_acc == max(final_acc), 1);
    estimated(estimated < threshold(index)) = 0;
    estimated(estimated > 0               ) = 1;
    evals = Evaluate(Test_Labels(:),estimated(:));
    disp('------------');
    fprintf('Threshold = %f \n\n', threshold(index));
    average = (evals(2) + evals(3))/2;
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
    estimate1 = estimated;
% -----------------------------------K-Nearest Neighbors------------------------------------
    disp('KNN');
%     IDX = knnsearch(Train_features',Test_features', 'Distance', 'jaccard');
%     tic
%     for i=1:length(IDX)
%         estimated_KNN(:,i) = Train_Labels(:, IDX(i));
%     end
%     toc
%     evals = Evaluate(Test_Labels(:),estimated_KNN(:));
%     average = (evals(2) + evals(3))/2;
%     fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
%     estimate2 = estimated_KNN;
        K = 5;
        IDX = knnsearch(Train_features',Test_features', 'Distance', 'Cosine', 'k', K);

        estimated_objects_of_test = zeros(size(Train_Labels, 1), size(Train_Labels, 2));
        tic
       for k=1:K
            for i=1:length(IDX)
                estimated_objects_of_test(:,i) = estimated_objects_of_test(:,i) + Train_Labels(:, IDX(i,k));
            end
       end
       toc
       if K ~= 1
           estimated_objects_of_test(estimated_objects_of_test < ((K-1)/2)) = 0;
           estimated_objects_of_test(estimated_objects_of_test >  0       ) = 1;
       end
%        result = estimated_objects_of_test;
        evals = Evaluate(Test_Labels(:), estimated_objects_of_test(:));
        
        fprintf('\nDistance = %s\n', 'Cosine');
        Print_Evaluations(evals);
        estimate2 = estimated_objects_of_test;
% -----------------------------------Decision Tree------------------------------------
    disp('Decision Tree');
    Num_of_Images = size(Train_Labels, 2);
    Num_of_Classes = size(Train_Labels, 1);
    
    for i=1:Num_of_Classes
        Dtree_model{i} = fitctree(Train_features', Train_Labels(i,:)');
    end
    tic
    for i=1:Num_of_Classes
       estimated_dtree(i,:) = predict(Dtree_model{i}, Test_features');
    end
    toc
    evals = Evaluate(Test_Labels(:),estimated_dtree(:));
    average = (evals(2) + evals(3))/2;
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
    estimate3 = estimated_dtree;
% -----------------------------------Support Vector Machine------------------------------------
    disp('Support Vector Machine');

    
    for i=1:Num_of_Classes
%         SVM_model{i} = fitcsvm(Train_features', Train_Labels(i,:)');
        SVM_model{i} = fitcsvm(Train_features', Train_Labels(i,:)', 'KernelFunction', 'polynomial', 'PolynomialOrder', 5, ...
            'Cost', 10.0, 'Verbose', 2);
    end
    tic
    for i=1:Num_of_Classes
       estimated_SVM(i,:) = predict(SVM_model{i}, Test_features');
    end
    toc
    evals = Evaluate(Test_Labels(:),estimated_SVM(:));
    average = (evals(2) + evals(3))/2;
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
    estimate4 = estimated_SVM;
% -----------------------------------Gaussian Process Regression------------------------------------
    disp('Gaussian Process Regression');
    for i=1:Num_of_Classes
        fprintf('Class Number = %d\n', i);
        model_gpr{i} = fitrgp(Train_features', Train_Labels(i, :)');
    end
    tic
    for i=1:Num_of_Classes
        estimate_gpr(i, :) = predict(model_gpr{i}, Test_features');
    end
    toc
    
    
    Kfolds = 5;
    [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(Train_features, Train_Labels, Kfolds);

    disp('Finding Threshold');
    threshold = (0:0.0001:1);
    
    acc = zeros(length(threshold), Kfolds);
    
    for x=1:Kfolds
        fprintf('Kfold = %d\n', x);
        for i=1:Num_of_Classes
            ft_model{i} = fitrgp(Tr_features{x}', Tr_Labels{x}(i, :)');
        end

        for i=1:Num_of_Classes
            ft_final_estimate(i, :) = predict(ft_model{i}, Te_features{x}');
        end

        for i=1:length(threshold)
            ft_estimated = ft_final_estimate;
        
            ft_estimated(ft_estimated < threshold(i)) = 0;
            ft_estimated(ft_estimated > 0           ) = 1;

            evals = Evaluate(Te_Labels{x}(:),ft_estimated(:));
            
            acc(i, x) = evals(1);
            
        end
    end
    
    final_acc = mean(acc');
    
    index = find(final_acc == max(final_acc), 1);
    post_estimate = estimate_gpr;
    post_estimate(post_estimate < threshold(index) ) = 0;
    post_estimate(post_estimate > 0 ) = 1;
    evals = Evaluate(Test_Labels(:),post_estimate(:));
    Print_Evaluations(evals);
    estimate5 = post_estimate;
    
% -----------------------------------Final Computation------------------------------------
    disp('Final Score');
    final_estimate = estimate1 + estimate2 + estimate3 + estimate4 + estimate5;
    estimate6 = final_estimate;
    final_estimate(final_estimate < 3) = 0;
    final_estimate(final_estimate > 1) = 1;
    evals = Evaluate(Test_Labels(:),final_estimate(:));
    estimate7 = final_estimate;
    Print_Evaluations(evals);
% -----------------------------------Neural Network------------------------------------
%     disp('Neural Network');
%      NN_on_Train_features_Train_Labels_temp = feedforwardnet(10);
%     NN_on_Train_features_Train_Labels_temp.layers{1}.transferFcn = 'satlin';
%     NN_on_Train_features_Train_Labels_temp.trainFcn = 'trainrp';
%     NN_on_Train_features_Train_Labels_temp = configure(NN_on_Train_features_Train_Labels_temp, Train_features, Train_Labels);
%     NN_on_Train_features_Train_Labels_temp = init(NN_on_Train_features_Train_Labels_temp);
% 
%     Train_features_GPU = nndata2gpu(single(Train_features));
%     Train_Labels_GPU = nndata2gpu(single(Train_Labels));
%     NN_on_Train_features_Train_Labels = train(NN_on_Train_features_Train_Labels_temp, Train_features_GPU, Train_Labels_GPU,'useGPU','yes');
%     tic
%     estimated = NN_on_Train_features_Train_Labels(single(Test_features));
%     toc
% 
%     Kfolds = 5;
%     [Tr_features, Te_features, Tr_Labels, Te_Labels] = Make_Kfolds(single(Train_features), single(Train_Labels), Kfolds);
% 
%     disp('Finding Threshold');
%     threshold = (0:0.0001:1);
% 
%     acc = zeros(length(threshold), Kfolds);
% 
%     for x=1:Kfolds
%         fprintf('Kfold = %d\n', x);
% 
%         ft_NN_on_Train_features_Train_Labels_temp = feedforwardnet(10);
%         ft_NN_on_Train_features_Train_Labels_temp.layers{1}.transferFcn = 'satlin';
%         ft_NN_on_Train_features_Train_Labels_temp.trainFcn = 'trainrp';
%         ft_NN_on_Train_features_Train_Labels_temp = configure(ft_NN_on_Train_features_Train_Labels_temp, Tr_features{x}, Tr_Labels{x});
%         ft_NN_on_Train_features_Train_Labels_temp = init(ft_NN_on_Train_features_Train_Labels_temp);
% 
%         ft_Train_features_GPU = nndata2gpu(Tr_features{x});
%         ft_Train_Labels_GPU = nndata2gpu(Tr_Labels{x});
%         ft_NN_on_Train_features_Train_Labels = train(ft_NN_on_Train_features_Train_Labels_temp, ft_Train_features_GPU, ft_Train_Labels_GPU,'useGPU','yes');
%         ft_estimated = ft_NN_on_Train_features_Train_Labels(Te_features{x});
%         for i=1:length(threshold)
%             ft_estimated_nonCorrected = ft_estimated;
%             ft_estimated_nonCorrected(ft_estimated_nonCorrected < threshold(i)) = 0;
%             ft_estimated_nonCorrected(ft_estimated_nonCorrected > 0           ) = 1;
% 
%             evals = Evaluate(Te_Labels{x}(:),ft_estimated_nonCorrected(:));
% 
%             acc(i, x) = evals(1);
%         end
%     end
%     disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
% 
%     final_acc = mean(acc');
%     index = find(final_acc == max(final_acc), 1);
%     estimated_nonCorrected = estimated;
%     estimated_nonCorrected(estimated_nonCorrected < threshold(index)) = 0;
%     estimated_nonCorrected(estimated_nonCorrected > 0               ) = 1;
%     disp('Non Corrected estimate...');
%     evals = Evaluate(Test_Labels(:),estimated_nonCorrected(:));
%     Print_Evaluations(evals);
%     estimate5 = estimated_nonCorrected;
end