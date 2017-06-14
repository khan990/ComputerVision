


function SoftMax_x(Train_features, Test_features, Train_Labels, Test_Labels)

%     softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);
    softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false, 'LossFunction', 'mse');
%     softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);

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
            sen(i, x) = evals(2);
            spe(i, x) = evals(3);
            
        end
    end
    
    final_acc = mean(acc');
%     final_sen = mean(sen');
%     final_spe = mean(spe');
    
%     plot(threshold, final_acc, threshold, final_sen, threshold, final_spe);
%     hold on;
%     legend('Accuracy', 'Sensitivity', 'Specificity');
%     hold off;
%     pause;
    index = find(final_acc == max(final_acc), 1);
%     index = input('Put value: ');
%     estimated = final_estimate;
        
    estimated(estimated < threshold(index)) = 0;
%     estimated(estimated < 0.000001) = 0;
    estimated(estimated > 0               ) = 1;
 
    evals = Evaluate(Test_Labels(:),estimated(:));
%     Print_Evaluations(evals);
    disp('------------');
    fprintf('Threshold = %f \n\n', threshold(index));
    average = (evals(2) + evals(3))/2;
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
    
end