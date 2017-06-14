


% upperbound = 0.0001;
% lowerbound = 0;

function [evals, estimated] = SoftMax(Train_features, Test_features, Train_Labels, Test_Labels, upperbound, lowerbound)

    softnet = trainSoftmaxLayer(double(Train_features),Train_Labels, 'ShowProgressWindow', false);

    estimated = softnet(double(Test_features));

    range = (0.001:0.00001:0.009);
    
    j=1;
    for i=range
        post_estimate = estimated;
        post_estimate( post_estimate < i ) = 0;
        post_estimate(post_estimate > 0) = 1;
        
        evals = Evaluate(Test_Labels(:),post_estimate(:));
        
        acc(j) = evals(1);
        sen(j) = evals(2);
        spe(j) = evals(3);
        j = j + 1;
    end
    estimated( estimated < upperbound)=0;

    estimated(estimated > lowerbound)=1;

    evals = Evaluate(Test_Labels(:),estimated(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    
    plot(range, acc, range, sen, range, spe);
    hold on;
    legend('Accuracy', 'Sensitivity', 'Specificity');
    hold off;
end
% SoftMax(Train_CNN_features, Test_CNN_features, Train_Labels, Test_Labels, 0.0001, 0)
% Accuracy		0.812615
% Sensitivity		0.814725
% Specificity		0.811902
% Precision		0.594451
% Recall			0.814725
% F_Measure		0.687372
% Gmean			0.813312