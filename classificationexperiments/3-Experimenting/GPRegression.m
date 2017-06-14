


function GPRegression(Train_feature, Test_feature, Train_Label, Test_Label)

    Num_of_Classes = size(Train_Label, 1);
    Num_of_Images = size(Train_feature, 2);
    
    
    for i=1:Num_of_Classes
        fprintf('Class Number = %d\n', i);
        model{i} = fitrgp(Train_feature', Train_Label(i, :)');
    end
    tic
    for i=1:Num_of_Classes
        estimate(i, :) = predict(model{i}, Test_feature');
    end
    toc
    acc = zeros(size(0.1:0.1:0.9));
    sen = zeros(size(0.1:0.1:0.9));
    spe = zeros(size(0.1:0.1:0.9));
    j=1;
    for i=0.1:0.01:0.9
        post_estimate = estimate;
        post_estimate(post_estimate < i ) = 0;
        post_estimate(post_estimate > 0 ) = 1;
        evals = Evaluate(Test_Label(:),post_estimate(:));
        acc(j) = evals(1);
        sen(j) = evals(2);
        spe(j) = evals(3);
        j = j + 1;
    end
    post_estimate = estimate;
    post_estimate(post_estimate < 0.25 ) = 0;
    post_estimate(post_estimate > 0 ) = 1;
    evals = Evaluate(Test_Label(:),post_estimate(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    plot(0.1:0.01:0.9, acc, 0.1:0.01:0.9, sen, 0.1:0.01:0.9, spe);
    hold on;
    legend( 'Accuracy', 'Sensitivity', 'Specificity');
    hold off;
end