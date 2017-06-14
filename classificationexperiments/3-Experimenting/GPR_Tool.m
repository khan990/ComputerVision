



function GPR_Tool(Train_features, Test_features, Train_Labels, Test_Labels)
    
    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
    for i=1:Num_of_Classes
        i
        meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
        covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
        likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
        covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);


        hyp2 = minimize(hyp2, @gp, -750, @infExact, [], covfunc, likfunc,  Train_features', Train_Labels(i,:)');

        [m(i,:) s2] = gp(hyp2, @infExact, [], covfunc, likfunc, Train_features', Train_Labels(i,:)', Test_features');

    end
    
    vars = (0:0.01:1);
    
    for i=1:length(vars)
        post_m = m;
        post_m(post_m < vars(i)) = 0;
        post_m(post_m > 0  ) = 1;
        evals = Evaluate(Test_Labels(:),post_m(:));
        acc(i) = evals(1);
        sen(i) = evals(2);
        spe(i) = evals(3);
    end
    
    figure;
    hold on;
    title('Accuracy');
    plot(vars, acc);
    hold off;
    
    figure;
    hold on;
    title('Sensitivity');
    plot(vars, sen);
    hold off;
    
    figure;
    hold on;
    title('Specificity');
    plot(vars, spe);
    hold off;
    
    m(m < 0.98) = 0;
    m(m > 0  ) = 1;
    evals = Evaluate(Test_Labels(:),m(:));
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));


end
% 
% Accuracy		0.737440
% Sensitivity		0.000000
% Specificity		1.000000
% Precision		NaN
% Recall			0.000000
% F_Measure		NaN
% Gmean			0.000000