


function SVMLIB_Experiment(Train_features, Test_features, Train_Labels, Test_Labels)
    diary('SVMLIB_Experiment.txt');

    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    
%     Brute FORCE libsvm_options
    svm_type = {'svm_type'};
    svm_type_options = {0 1 2 3 4};
    
    kernel_type = {'kernel_type'};
    kernel_type_options = {0 1 2 3 4};
    
    degree = {'degree'};
    degree_options = (1:1:25);
    
    gamma = {'gamma'};
    gamma_options = (1:1:25);
    
    coef0 = {'coef0'};
    coef0_options = (0:1:10);
    
    cost = {'cost'};
    cost_option = (1:1:10);
    
    nu = {'nu'};
    nu_options = (0:0.1:1);
    
    epsilon = {'epsilon'};
    epsilon_options = (0.0001);
    
    



    for A=1:length(svm_type_options)
        for B=1:length(kernel_type_options)
            for C=1:length(degree_options)
                for D=1:length(gamma_options)
                    for E=1:length(coef0_options)
                        for F=1:length(cost_option)
                            for G=1:length(nu_options)
                                for H=1:length(epsilon_options)
                                    CommandString = '';
                                    disp('-------------------------------');
%                                     fprintf('svm_type_options = %d\n', A);
%                                     fprintf('kernel_type_options = %d\n', B);
%                                     fprintf('degree_options = %d\n', C);
%                                     fprintf('gamma_options = %d\n', D);
%                                     fprintf('coef0_options = %d\n', E);
%                                     fprintf('cost_option = %d\n', F);
%                                     fprintf('nu_options = %f\n', G);
%                                     fprintf('epsilon_options = %f\n', H);
                                    
                                    CommandString = sprintf('%s-s %d ', CommandString, svm_type_options{A});
                                    CommandString = sprintf('%s-t %d ', CommandString, kernel_type_options{B});
                                    CommandString = sprintf('%s-d %d ', CommandString, degree_options(C));
                                    CommandString = sprintf('%s-g %d ', CommandString, gamma_options(D));
                                    CommandString = sprintf('%s-r %d ', CommandString, coef0_options(E));
                                    CommandString = sprintf('%s-c %d ', CommandString, cost_option(F));
                                    CommandString = sprintf('%s-n %d ', CommandString, cost_option(F));
                                    CommandString = sprintf('%s-e %d ', CommandString, epsilon_options(H));
                                    fprintf('%s\n', CommandString );
                                    for i=1:Num_of_Classes
                                        model{i} = svmtrain(double(Train_Labels(i,:)'), double(Train_features'), CommandString);

                                    end

                                    for i=1:Num_of_Classes
                                        fprintf('Training labels for this classId %d are %d\n', i, sum(Train_Labels(i,:)));
                                        estimated(i,:) = svmpredict(double(Test_features(i,:)'), double(Test_features'), model{i});
                                    end

                                    fprintf('\n\n');
                                    evals = Evaluate(Test_Labels(:),estimated(:));
                                    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    diary off;
end

% function SVMLIB_Experiment(Train_features, Test_features, Train_Labels, Test_Labels)
% 
%     Num_of_Classes = size(Train_Labels, 1);
%     Num_of_Images = size(Train_Labels, 2);
%     
%     
%     for i=1:Num_of_Classes
%         model{i} = svmtrain(double(Train_Labels(i,:)'), double(Train_features'));
%         
%     end
%     
%     for i=1:Num_of_Classes
%         fprintf('Training labels for this classId %d are %d\n', i, sum(Train_Labels(i,:)));
%         estimated(i,:) = svmpredict(double(Test_features(i,:)'), double(Test_features'), model{i});
%     end
% 
%     disp('-----------------------------------');
%     evals = Evaluate(Test_Labels(:),estimated(:));
%     fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
% 
% end