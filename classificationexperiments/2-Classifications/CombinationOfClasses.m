% 
% Accuracy		0.636231
% Sensitivity		0.421965
% Specificity		0.708741
% Precision		0.328985
% Recall			0.421965
% F_Measure		0.369719
% Gmean			0.546868

% % Latest
% Accuracy		0.653615
% Sensitivity		0.235473
% Specificity		0.795120
% Precision		0.280029
% Recall			0.235473
% F_Measure		0.255825
% Gmean			0.432700

function CombinationOfClasses(Algorithm, Train_Features, Test_Features, Train_Labels, Test_Labels)
    
    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
    Combinations = 2;
    
    if (Algorithm == 'SoftMax')
        x = 1:Num_of_Classes;
        x_combinations = combnk(x, Combinations);
        
        NewLabels = zeros(size(x_combinations, 1), Num_of_Images);
        
        for i=1:Num_of_Images
            for j=1:size(x_combinations, 1)
                if ( Train_Labels(x_combinations(j, 1), i) == 1 ) && ( Train_Labels(x_combinations(j, 2), i) == 1)
                    NewLabels(j, i) = 1;
                else
                    NewLabels(j, i) = 0;
                end
            end
        end
        
        softnet = trainSoftmaxLayer(Train_Features,Train_Labels, 'ShowProgressWindow', false);
        
        estimated = softnet(double(Test_Features));
        estimated( estimated < 0.0001)=0;
        estimated(estimated > 0)=1;
        
        final_estimated = zeros(Num_of_Classes, Num_of_Images);
        
        for i=1:Num_of_Images
            for j=1:Num_of_Classes
                firstColumn = find(x_combinations(:,1) == j);
                secondColumn = find(x_combinations(:,2) == j);
                
                count1s = 0;
                count0s = 0;


                for m=1:size(firstColumn)
                    if ( estimated(m, i) == 1 ) 
%                         final_estimated(j, i) = 1;
                        count1s = count1s + 1;
                    else
%                         final_estimated(j, i) = 0;
                        count0s = count0s + 1;
                    end
                end
                
                for n=1:size(secondColumn)
                    if ( estimated(n, i) == 1 ) 
%                         final_estimated(j, i) = 1;
                        count1s = count1s + 1;
                    else
%                         final_estimated(j, i) = 0;
                        count0s = count0s + 1;
                    end
                end
                
                if( count1s >= count0s)
                    final_estimated(j, i) = 1;
                else
                    final_estimated(j, i) = 0;
                end
            end
        end
        
        evals = Evaluate(Test_Labels(:),final_estimated(:));
        fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nPrecision\t\t%f\nRecall\t\t\t%f\nF_Measure\t\t%f\nGmean\t\t\t%f\n',evals(1), evals(2), evals(3), evals(4), evals(5), evals(6), evals(7));
    end
end
