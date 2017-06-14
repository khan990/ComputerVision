


function DecisionTreeExperimentation(Train_features, Test_features, Train_Labels, Test_Labels)
	diary('DecisionTreeExperimentation3.txt');
    Num_of_Classes = size(Train_Labels, 1);
    Num_of_Images = size(Train_Labels, 2);
	
	AlgorithmForCategorical = {'Exact' 'PullLeft' 'PCA' 'OVAbyClass'};
	AlgorithmForCategorical_string = 'AlgorithmForCategorical';
	
	MaxNumCategories = 5:5:30;
	MaxNumCategories_string = 'MaxNumCategories';
	
	MaxNumSplits = 1:5:30;
	MaxNumSplits_string = 'MaxNumSplits';
	
	ScoreTransform = {'doublelogit' 'invlogit' 'ismax' 'logit' 'identity' 'sign' 'symmetric' 'symmetriclogit' 'symmetricismax'};
	ScoreTransform_string = 'ScoreTransform';
	
	SplitCriterion = {'gdi' 'twoing' 'deviance'};
	SplitCriterion_string = 'SplitCriterion';
	
	Surrogate = {'on' 'all' 'off'};
	Surrogate_string = 'Surrogate';
	
	Prune = {'on'};
	Prune_string = 'Prune';
	
	Prior = {'uniform' 'empirical'};
	Prior_string = 'Prior';
	
	for I=1:length(AlgorithmForCategorical)
		for J=1:length(MaxNumCategories)
			for K=1:length(MaxNumSplits)
				for M=1:length(ScoreTransform)
					for N=1:length(SplitCriterion)
						for O=1:length(Surrogate)
							for P=1:length(Prune)
								for Q=1:length(Prior)

									Tree_Array = cell(Num_of_Classes, 1);
									
									for i=1:Num_of_Classes
										fprintf('Training Class Number = %d\n', i);
										Tree_Array{i} = fitctree(Train_features', Train_Labels(i,:)', ...
														AlgorithmForCategorical_string, AlgorithmForCategorical{I}, ...
														MaxNumCategories_string, MaxNumCategories(J), MaxNumSplits_string, MaxNumSplits(K), ...
														ScoreTransform_string, ScoreTransform{M}, SplitCriterion_string, SplitCriterion{N}, ...
														Surrogate_string, Surrogate{O}, Prune_string, Prune{P}, Prior_string, Prior{Q});
									end
									
									Estimated = zeros(Num_of_Classes, Num_of_Images);
									tic
									for i=1:Num_of_Classes
										Estimated(i,:) = predict(Tree_Array{i}, Test_features');
									end
									toc

									disp('------------');
                                    evals = Evaluate(Test_Labels(:),Estimated(:));
                                    average = (evals(2) + evals(3))/2;
                                    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\nAverage\t\t%f\n',evals(1), evals(2), evals(3), average);
									disp('--------------------------------');
									fprintf('%s\t\t%s\n%s\t\t%d\n%s\t\t%d\n%s\t\t%s\n%s\t\t%s\n%s\t\t%s\n%s\t\t%s\n%s\t\t%s\n', ...
														AlgorithmForCategorical_string, AlgorithmForCategorical{I}, ...
														MaxNumCategories_string, MaxNumCategories(J), MaxNumSplits_string, MaxNumSplits(K), ...
														ScoreTransform_string, ScoreTransform{M}, SplitCriterion_string, SplitCriterion{N}, ...
														Surrogate_string, Surrogate{O}, Prune_string, Prune{P}, Prior_string, Prior{Q});
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