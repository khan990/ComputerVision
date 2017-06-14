



function Print_Evaluations(Evals)
    disp('------------------------------');
    fprintf('Accuracy\t\t%f\nSensitivity\t\t%f\nSpecificity\t\t%f\n', Evals(1), Evals(2), Evals(3));
    average = (Evals(2) + Evals(3))/2;
    fprintf('Average\t\t%f\n', average);
    disp('------------------------------');
end