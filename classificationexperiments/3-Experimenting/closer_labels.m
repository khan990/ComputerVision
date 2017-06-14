

function label_knn = closer_labels(Train_Labels_New)
    Num_of_classes = 25;

    someclasses = [1:25];

    combination_of_classes = combnk(someclasses, 2);

    for i=1:length(combination_of_classes)
        label_knn{i} = fitcknn(Train_Labels_New(combination_of_classes(i,1), :)', Train_Labels_New(combination_of_classes(i, 2), :));
    end

end