

function [Tr_features_ret, Te_features, Tr_Labels_ret, Te_Labels_ret] = Make_Kfolds(features, Labels, Num_of_folds)

    Num_of_Classes = size(Labels, 1);
    Num_of_Images = size(Labels, 2);
    
    remainder = mod(Num_of_Images, Num_of_folds);
    
    if (remainder ~= 0)
        error('Irregular number of k-cross validations...');
    end
    
    chunks = Num_of_Images / Num_of_folds;
    
    Tr_features_ret = cell(Num_of_folds, 1);
    Tr_Labels_ret = cell(Num_of_folds, 1);
    
    Te_features = cell(Num_of_folds, 1);
    Te_Labels_ret = cell(Num_of_folds, 1);
    
    indices = (1:Num_of_Images);
    
    
    for i=1:Num_of_folds
        Tr_features_ret{i} = zeros(size(features, 1), chunks);
        Tr_Labels_ret{i} = zeros(Num_of_Classes, chunks);
        
        Te_features{i} = zeros(size(features, 1), chunks);
        Te_Labels_ret{i} = zeros(Num_of_Classes, chunks);
        
        Te_indices = ( (chunks * (i-1)) + 1 : (chunks * (i-1)) + chunks);
        Tr_indices = indices;
        Tr_indices(Te_indices) = [];
        
        Tr_features_ret{i} = features(:, Tr_indices);
        Tr_Labels_ret{i} = Labels(:, Tr_indices);
        
        Te_features{i} = features(:, Te_indices);
        Te_Labels_ret{i} = Labels(:, Te_indices);
        
%         fprintf('Training...\n');
%         fprintf('Start index = %d\t\t\tEnd Index = %d\n', Tr_indices(1), Tr_indices(end));
%         
%         fprintf('Testing...\n');
%         fprintf('Start index = %d\t\t\tEnd Index = %d\n', Te_indices(1), Te_indices(end));
    end
end