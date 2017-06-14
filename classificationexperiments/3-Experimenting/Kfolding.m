function [Training, Testing] = Kfolding(Matrix, NumOfFolds)

    NumOfColums = size(Matrix, 2)/NumOfFolds;
    
    Training = zeros(size(Matrix, 1), NumOfColums, NumOfFolds-1);
    NumOfFolds
    
    for i=1:NumOfFolds
        if i < NumOfFolds
            if i == 1
                fprintf('i = %d, index = %d to %d \n', i, 1, (i*NumOfColums));
                Training(:,:,i) = Matrix(:, (1):(i*NumOfColums));
            else
                fprintf('i = %d, index = %d to %d \n', i, ((i-1)*NumOfColums) + 1, ((i-1)*NumOfColums) + NumOfColums);
                Training(:,:,i) = Matrix(:, ((i-1)*NumOfColums) + 1:((i-1)*NumOfColums) + NumOfColums);
            end
        else
            fprintf('i = %d, index = %d to %d \n', i, ((i-1)*NumOfColums) + 1, ((i-1)*NumOfColums) + NumOfColums);
            Testing = Matrix(:, ((i-1)*NumOfColums) + 1 :((i-1)*NumOfColums) + NumOfColums);
        end
    end

end