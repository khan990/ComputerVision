


function [Features] = Full_GoogleNet(Images)
    net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) 
    disp('Feature Extraction starting....');
%     tic
    for i=1:length(Images)
        fprintf('feature extraction for image = %d\n', i);
        tic
        %i
%         im = imread(Images{i}) ;
        im_ = single(Images{i}) ; % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = im_ - net.meta.normalization.averageImage ;

        % run the CNN
        net.eval({'data', im_}) ;

        % obtain the CNN otuput
        Features(i,:) = net.vars(net.getVarIndex('prob')).value ;
        Features(i,:) = squeeze(gather(Features(i,:))) ;
        toc
        
        fprintf('\n\n');
    end
%     disp('Final time:');
%     toc
end