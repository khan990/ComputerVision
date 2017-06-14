function [features]=CNN_features(im,net)


im = single(im);
im_ = imresize(im, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;
% run the CNN
net.eval({'data', im_}) ;
features(:) = reshape(net.vars(length(net.vars)).value, 1,[]);

