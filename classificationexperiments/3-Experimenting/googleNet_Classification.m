


function pics = google_Net_Classification(Train_features, Train_Images, ClassNames)

    for i=1:length(Train_Images)
        selected_Classes = find(Train_features(4,:) > max(Train_features(4,:)/4));
        
        pic = Train_Images{i};
        pic_edited = padarray(pic, 360, 1, 'pre');
    
        for j=1:length(selected_Classes)
            
            if j < 10
                pic_edited = insertText(pic_edited, [40, j*40], ClassNames(selected_Classes(j)) , 'FontSize', 20);
            end
            if j > 9 && j < 19
                pic_edited = insertText(pic_edited, [40+500, (j-9)*40], ClassNames(selected_Classes(j)) , 'FontSize', 20);
            end
            if j > 18 && j < 28
                pic_edited = insertText(pic_edited, [40+500+500, (j-18)*40], ClassNames(selected_Classes(j)) , 'FontSize', 20);
            end
            if j > 27 && j < 37
                pic_edited = insertText(pic_edited, [40+500+500+500, (j-27)*40], ClassNames(selected_Classes(j)) , 'FontSize', 20);
            end
            
        end
        
        pics{i} = pic_edited;
    end

end