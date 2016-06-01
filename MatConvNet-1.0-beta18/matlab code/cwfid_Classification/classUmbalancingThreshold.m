trainingDir = fullfile(vl_rootnn, 'data', 'cwfid','training');
annotationDir = fullfile(trainingDir,'annotation');

listing = dir(annotationDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
%get the annotation image
for i=1 : size(listing,1)
    AnnotationIm(:,:,:,i)=imread(fullfile(annotationDir,listing(i).name));
end

ground = 0;
weed = 0;
crop = 0;
tot = 0;

for i=1 : size(AnnotationIm,4)
    for j=1 : size(AnnotationIm,1)
        for k=1 :size(AnnotationIm,2)
            tot = tot+1;
            if AnnotationIm(j,k,1,i) == 255
                weed = weed+1;
            else if AnnotationIm(j,k,2,i) == 255
                    crop = crop +1;
                else
                    ground = ground+1;
                end
            end
        end
    end
end

groundProb = ground/tot;
weedProb = weed/tot;
cropProb = crop/tot;

