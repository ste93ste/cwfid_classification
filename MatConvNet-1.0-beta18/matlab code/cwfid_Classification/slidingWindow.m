close all
clear
clc

%define the size of the patches
dim = 51;
%define the stride of the sliding window
stride = 10;
%crop = 1, weed = 2, ground = 3
crop = 1;
weed = 2;
ground = 3;

% load the pre-trained CNN
net = cnn_cwfid();
%remove last layer: softmax  
net.layers{end}.type = 'softmax';
net.mode = 'test';

testingDir = fullfile(vl_rootnn, 'data', 'cwfid','testing');
testingSlidingWindowDir = fullfile(testingDir,'sliding window');
annotationDir = fullfile(testingDir,'annotation');

listing = dir(testingSlidingWindowDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
%get the testing image
for i=1 : size(listing,1)
    slidingWindowImTesting(:,:,:,i)=imread(fullfile(testingSlidingWindowDir,listing(i).name));
end

listing = dir(annotationDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
%get the annotation image
for i=1 : size(listing,1)
    AnnotationIm(:,:,:,i)=imread(fullfile(annotationDir,listing(i).name));
end

%for all the images
for i=1 : size(slidingWindowImTesting,4)
    cont1 = 1;
    for j=1 : stride : (size(slidingWindowImTesting,2)-dim)
       for k=1 : stride : (size(slidingWindowImTesting,1)-dim)
           %get the patches
           im(:,:,:,cont1) = single(imcrop(slidingWindowImTesting(:,:,:,i),...
               [j,k,dim-1,dim-1]));
           cont1 = cont1+1;
       end
    end
   %evaluates the patches    
   res = vl_simplenn(net, im);
   scores = squeeze(gather(res(end).x)) ;
   [bestScore, best] = max(scores) ;

   I = reshape(best,int16((size(slidingWindowImTesting,1)-dim)/stride),...
        int16((size(slidingWindowImTesting,2)-dim)/stride));
   Icrop = reshape(scores(crop,:),int16((size(slidingWindowImTesting,1)-dim)/stride),...
        int16((size(slidingWindowImTesting,2)-dim)/stride));
   Icrop = imresize(Icrop,[size(slidingWindowImTesting,1),size(slidingWindowImTesting,2)]);
   Iweed = reshape(scores(weed,:),int16((size(slidingWindowImTesting,1)-dim)/stride),...
        int16((size(slidingWindowImTesting,2)-dim)/stride));
   Iweed = imresize(Iweed,[size(slidingWindowImTesting,1),size(slidingWindowImTesting,2)]);
   Iground = reshape(scores(ground,:),int16((size(slidingWindowImTesting,1)-dim)/stride),...
        int16((size(slidingWindowImTesting,2)-dim)/stride));
   Iground = imresize(Iground,[size(slidingWindowImTesting,1),size(slidingWindowImTesting,2)]); 
   
   %create an image with the best class with color: crop=green weed=red
   %ground=black
   for s=1 : size(I,1)
        for v=1 : size(I,2)
            if I(s,v) == crop
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,1) = 0;
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,2) = 255;
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,3) = 0;
            end
            if I(s,v) == weed
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,1) = 255;
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,2) = 0; 
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,3) = 0;
            end
            if I(s,v) == ground
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,1) = 0;
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,2) = 0;
               Im((s-1)*stride+1:s*stride,(v-1)*stride+1:v*stride,3) = 0;
            end
        end
   end
   
   figure
   subplot(3,2,1)
   imshow(slidingWindowImTesting(:,:,:,i));
   title('original')
   subplot(3,2,2)
   imshow(AnnotationIm(:,:,:,i));
   title('annotation')
   subplot(3,2,3)
   imshow(Im)
   title('best')
   subplot(3,2,4)
   imagesc(Icrop),colorbar
   title('crop')
   subplot(3,2,5)
   imagesc(Iweed),colorbar
   title('weed')
   subplot(3,2,6)
   imagesc(Iground),colorbar
   title('ground')
   
   %get the label from annotation and the predicted class
   cont = 0;
   for s = 1 : size(Im,1)
       for v = 1 : size(Im,2)
           cont= cont+1;
           if AnnotationIm(s,v,1,i) == 255
               label(cont)= weed;
           else if AnnotationIm(s,v,2,i) == 255
                    label(cont)= crop;
                else 
                label(cont)= ground;
               end
           end
           if Im(s,v,1) == 255
               predicted(cont)= weed;
           else if Im(s,v,2) == 255
                  predicted(cont)= crop;
                else
                predicted(cont)= ground;
               end
           end
       end
   end
   
   %calculate confusion matrix
   C = confusionmat(label,predicted);
   Accuracy = (C(1,1)+C(2,2)+C(3,3))/(C(1,1)+C(1,2)+C(1,3)+C(2,1)+C(2,2)...
       +C(2,3)+C(3,1)+C(3,2)+C(3,3));
   
end