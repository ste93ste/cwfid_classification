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

%percentage of pixel for class based on training set. Calculate in the
%classUmbalancingThreshold.m script
groundPercentage = 0.925;
cropPercentage = 0.0154;
weedPercentage = 0.0596;


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
           im(:,:,:,cont1) = (imcrop(slidingWindowImTesting(:,:,:,i),...
               [j,k,dim-1,dim-1]));
           if AnnotationIm(int32(k+dim/2-1),int32(j+dim/2-1),1,i) == 255
               label(cont1)= weed;
           else if AnnotationIm(int32(k+dim/2-1),int32(j+dim/2-1),2,i) == 255
                    label(cont1)= crop;
                else 
                label(cont1)= ground;
               end
           end
           cont1 = cont1+1;
       end
    end
   
   %put the testing image all together
    im = single(reshape(cat(4,im),51,51,3,[]));
   
   %evaluates the patches    
   res = vl_simplenn(net,im);
   scores = squeeze(gather(res(end).x)) ;
   
   %calculate ground threshold
   groundScores = sort(scores(3,:),'descend');
   thresholdGround = groundScores(int32(cont1*groundPercentage));
   
   %calculate crop threshold
   cropScores = sort(scores(1,:),'descend');
   thresholdCrop = cropScores(int32(cont1*cropPercentage));
 
   for j=1 : size(scores,2)
      if scores(3,j)> thresholdGround
          best(j)= ground;
      else if scores(1,j) > thresholdCrop
              best(j) = crop;
          else
              best(j) = weed;
          end
      end
   end
   

 %calculate confusion matrix
   C = confusionmat(label,best);
   Accuracy(i) = (C(1,1)+C(2,2)+C(3,3))/(C(1,1)+C(1,2)+C(1,3)+C(2,1)+C(2,2)...
       +C(2,3)+C(3,1)+C(3,2)+C(3,3));
   
   I = reshape(best,int16((size(slidingWindowImTesting,1)-dim)/stride),...
        int16((size(slidingWindowImTesting,2)-dim)/stride));
  
   
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
   subplot(1,3,1)
   imshow(slidingWindowImTesting(:,:,:,i));
   title('original')
   subplot(1,3,2)
   imshow(AnnotationIm(:,:,:,i));
   title('annotation')
   subplot(1,3,3)
   imshow(Im)
   title('best')
   
 end

accuracyTot=0;
for i=1: size(Accuracy)
accuracyTot = accuracyTot+Accuracy(i)
end
accuracyTot = accuracyTot/i;
