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

se = strel('disk', 8);  % Shape for the erosion of the center pixels for the patches.

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
    countG = 0;
    countW = 0;
    countC = 0;
    IMask1 = im2bw(AnnotationIm(:,:,1,i));    % Binary mask (weed)
    IMask2 = im2bw(AnnotationIm(:,:,2,i));    % Binary mask (crop)
    IMask3 = ~(im2bw(AnnotationIm(:,:,1,i)) | im2bw(AnnotationIm(:,:,2,i))); % Binary mask (ground)
    Centers1 = IMask1(ceil(dim/2):end-floor(dim/2),...
                          ceil(dim/2):end-floor(dim/2));
    Centers1 = imerode(Centers1,se);  % Perform erosion      
    [y1, x1] = find(Centers1);  % x,y: coordinate arrays for nonzero elts.
    Centers2 = IMask2(ceil(dim/2):end-floor(dim/2),...
                          ceil(dim/2):end-floor(dim/2));
    [y2, x2] = find(Centers2);  % x,y: coordinate arrays for nonzero elts.                  
    Centers2 = imerode(Centers2,se);  % Perform erosion                 
    Centers3 = IMask3(ceil(dim/2):end-floor(dim/2),...
                          ceil(dim/2):end-floor(dim/2));
    Centers3 = imerode(Centers3,se);  % Perform erosion 
    [y3, x3] = find(Centers3);  % x,y: coordinate arrays for nonzero elts.
   
      for j=1 : stride : (size(slidingWindowImTesting,2)-dim)
       for k=1 : stride : (size(slidingWindowImTesting,1)-dim)
           if  find(ismember(find(x1(:)==j),find(y1(:)==k)),1)
               label(cont1)= weed;
               countW= countW+1;
               im(:,:,:,cont1) = slidingWindowImTesting(k:k+dim-1, ...
                   j:j+dim-1,:,i);
               cont1=cont1+1;
               
           end
   
           if find(ismember(find(x2(:)==j),find(y2(:)==k)),1)
               label(cont1)= crop;
               countC= countC+1;
                 im(:,:,:,cont1) = slidingWindowImTesting(k:k+dim-1, ...
                   j:j+dim-1,:,i);
               cont1=cont1+1;
               
           end
   
            if find(ismember(find(x3(:)==j),find(y3(:)==k)),1)
          
               label(cont1)= ground;
               countG= countG+1;
               im(:,:,:,cont1) = slidingWindowImTesting(k:k+dim-1, ...
                   j:j+dim-1,:,i);
              cont1 = cont1+1;
             
            end
       end
      end
      
          
   %evaluates the patches    
   res = vl_simplenn(net, single(im));
   scores = squeeze(gather(res(end).x)) ;
   [bestScore, best] = max(scores) ;
   
     %calculate confusion matrix
   C = confusionmat(label,best);
   Accuracy(i) = (C(1,1)+C(2,2)+C(3,3))/(C(1,1)+C(1,2)+C(1,3)+C(2,1)+C(2,2)...
       +C(2,3)+C(3,1)+C(3,2)+C(3,3));
end
accuracyTot=0;
for i=1: size(Accuracy)
accuracyTot = accuracyTot+Accuracy(i)
end
accuracyTot = accuracyTot/i;