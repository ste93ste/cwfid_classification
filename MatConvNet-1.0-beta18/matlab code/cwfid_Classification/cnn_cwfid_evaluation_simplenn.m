clear

% load the pre-trained CNN
net = cnn_cwfid();

testingDir = fullfile(vl_rootnn, 'data', 'cwfid','testing');
testingCropDir = fullfile(testingDir,'crop');
testingWeedDir = fullfile(testingDir,'weed');
testingGroundDir = fullfile(testingDir,'ground');

%only the second half of the test images are using 
%because the first half is for validation
%get testing crop images 
listing = dir(testingCropDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : (size(listing,1) - 500)
    cropImTesting(:,:,:,i)=imread(fullfile(testingCropDir,listing(i+500).name));
end

%get testing weed images 
listing = dir(testingWeedDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : (size(listing,1) - 500)
    weedImTesting(:,:,:,i)=imread(fullfile(testingWeedDir,listing(i+500).name));
end

%get testing ground images 
listing = dir(testingGroundDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : (size(listing,1) - 500)
    groundImTesting(:,:,:,i)=imread(fullfile(testingGroundDir,listing(i+500).name));
end

%put the testing image all together
data = single(reshape(cat(4,cropImTesting,weedImTesting,groundImTesting),51,51,3,[]));
%subtract the data mean 
dataMean = mean(data(:,:,:),3);
data = bsxfun(@minus, data, dataMean) ;

%remove last layer: softmax  
net.layers{end}.type = 'softmax';
net.mode = 'test';

% run the CNN
res = vl_simplenn(net, data);

% get the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;

%calculate accuracy
error = 0;
i = 1;
for j= 1:500
    if best(j) ~= i
        error= error+1;
    end
end

%accuracy Crop
accuracyCrop = 1-error/500;

error = 0;
i = 2;
for j= 501:1000
    if best(j) ~= i
        error= error+1;
    end
end

%accuracy weed
accuracyWeed = 1-error/500;

error = 0;
i = 3;
for j= 1001:1500
    if best(j) ~= i
        error= error+1;
    end
end

%accuracy ground
accuracyGround = 1-error/500;

%global accuracy of the network
accuracy = (accuracyCrop+accuracyWeed+accuracyGround)/3;