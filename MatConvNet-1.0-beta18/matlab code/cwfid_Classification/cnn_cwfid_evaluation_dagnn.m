clear

% load the pre-trained CNN
net = cnn_cwfid();

testingDir = fullfile(vl_rootnn, 'data', 'cwfid','testing');
testingCropDir = fullfile(testingDir,'crop');
testingWeedDir = fullfile(testingDir,'weed');
testingGroundDir = fullfile(testingDir,'ground');

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

data = single(reshape(cat(4,cropImTesting,weedImTesting,groundImTesting),51,51,3,[]));
%put the testing image all together
data = single(reshape(cat(4,cropImTesting,weedImTesting,groundImTesting),51,51,3,[]));
%subtract the data mean 
dataMean = mean(data(:,:,:),3);
data = bsxfun(@minus, data, dataMean) ;

% run the CNN
net.eval({'input',data}) ;

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prediction')).value ;
scores = squeeze(gather(scores)) ;

% show the classification results
[bestScore, best] = max(scores) ;

%calculate accuracy
error = 0;
i = 1;
for j= 1:500
    if best(j) ~= i
        error= error+1;
    end
end

accuracyCrop = 1-error/500;

error = 0;
i = 2;
for j= 500:1000
    if best(j) ~= i
        error= error+1;
    end
end

accuracyWeed = 1-error/500;

error = 0;
i = 3;
for j= 1000:1500
    if best(j) ~= i
        error= error+1;
    end
end

accuracyGround = 1-error/500;

%set labels
label = [ones(500,1);2*ones(500,1);3*ones(500,1)];

%calculate confusion matrix
C = confusionmat(label,best);
Accuracy = (C(1,1)+C(2,2)+C(3,3))/(C(1,1)+C(1,2)+C(1,3)+C(2,1)+C(2,2)...
       +C(2,3)+C(3,1)+C(3,2)+C(3,3));
