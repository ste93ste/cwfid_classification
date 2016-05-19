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

accuracy = (accuracyCrop+accuracyWeed+accuracyGround)/3;
