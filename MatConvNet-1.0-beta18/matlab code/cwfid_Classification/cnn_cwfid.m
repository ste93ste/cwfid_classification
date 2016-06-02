%This function create the network and train it with the CWFID dataset
function [net, info] = cnn_cwfid(varargin)

%setup MatConvNet
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

%Uncomment the line below if you want to use GPU
%vl_compilenn('enableGpu', true);

%batch normalization
opts.batchNormalization = true ;
%network type: simplenn or dagnn
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

%create the folder (bnorm if opts.batchNormalization = true
sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['cwfid-baseline' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataDir = fullfile(vl_rootnn, 'data', 'cwfid') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

%if you want to use a different network change the function name
%e.g. cnn_cwfid_init_1
%if you have already a folder with a trained network, please remove it
%before train
net = cnn_cwfid_init_5('batchNormalization', opts.batchNormalization, ...
                     'networkType', opts.networkType) ;
imdb = getCwfidImdb(opts) ;
mkdir(opts.expDir) ;
save(opts.imdbPath, '-struct', 'imdb') ;


net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:3,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

%switch to different training type depending the network type is simplenn
%or dagnn
switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train);

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getCwfidImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure
cropLabel = 1;
weedLabel = 2;
groundLabel = 3;

trainingDir = fullfile(opts.dataDir,'training');
%only the first half of the testing images are using for validation
validationDir = fullfile(opts.dataDir,'testing');

trainingCropDir = fullfile(trainingDir,'crop');
trainingWeedDir = fullfile(trainingDir,'weed');
trainingGroundDir = fullfile(trainingDir,'ground');

validationCropDir = fullfile(validationDir,'crop');
validationWeedDir = fullfile(validationDir,'weed');
validationGroundDir = fullfile(validationDir,'ground');

%get training crop images and labels
listing = dir(trainingCropDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : size(listing)
    cropImTraining(:,:,:,i)=imread(fullfile(trainingCropDir,listing(i).name));
    cropLabelTraining(i) = cropLabel;
end

%get training weed images and labels
listing = dir(trainingWeedDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : size(listing)
    weedImTraining(:,:,:,i)=imread(fullfile(trainingWeedDir,listing(i).name));
    weedLabelTraining(i) = weedLabel;
end

%get training ground images and labels
listing = dir(trainingGroundDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : size(listing)
    groundImTraining(:,:,:,i)=imread(fullfile(trainingGroundDir,listing(i).name));
    groundLabelTraining(i) = groundLabel;
end

%get validation crop images and labels
listing = dir(validationCropDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : (size(listing,1) - 500)
    cropImValidation(:,:,:,i)=imread(fullfile(validationCropDir,listing(i).name));
    cropLabelValidation(i) = cropLabel;
end

%get validation weed images and labels
listing = dir(validationWeedDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : (size(listing,1) - 500)
    weedImValidation(:,:,:,i)=imread(fullfile(validationWeedDir,listing(i).name));
    weedLabelValidation(i) = weedLabel;
end

%get validation ground images and labels
listing = dir(validationGroundDir);
listing=listing(~ismember({listing.name},{'.','..','Thumbs.db'}));
for i=1 : (size(listing,1) - 500)
    groundImValidation(:,:,:,i)=imread(fullfile(validationGroundDir,listing(i).name));
    groundLabelValidation(i) = groundLabel;
end

%create the set of labels 1 is for training 2 is for validation
set = [ones(1,numel([cropLabelTraining weedLabelTraining groundLabelTraining]))...
    2*ones(1,numel([cropLabelValidation weedLabelValidation groundLabelValidation]))];

data = single(reshape(cat(4,cropImTraining,weedImTraining,groundImTraining,...
    cropImValidation,weedImValidation,groundImValidation),51,51,3,[]));
%subtract the data mean 
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;

imdb.images.data = data ;
imdb.images.labels = cat(2,cropLabelTraining,weedLabelTraining,groundLabelTraining,...
    cropLabelValidation,weedLabelValidation,groundLabelValidation) ;

imdb.images.set = set ;
imdb.meta.sets = {'train','val'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:3,'uniformoutput',false) ;
