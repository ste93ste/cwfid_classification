% IMAGE ANALYSIS - PROJECT 2015/2016
%
% Script for extracting from CWFID N random image patches centered
% on pixels which are marked in the ground truth as a given class.
% Patches can be extracted for testing or training. Training paches are
% extracted from the first 40 imgs. of the dataset. Testing patches come
% from the last 20.
%
% Requires the CWFID dataset to be downloaded and placed in the workspace.

warning('off', 'Images:initSize:adjustingMag');
close all, clear all
clc

%% Configurable parameters

N = 1000;           % Number of patches for each class.
psz = 51;           % Size of the sides of the patch [pixels].

option = 1;         % 1 for training patches (extracted from first 40 images),
                        % 0 for test patches (extracted from last 20).
                        
se = strel('disk', 8);  % Shape for the erosion of the center pixels for the patches.

%% Code

np = N * ones(1,3); % Overall number of remaining patches to be extracted 
                        % for each class. One counter is not sufficient since
                        % not all images have information for all three classes.
classes = [1 1 1];      % Class selector for patch extraction

if option == 0      % Extracting set of patches for training
   n_im = 40;                   % Number of images of the dataset for trai.
   bpath = 'patches\training\'; % Base path for folder
elseif option == 1  % Extracting set of patches for testing
   n_im = 20;                   % Number of images of the dataset for test.
   bpath = 'patches\testing\';  % Base path for folder
else
    return;
end

mkdir(bpath);       % Create directories
mkdir(bpath,'weed'); 
mkdir(bpath,'crop'); 
mkdir(bpath,'ground');

fs = ['%0' num2str(floor(log10(N))+1) 'd']; % Format string used for output
disp('Begin extracting random 51x51 patch images:')
str = [fs '\t' fs '\t' fs];
str2 = ['Remaining patches:\n\tweed\tcrop\tground\n\t' str];
fprintf(str2, np(1), np(2), np(3))

while(sum(np))  % while there are remaining patches 
    rand_im_ix = randi(n_im,1,max(np));  
                                        % Image numbers to get patches from.
                                        % This maintains randomness
                                        % minimizing image reads.
                                       
    imgs = zeros(1,n_im);               % Number of patches / each image in
                                            % rand_im_ix
    for n = 1:max(np)
        imgs(rand_im_ix(n)) = imgs(rand_im_ix(n)) + 1;
    end

    for im = find(imgs)
        ann_path = strcat( 'dataset-1.0\annotations\', sprintf('%03d',im + 40*option),...
                '_annotation.png');     % Path for annotation file
            
        img_path = strcat( 'dataset-1.0\images\', sprintf('%03d',im + 40*option),...
                '_image.png');          % Path for image file

        A = imread( ann_path, 'png');   % Read annotation mask
        I = imread( img_path, 'png');   % Read corresponding image   

        for n = 1:imgs(im)     % Each iteration finds a new rand. 51x51 patch

            c = 1:3;
            classes = ~~np;    % Flags indicating remaining patches != 0
            for c = c(classes) % Main loop for each class (weed, crop, ground) 
                
                switch c      % Perform class-oriented operations
                    case 1
                        IMask = im2bw(A(:,:,1));    % Binary mask (weed)
                        filename = [bpath 'weed\' sprintf(fs,np(c)) '_weed.png'];
                    case 2
                        IMask = im2bw(A(:,:,2));    % Binary mask (crop)
                        filename = [bpath 'crop\' sprintf(fs,np(c)) '_crop.png'];
                    otherwise                       % Binary mask (ground)
                        IMask = ~(im2bw(A(:,:,1)) | im2bw(A(:,:,2)));
                        filename = [bpath 'ground\' sprintf(fs,np(c)) '_ground.png'];
                end
                
                Centers = IMask(ceil(psz/2):end-floor(psz/2),...
                          ceil(psz/2):end-floor(psz/2)); % Possible center points
                Centers = imerode(Centers,se);  % Perform erosion
                [y, x] = find(Centers);  % x,y: coordinate arrays for nonzero elts.

                if ~(isempty(y) || isempty(x)) % Then this image has 
                                               % annotations for this class 
               
                    np(c) = np(c) - 1;    % Decrement remaining patches
                    p = randi(length(y)); % Index for (random again) center pix
                    P = I(y(p):y(p)+psz-1, x(p):x(p)+psz-1, :); % Patch
                    imwrite(P, filename, 'png');
                end  
            end
            for i = 1:2+3*floor(log10(N)+1), fprintf('\b'); end     % Super stylish output
            fprintf(str, np(1), np(2), np(3))
        end
    end
end
fprintf('\nDone!!\n')
