%% Test R-CNN Stop Sign Detector
% The R-CNN object detector can now be used to detect caves in images.
% Try it out on a test image or video:
clear;clc;
load('cave_rcnn.mat') ;
% load('trainingData2.mat');
% load('learningdatasave.mat');
num = [];
obj = VideoReader('your image or video');%������Ƶλ�� your path
% numFrames = 5;
numFrames = obj.NumberOfFrames;% ֡������

 for k = 1 : numFrames% ��ȡǰ15֡

     frame = read(obj,k);%��ȡ��k֡
     testImage = frame;
%      testImage = imread('your image');
     testImage = imresize(testImage,[512 512]);
%      figure,imshow(testImage);
     % Detect stop signs
     [bboxes, score, label] = detect(cave_rcnn, testImage, 'MiniBatchSize', 128);

%%
% The R-CNN object |detect| method returns the object bounding boxes, a
% detection score, and a class label for each detection. The labels are
% useful when detecting multiple objects, e.g. stop, yield, or speed limit
% signs. The scores, which range between 0 and 1, indicate the confidence
% in the detection and can be used to ignore low scoring detections.

% Display the detection results
[score, idx] = max(score);
% score = 0.9;
bbox = bboxes(idx, :);
annotation = sprintf('%s:(Confidence=%f)', label(idx), score);

outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);

% figure
% imshow(outputImage)
resname = strcat('E:\results\0000',num2str(k),'.jpg');%save the resuls
imwrite(outputImage,resname);
regionalname = strcat('E:\original\0000',num2str(k),'.jpg');% save the original images
imwrite(testImage,regionalname);
%% ues the grayshape function to save the learningdata for training
%this part is lifelong learning, using a evaluate model(by gray and shape features)
% if score >= 0.8
% croptestImage = imcrop(testImage,bbox);
% grayshape_score = grayshape(croptestImage);
% if grayshape_score>0.8
%     figure;
%     imshow(outputImage);
%     learningdatafilename = strcat('E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\0000\0000',num2str(k),'.jpg');
%     imwrite(testImage,learningdatafilename);% ����֡
%     learningCell={ learningdatafilename, bbox };    
%     learningdata = cell2table(learningCell);
%     learningdata.Properties.VariableNames = {'imageFilename','cave'};
%     learningdatasave = [learningdatasave;learningdata];
% %     trainingData2 =[trainingData2;learningdata];
% %     num(j) = numFrames;
% end
% end
 end