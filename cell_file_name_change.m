% ('vehicles/image_00001.jpg');
% 
% ('隧道航拍\001.jpg');
%% ground truth数据转成训练格式
trainingData2 = objectDetectorTrainingData(gTruth);


%% 转换后数据修改imageFilename满足数据格式要求
% for i=1:1240
% 
%     strtmp=strcat('cave2\',num2str(i),'.jpg');
%     
% trainingData2.imageFilename(i)={strtmp};
% 
% end