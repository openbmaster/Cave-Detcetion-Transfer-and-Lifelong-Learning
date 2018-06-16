clear;close all;clc;
obj = VideoReader('E:\北京交通大学\基于灰度形状洞库识别\航拍门头沟东方红隧道-国语720P.mp4');%输入视频位置
numFrames = obj.NumberOfFrames;% 帧的总数
 for k = 1 : numFrames% 读取前15帧
     frame = read(obj,k);%读取第几帧
     % imshow(frame);%显示帧
%      frame=imresize(frame,[128 128]);
     imwrite(frame,strcat('E:\北京交通大学\毕业论文\自建数据集\东方红隧道\0000',num2str(k),'.jpg'),'jpg');% 保存帧
 end
 
% clear;
% close all;
% clc;
% str=':\Matlab2017b\toolbox\vision\visiondata\mini_cave\';
% hs = tight_subplot(6, 6, [0.01, 0.0001], [0.01, 0.01], [0.01, 0.01]);
% for p=1:25  %n为图库中的图片数
%     ai=imread([str,num2str(p),'.jpg']);
%     ci=imresize(ai,[64 64]);        
% %   figure
%     axes(hs(p));
%     imshow(ci);
%     hold on;
%     imwrite(ci,strcat('E:\Matlab2017b\toolbox\vision\visiondata\归一化cave\',num2str(p),'.jpg'));
% end