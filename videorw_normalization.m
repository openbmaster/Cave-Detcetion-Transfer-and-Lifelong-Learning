clear;close all;clc;
obj = VideoReader('E:\������ͨ��ѧ\���ڻҶ���״����ʶ��\������ͷ�����������-����720P.mp4');%������Ƶλ��
numFrames = obj.NumberOfFrames;% ֡������
 for k = 1 : numFrames% ��ȡǰ15֡
     frame = read(obj,k);%��ȡ�ڼ�֡
     % imshow(frame);%��ʾ֡
%      frame=imresize(frame,[128 128]);
     imwrite(frame,strcat('E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\���������\0000',num2str(k),'.jpg'),'jpg');% ����֡
 end
 
% clear;
% close all;
% clc;
% str=':\Matlab2017b\toolbox\vision\visiondata\mini_cave\';
% hs = tight_subplot(6, 6, [0.01, 0.0001], [0.01, 0.01], [0.01, 0.01]);
% for p=1:25  %nΪͼ���е�ͼƬ��
%     ai=imread([str,num2str(p),'.jpg']);
%     ci=imresize(ai,[64 64]);        
% %   figure
%     axes(hs(p));
%     imshow(ci);
%     hold on;
%     imwrite(ci,strcat('E:\Matlab2017b\toolbox\vision\visiondata\��һ��cave\',num2str(p),'.jpg'));
% end