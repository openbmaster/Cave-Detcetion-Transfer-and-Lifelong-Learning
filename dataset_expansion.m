%% ���ѧϰ���ݼ�����
clc;
clear;
p = 0;
% file = dir('E:\Matlab2017b\toolbox\vision\visiondata\��һ��cave\*.jpg');
% len = length(file);
% I=imread('E:\������ͨ��ѧ\��ҵ����\ͼƬ\����\1.jpg');

%for n = 1:len
    
% I=imread(strcat('E:\Matlab2017b\toolbox\vision\visiondata\��һ��cave\',file(n).name));
I = imread('E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\�ü���\cave17.jpg');
%%
%p=p+1;
p=1;
I1=imrotate(I,5,'bilinear','loose');%��ת
% saveas();
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I1,tmpstr);

%% ��ת
I2=imrotate(I,-5,'bilinear','loose');
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
p=p+1;
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I2,tmpstr);
% saveas();

I3=imrotate(I,10,'bilinear','loose');
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I3,tmpstr);

% saveas();
I4=imrotate(I,-10,'bilinear','loose');
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I4,tmpstr);

% saveas();
I5=imrotate(I,15,'bilinear','loose');
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I5,tmpstr);

% saveas();
I6=imrotate(I,-15,'bilinear','loose');
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I6,tmpstr);

%% �Ҷȱ仯
% saveas();
J = imadjust(I,[.2 .3 0; .6 .7 1],[]);%�Ҷȱ任
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(J,tmpstr);



%% ����
%saveas
F1 = imnoise(I,'speckle',0.1); %�ߵ�����
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(F1,tmpstr);

%saveas
F2 = imnoise(I,'gaussian',0.1); %��˹����
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(F2,tmpstr);

%saveas
% F3 = imnoise(I,'localvar',0.1); %������
% p=p+1;
% str = 'E:\������ͨ��ѧ\��ҵ����\ͼƬ\��������\';
% tmpstr = strcat(str,num2str(p),'.jpg');
% imwrite(F3,tmpstr);

%saveas
F4 = imnoise(I,'salt & pepper',0.1); %��������
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(F4,tmpstr);

%saveas
F5 = imnoise(I,'poisson'); %��������
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(F5,tmpstr);

%% ģ������
%saveas
PSF1 = fspecial('motion',10,10);
N11 = imfilter(I,PSF1,'conv','circular');%�˶�ģ��
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N11,tmpstr);

PSF2 = fspecial('motion',12,12);
N12 = imfilter(I,PSF2,'conv','circular');%�˶�ģ��
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N12,tmpstr);

PSF2 = fspecial('motion',15,15);
N12 = imfilter(I,PSF2,'conv','circular');%�˶�ģ��
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N12,tmpstr);

PSF3 = fspecial('motion',20,20);
N13 = imfilter(I,PSF3,'conv','circular');%�˶�ģ��
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N13,tmpstr);

PSF4 = fspecial('motion',25,25);
N14 = imfilter(I,PSF4,'conv','circular');%�˶�ģ��
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N14,tmpstr);

%saveas
r=10;
PSF2=fspecial('disk',r);   %�õ�����ɢ����
N2 = imfilter(I,PSF2,'symmetric','conv');  %ʵ��ɢ��ģ��
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N2,tmpstr)

%����任
[w,h]=size(I);
theta=pi/4;
t=[100,100];
s=0.5;
% test affine transform
H_a=projective2d([1 0.5 t(1);
                 0 0.5 t(2);
                 0 0  1]');
newimg=imwarp(I,H_a);
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);
% figure,imshow(newimg);

%��Ӱ�任1
[w,h]=size(I);
theta=pi/4;
t=[100,100];
s=0.5;
% test projective transform
H_P=projective2d([0.765,-0.122,-0.0002;
                 -0.174,0.916,9.050e-05;
                  105.018,123.780,1]);
newimg=imwarp(I,H_P);
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);
% figure,imshow(newimg);

%��Ӱ�任2
[w,h]=size(I);
theta=pi/4;
t=[100,100];
s=0.5;
% test projective transform
H_P=projective2d([0.6,-0.1,-0.0002;
                 -0.1,0.9,9.050e-05;
                  105,123,1]);
newimg=imwarp(I,H_P);
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);


%��Ӱ�任3
[w,h]=size(I);
theta=pi/4;
t=[100,100];
s=0.5;
% test projective transform
H_P=projective2d([0.5,-0.2,-0.0002;
                 -0.2,0.8,9.050e-05;
                  100,110,1]);
newimg=imwarp(I,H_P);
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);

%��Ӱ�任4
[w,h]=size(I);
theta=pi/4;
t=[100,100];
s=0.5;
% test projective transform
H_P=projective2d([0.5,-0.2,-0.0002;
                 -0.2,0.8,9.050e-05;
                  100,110,1]);
newimg=imwarp(I,H_P);
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);

%��Ӱ�任6
[w,h]=size(I);
theta=pi/4;
t=[100,100];
s=0.5;
% test projective transform
H_P=projective2d([0.965,-0.122,-0.0002;
                 -0.04,0.916,9.050e-05;
                  120.018,123.780,1]);
newimg=imwarp(I,H_P);
p=p+1;
str = 'E:\������ͨ��ѧ\��ҵ����\�Խ����ݼ�\��ѡ����\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);
%end