%% 深度学习数据集扩充
clc;
clear;
p = 0;
% file = dir('E:\Matlab2017b\toolbox\vision\visiondata\归一化cave\*.jpg');
% len = length(file);
% I=imread('E:\北京交通大学\毕业论文\图片\洞库\1.jpg');

%for n = 1:len
    
% I=imread(strcat('E:\Matlab2017b\toolbox\vision\visiondata\归一化cave\',file(n).name));
I = imread('E:\北京交通大学\毕业论文\自建数据集\裁剪后\cave17.jpg');
%%
%p=p+1;
p=1;
I1=imrotate(I,5,'bilinear','loose');%旋转
% saveas();
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I1,tmpstr);

%% 旋转
I2=imrotate(I,-5,'bilinear','loose');
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
p=p+1;
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I2,tmpstr);
% saveas();

I3=imrotate(I,10,'bilinear','loose');
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I3,tmpstr);

% saveas();
I4=imrotate(I,-10,'bilinear','loose');
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I4,tmpstr);

% saveas();
I5=imrotate(I,15,'bilinear','loose');
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I5,tmpstr);

% saveas();
I6=imrotate(I,-15,'bilinear','loose');
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(I6,tmpstr);

%% 灰度变化
% saveas();
J = imadjust(I,[.2 .3 0; .6 .7 1],[]);%灰度变换
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(J,tmpstr);



%% 噪声
%saveas
F1 = imnoise(I,'speckle',0.1); %斑点噪声
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(F1,tmpstr);

%saveas
F2 = imnoise(I,'gaussian',0.1); %高斯噪声
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(F2,tmpstr);

%saveas
% F3 = imnoise(I,'localvar',0.1); %白噪声
% p=p+1;
% str = 'E:\北京交通大学\毕业论文\图片\数据扩充\';
% tmpstr = strcat(str,num2str(p),'.jpg');
% imwrite(F3,tmpstr);

%saveas
F4 = imnoise(I,'salt & pepper',0.1); %椒盐噪声
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(F4,tmpstr);

%saveas
F5 = imnoise(I,'poisson'); %泊松噪声
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(F5,tmpstr);

%% 模糊噪声
%saveas
PSF1 = fspecial('motion',10,10);
N11 = imfilter(I,PSF1,'conv','circular');%运动模糊
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N11,tmpstr);

PSF2 = fspecial('motion',12,12);
N12 = imfilter(I,PSF2,'conv','circular');%运动模糊
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N12,tmpstr);

PSF2 = fspecial('motion',15,15);
N12 = imfilter(I,PSF2,'conv','circular');%运动模糊
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N12,tmpstr);

PSF3 = fspecial('motion',20,20);
N13 = imfilter(I,PSF3,'conv','circular');%运动模糊
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N13,tmpstr);

PSF4 = fspecial('motion',25,25);
N14 = imfilter(I,PSF4,'conv','circular');%运动模糊
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N14,tmpstr);

%saveas
r=10;
PSF2=fspecial('disk',r);   %得到点扩散函数
N2 = imfilter(I,PSF2,'symmetric','conv');  %实现散焦模糊
p=p+1;
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(N2,tmpstr)

%仿射变换
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
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);
% figure,imshow(newimg);

%射影变换1
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
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);
% figure,imshow(newimg);

%射影变换2
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
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);


%射影变换3
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
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);

%射影变换4
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
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);

%射影变换6
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
str = 'E:\北京交通大学\毕业论文\自建数据集\精选扩充\';
tmpstr = strcat(str,num2str(p),'.jpg');
imwrite(newimg,tmpstr);
%end