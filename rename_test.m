%% rename
file = dir('E:\Matlab2017b\toolbox\vision\visiondata\cave2\*.jpg');
len = length(file);
for i = 1 : len
   oldname = file(i).name;
   I=imread(oldname);
%    tmpname=strcat('0000',num2str(i),'.jpg');
   newname = strcat('0000',num2str(i),'.jpg');
%    eval(['!rename' 32 oldname 32 newname]);
   imwrite(I,strcat('E:\Matlab2017b\toolbox\vision\visiondata\cave\',newname));
end