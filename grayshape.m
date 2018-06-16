function [m] = grayshape(croptestImage)

RGB = croptestImage;
% figure,imshow(RGB);

%%
% Step 2: Convert image from rgb to gray 
GRAY = rgb2gray(RGB);
% figure,imshow(GRAY);
%%
% Step 3: Threshold the image Convert the image to black and white in order
% to prepare for boundary tracing using bwboundaries. 

BW=wolf(GRAY, [250 250]);% wolf法二值化局部阈值

% se1=[0   0   0   0   0   0   1;
%     0   0   0   0   0   1   0;
%     
%     0   0   0   0   1   0   0;
%     0   0   0   1   0   0   0;
%     0   0   1   0   0   0   0;
%     0   1   0   0   0   0   0;
%     1   0   0   0   0   0   0];
% BW=imdilate(BW,se1,2); % 膨胀1
% se2=[1   0   0   0   0   0   0;
%     0   1   0   0   0   0   0;
%     0   0   1   0   0   0   0;
%     0   0   0   1   0   0   0;
%     0   0   0   0   1   0   0;
%     0   0   0   0   0   1   0;
%     0   0   0   0   0   0   1];
% BW=imdilate(BW,se2,2); % 膨胀2
% se3=[0   0   0   0   0   0   0;
%     0   0   0   0   0   0   0;
%     0   0   0   0   0   0   0;
%     1   1   1   1   1   1   1;
%     0   0   0   0   0   0   0;
%     0   0   0   0   0   0   0;
%     0   0   0   0   0   0   0];
% BW=imdilate(BW,se3,2); % 膨胀3
% se4=[0   0   0   1   0   0   0;
%     0   0   0   1   0   0   0;
%     0   0   0   1   0   0   0;
%     0   0   0   1   0   0   0;
%     0   0   0   1   0   0   0;
%     0   0   0   1   0   0   0;
%     0   0   0   1   0   0   0];
% BW=imdilate(BW,se4,50); % 膨胀4

%%
% Step 4: Invert the Binary Image
BW = ~ BW;
% figure,imshow(BW);
%%
%%删除小面积区域
% BW = bwareaopen(BW,5000);

% title('删除小面积')
% fill a gap in the pen's cap
% se = strel('disk',2);
% BW = imclose(BW,se);
% figure,imshow(BW);
% fill any holes, so that regionprops can be used to estimate
% the area enclosed by each of the boundaries
% BW = imfill(BW,'holes');
% figure,imshow(BW);
%%
% Step 5: Find the boundaries Concentrate only on the exterior boundaries.
% Option 'noholes' will accelerate the processing by preventing
% bwboundaries from searching for inner contours. 
[B,L] = bwboundaries(BW, 'noholes');


%%
% Step 6: Determine objects properties
stats = regionprops(L, 'all'); % we need 'BoundingBox' and 'Extent'

ME=zeros(1,length(B));
for k = 1:length(B)

  % obtain (X,Y) boundary coordinates corresponding to label 'k'
  boundary = B{k};

  % compute a simple estimate of the object's perimeter周长
  delta_sq = diff(boundary).^2;    
  perimeter = sum(sqrt(sum(delta_sq,2)));
  
  % obtain the area calculation corresponding to label 'k'面积
  area = stats(k).Area;
  
  % compute the roundness metric计算圆度
  metric = 4*pi*area/perimeter^2;
  if metric == inf
      metric = 0;
  end
  ME(k)=metric;
end
 
     [m,i]=max(ME);
%      metric_string = sprintf('%2.2f',m);
%      boundary = B{i};

%      figure;
%      imshow(RGB);
%      hold on;
%      plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
%      text(boundary(1,2)-35,boundary(1,1)+13,metric_string,'Color','r',...
%      'FontSize',60,'FontWeight','bold');
end

