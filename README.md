# CaveTargetDetector

#matlab 2017b

#This project focus on how to find the cave target like cave, tunnel, some warehouse, et al.

#A trained model named cave_rcnn.mat use about 1240 images, fast rcnn(from matlab documentation),  cifar-10 dataset

#Using this project you should,
#1.open test1.m
#2.load test images or videos
#3.maybe you should change some code to adapt to images or videos
#4 run

#If you want to train a new model, you should,
#1.using matlab app image labeler load your data
#2.label and export data to workshop
#3.using cell_file_name_change.m
#4.change some filename in SeriesDetectorExample.m as your export filename
#5.you should be careful at something like 'true' or 'false', and you can search 'fast rcnn' at matlab documentation 