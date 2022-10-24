clear
clc
close all

%% Reading in Video
vidReader=VideoReader('SoccerVid_Good_Trim.avi');

opticFlow=opticalFlowLK;

%{
h=figure;
movegui(h);
hViewPanel=uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);
%}

tic
i=1;
while hasFrame(vidReader)
    frameRGB=readFrame(vidReader);
    frameGray=im2gray(frameRGB);
    flow=estimateFlow(opticFlow,frameGray);

   % imshow(frameGray);
   % hold on
    %plot(flow,'DecimationFactor',[5 5],'ScaleFactor',2,'Parent',hPlot);
    %hold off
    %pause(10^-3);
    disp(i);
    i=i+1;
end
timeend=toc;
disp(timeend);

