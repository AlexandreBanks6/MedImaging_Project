clear 
clc
close all

videoPath = "C:\Users\randy\Downloads\MooreBanks_Results\MooreBanks_Results\Roll2\Recorder_2_Nov11_20-17-45.mp4";
robotDataPath = "C:\Users\randy\Downloads\MooreBanks_Results\MooreBanks_Results\Roll2\data.csv";

[robot_data_resamp,frame_vec, t_dvrk,time_us] = dvrk_tooldat_function(videoPath,robotDataPath,0);



%iterate through frames, collecting absolute magnitude of optical flow  and
%plot results
vidReader=VideoReader(videoPath); %Create an object with the video
opticFlow=opticalFlowLK('NoiseThreshold',0.001); %Increasing number means movement of object has less impact on calculation
% Parameters
VelThresh=1; %Threshold for velocity magnitudes that are not considered movement
radmin=8;
radmax=30;
HughSensit=0.8;

 m=1;
 flowMag = [];
    while hasFrame(vidReader)
        frame=readFrame(vidReader);
        CropRec=[1607.51,85.51,883.98,740.98];
        frameCropped=imcrop(frame,CropRec); %Crops just the US portion of the image
        frameGray=rgb2gray(frameCropped);
        flow=estimateFlow(opticFlow,frameGray);
        flowMag(end+1) = sum(sum(flow.Magnitude));
        m=m+1;
    end
    %process the roll angle 
    rollAngle = robot_data_resamp(:,4);
    for i = 1:length(rollAngle)
        if rollAngle(i)>100
            rollAngle(i) = 100;
        elseif rollAngle(i)<-100
            rollAngle(i) = -100;
        end
    end 
    a = 1;
    b = [1/4 1/4 1/4 1/4];
    flowMagFiltered = filter(b,a,flowMag);
    figure;
    subplot(1,2,1); plot(flowMagFiltered)
    subplot(1,2,2); plot(rollAngle)

    %% Notes: 
    %The above method had difficulty finding the correct roll angle. The
    %frame with the highest amount of optical flow did not correlate to the
    %frame that correlates to the roll angle

    %final angle in roll2 is -5.11049

