clear 
clc
close all

%% ---------------<Reading Data>---------------

% datapath_robot=['C:\Users\playf\OneDrive\Documents\UBC\Alexandre_UNI_2022_2023\' ...
%     'Semester1\ELEC_523_MedImaging\Project\MooreBanks_Results\Trial4\Registration\data.csv'];
% datapath_video=['C:/Users/playf/OneDrive/Documents/UBC/Alexandre_UNI_2022_2023/Semester1'...
%     '/ELEC_523_MedImaging/Project/MooreBanks_Results/Trial4/Registration/Recorder_2_Nov11_20-01-30.mp4'];

datapath_robot=['C:\Users\playf\OneDrive\Documents\UBC\Alexandre_UNI_2022_2023\' ...
    'Semester1\ELEC_523_MedImaging\Project\MooreBanks_Results\Trial3\Registration\data.csv'];
datapath_video=['C:/Users/playf/OneDrive/Documents/UBC/Alexandre_UNI_2022_2023/Semester1'...
    '/ELEC_523_MedImaging/Project/MooreBanks_Results/Trial3/Registration/Recorder_2_Nov11_19-57-18.mp4'];



%t_dvrk is the old dvrk time, and time_us is the ultrasound time (the one
%we use)
[robot_data_resamp,t_dvrk,time_us]=dvrk_tooldat_function(datapath_video,datapath_robot,0); %Interpolates the da Vinci data, to match the US frames

dvrk_xyz=robot_data_resamp(:,[1:3]); %End-effector xyz (this is the reference track)

% Finding Points in Each Frame
[framearray,framevec]=PointDetect(datapath_video); %Returns cell where each cell corresponds to points in a frame
                                                    %framevec is a vector
                                                    %of the frames
                                                    %corresponding to
                                                    %points found


%% This is for testing
[trackpoints,framevec]=TrackFinder(framearray,framevec,datapath_robot); %This returns the track of points


%% -----------------------<Function definitions>---------------------------
function [pointarray,framevec]=PointDetect(datapath)
%Setting up Optic Flow Object
opticFlow=opticalFlowLK('NoiseThreshold',0.001); %Increasing number means movement of object has less impact on calculation
%Reading in video
vidReader=VideoReader(datapath);
% Parameters
VelThresh=1; %Threshold for velocity magnitudes that are not considered movement
radmin=8;
radmax=30;
HughSensit=0.8;

CropRec=[1607.51,85.51,883.98,740.98];
m=1;
count=1;
pointarray={};
framevec=[];
    while hasFrame(vidReader)
        frame=readFrame(vidReader);
        frameCropped=imcrop(frame,CropRec); %Crops just the US portion of the image
        frameGray=rgb2gray(frameCropped);
        
        flow=estimateFlow(opticFlow,frameGray);
        Mask=imbinarize(flow.Magnitude,VelThresh);
        
        Mask=imopen(Mask,strel('disk',2));
        [centroids,radii,metric]=imfindcircles(Mask,[radmin radmax],'Sensitivity',HughSensit);

        if ~isempty(centroids)
            pointarray(count)={centroids};
            framevec=[framevec;m];
            count=count+1;
        end
               

        m=m+1;
        disp(m)
    end

end


function [trackpoints,framevec]=TrackFinder(framearray,framevec,datapath_robot)


end


