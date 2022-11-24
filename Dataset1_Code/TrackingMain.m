clear 
clc
close all

%% ---------------<Reading Data>---------------

% datapath_robot=['C:\Users\playf\OneDrive\Documents\UBC\Alexandre_UNI_2022_2023\' ...
%     'Semester1\ELEC_523_MedImaging\Project\MooreBanks_Results\Trial4\Registration\data.csv'];
% datapath_video=['C:/Users/playf/OneDrive/Documents/UBC/Alexandre_UNI_2022_2023/Semester1'...
%     '/ELEC_523_MedImaging/Project/MooreBanks_Results/Trial4/Registration/Recorder_2_Nov11_20-01-30.mp4'];

datapath_robot=['C:\Users\randy\Downloads\MooreBanks_Results\MooreBanks_Results\Trial4\Registration\data.csv'];
datapath_video=['C:\Users\randy\Downloads\MooreBanks_Results\MooreBanks_Results\Trial4\Registration\Recorder_2_Nov11_20-01-30.mp4'];

% datapath_robot=['C:\Users\playf\OneDrive\Documents\UBC\Alexandre_UNI_2022_2023\' ...
%     'Semester1\ELEC_523_MedImaging\Project\MooreBanks_Results\Trial3\Registration\data.csv'];
% datapath_video=['C:/Users/playf/OneDrive/Documents/UBC/Alexandre_UNI_2022_2023/Semester1'...
%     '/ELEC_523_MedImaging/Project/MooreBanks_Results/Trial3/Registration/Recorder_2_Nov11_19-57-18.mp4'];



%t_dvrk is the old dvrk time, and time_us is the ultrasound time (the one
%we use)
[robot_data_resamp,t_dvrk,time_us]=dvrk_tooldat_function(datapath_video,datapath_robot,0); %Interpolates the da Vinci data, to match the US frames

dvrk_xyz=robot_data_resamp(:,[1:3]); %End-effector xyz (this is the reference track)

% Finding Tracks
tracks=TrackDetect(datapath_video);




%% ------------------------<Similarity Measure>----------------------------
NumTracks=length(tracks); %The number of tracks detected

distMeas=zeros(1,NumTracks);
for i = 1:NumTracks %Loops for the number of trajectories we have 
    traj=tracks(i).TotalTrack; %Extracts array of trajectory for a given track
    frame_vec=tracks(i).framenum; %Vector of frames corresponding to point in the trajectoryf
    frame_vec=frame_vec;
    %Only keeping the first 50 frames
    frame_vec_indx=frame_vec<=50;
    frame_vec=frame_vec(frame_vec_indx);
    traj=traj(frame_vec_indx,:);

    ZeroRows=traj(:,1)==0;
    traj=traj(~ZeroRows,:); %Removes zeros from trajectory from US
    frame_vec=frame_vec(~ZeroRows,:); %Vector of frames corresponding to nonzero locations in US
    
    if isempty(frame_vec)
        distMeas(i)=10000;
        continue

    end
    
    %Standardizing The Data
    traj(:,1)=(traj(:,1)-mean(traj(:,1)))/std(traj(:,1));
    traj(:,2)=(traj(:,2)-mean(traj(:,2)))/std(traj(:,2));
    %PCA on the trajectory data
    [coeff,score,latent]=pca(traj);

    % Prior to PCA we must standardize our data by subtracting the mean and
    % dividing by the standard deviation
    dvrk_x=(dvrk_xyz(frame_vec,1)-mean(dvrk_xyz(frame_vec,1)))/std(dvrk_xyz(frame_vec,1));
    dvrk_y=(dvrk_xyz(frame_vec,2)-mean(dvrk_xyz(frame_vec,2)))/std(dvrk_xyz(frame_vec,2));
    dvrk_z=(dvrk_xyz(frame_vec,3)-mean(dvrk_xyz(frame_vec,3)))/std(dvrk_xyz(frame_vec,3));

    [refcoeff,refscore,reflatent]=pca([dvrk_x,dvrk_y,dvrk_z]); %Returns the loading vectors (principles components)


    %refscore_new=refscore(frame_vec,[1:2]);
    distMeas(i)=sqrt(sum((score(:,1)-refscore(:,1)).^2+(score(:,2)-refscore(:,2)).^2))/length(frame_vec);
    %We use the two-sample kolmogorov-smirnov test to quantify the
    %similarity between the two data distributions (because they have
    %different number of points (this can be changed later)
    %{
    [~,p1]=kstest2(score(:,1),refscore(:,1));
    [~,p2]=kstest2(score(:,2),refscore(:,2));
    distMeas(i)=(p1+p2)/2; %Smaller the p-value, more similar the distributions
    %}

end

min_indx=distMeas==min(distMeas);
%min_indx=2;
us_track_raw=tracks(min_indx).TotalTrack; %With zeros
frame_vec_raw=tracks(min_indx).framenum; %Vector of frames corresponding to point in the trajectory

%Only keeping the first 50 frames
frame_vec_indx=frame_vec_raw<=50;
frame_vec=frame_vec_raw(frame_vec_indx);
us_track=us_track_raw(frame_vec_indx,:);


ZeroRows=us_track(:,1)==0;
us_track=us_track(~ZeroRows,:); %Removes zeros from trajectory from US
frame_vec=frame_vec(~ZeroRows,:); %Vector of frames corresponding to nonzero locations in US
frame_vec=frame_vec;


%% Now we do registration
%Find three points with smallest Euclidean distance (after PCA)
us_track_st(:,1)=(us_track(:,1)-mean(us_track(:,1)))/std(us_track(:,1));
us_track_st(:,2)=(us_track(:,2)-mean(us_track(:,2)))/std(us_track(:,2));
%PCA on the trajectory data
[coeff,score,latent]=pca(us_track_st);

dvrk_x=(dvrk_xyz(frame_vec,1)-mean(dvrk_xyz(frame_vec,1)))/std(dvrk_xyz(frame_vec,1));
dvrk_y=(dvrk_xyz(frame_vec,2)-mean(dvrk_xyz(frame_vec,2)))/std(dvrk_xyz(frame_vec,2));
dvrk_z=(dvrk_xyz(frame_vec,3)-mean(dvrk_xyz(frame_vec,3)))/std(dvrk_xyz(frame_vec,3));

[refcoeff,refscore,reflatent]=pca([dvrk_x,dvrk_y,dvrk_z]);

EuclidVec=sqrt((score(:,1)-refscore(:,1)).^2+(score(:,2)-refscore(:,2)).^2);

Min_Nums=mink(EuclidVec,3); %Three smallest indecis
Min1=find(EuclidVec==Min_Nums(1));
Min2=find(EuclidVec==Min_Nums(2));
Min3=find(EuclidVec==Min_Nums(3));

Min1_frame=frame_vec(find(EuclidVec==Min_Nums(1)));
Min2_frame=frame_vec(find(EuclidVec==Min_Nums(2)));
Min3_frame=frame_vec(find(EuclidVec==Min_Nums(3)));

%Convert US Image coordinates into TRUS coordinates

%find maximum pixel value 
vidReader=VideoReader(datapath_video);
max_h = 0; 
max_g = 0;
while hasFrame(vidReader)
    frame=readFrame(vidReader);
    CropRec=[1607.51,85.51,883.98,740.98];
    frameCropped=imcrop(frame,CropRec); %Crops just the US portion of the image
    [max_h, max_g] = size(frameCropped);
    break;
end
h = us_track(Min1,1);
g = us_track(Min1,2);
l = 0.065; %length (and width) of ultrasound screen in meters
%assume all distances are in meters. 
TRUS_Point1=[-1*g/max_g*l, (max_h-h)/max_h*l*sin(2*pi*robot_data_resamp(Min1_frame,4)/360), (max_h-h)/max_h*l*cos(2*pi*robot_data_resamp(Min1_frame,4)/360)];

h = us_track(Min2,1);
g = us_track(Min2,2);
l = 0.065; %length (and width) of ultrasound screen in meters
%assume all distances are in meters. 
TRUS_Point2=[-1*g/max_g*l, (max_h-h)/max_h*l*sin(2*pi*robot_data_resamp(Min2_frame,4)/360), (max_h-h)/max_h*l*cos(2*pi*robot_data_resamp(Min2_frame,4)/360)];

h = us_track(Min3,1);
g = us_track(Min3,2);
l = 0.065; %length (and width) of ultrasound screen in meters
%assume all distances are in meters. 
TRUS_Point3=[-1*g/max_g*l, (max_h-h)/max_h*l*sin(2*pi*robot_data_resamp(Min3_frame,4)/360), (max_h-h)/max_h*l*cos(2*pi*robot_data_resamp(Min3_frame,4)/360)];

TRUS_Coordinates = [TRUS_Point1;TRUS_Point2;TRUS_Point3];

% daVinci_Coordinates
dV_Point1 = dvrk_xyz(Min1_frame,:);
dV_Point2 = dvrk_xyz(Min2_frame, :); 
dV_Point3 = dvrk_xyz(Min3_frame,:);
dV_Coordinates = [dV_Point1; dV_Point2; dV_Point3];
%compute the transform
T_dV_TRUS = LeastSquaresNumericalTransform(TRUS_Coordinates,dV_Coordinates)

%Use the transform to compute the da Vinci points based on TRUS points and
%plot the error
TRUS_data = []; %holds predicted da Vinci values
for i = 1:length(us_track)
    h = us_track(i,1);
    g = us_track(i,2);
    theta = robot_data_resamp(frame_vec(i),4)*2*pi/360; %in radians
    TRUS_data = [TRUS_data; (T_dV_TRUS*[-1*g/max_g*l, (max_h-h)/max_h*l*sin(theta), (max_h-h)/max_h*l*cos(theta), 1]')']; 
end
dV_data = [];
for i = 1:length(us_track)
    dV_data = [dV_data; dvrk_xyz(frame_vec(i),:)];
end

figure; 
plot(TRUS_data(:,3),"*")
hold on 
plot(dV_data(:,3), "*")
legend(["TRUS_predicited", "True"])
% TRUS_data =
% transformedTRUS = 

%% Verifying the points

vidReader=VideoReader(datapath_video);
j=1;
m=1;
while hasFrame(vidReader)
frame=readFrame(vidReader);
CropRec=[1607.51,85.51,883.98,740.98];
frameCropped=imcrop(frame,CropRec); %Crops just the US portion of the image
frameGray=rgb2gray(frameCropped);
imshow(frameGray);
if m==frame_vec(j)
    viscircles(us_track(j,:),10);
    pause(0.5);
    j=j+1;
end
pause(0.01);
m=m+1;
end







%% -----------------------<Function definitions>---------------------------
function tracks=TrackDetect(datapath)
    % Reading in Video
    vidReader=VideoReader(datapath); %Create an object with the video
    % Setting up the Object which contains the tracks
    %{
    ID=integer ID of the track
    Centroid=center of moving objects
    kalmanFilter=Used for motion-based tracking
    age=the number of frames since the track was detected
    totalVisibleCount= # of frames where the track was detected
    consecutiveInvisibleCount=the number of consecutive frames where track
    was not detected
    %}
    %bbox is also included as a test
    tracks=struct('id',{},'Centroid',{},'kalmanFilter',{},...
                    'age',{},'totalVisibleCount',{},'consecutiveInvisibleCount',{},'framenum',{},'TotalTrack',{});
    nextId=1; %This is the ID of the next track
    
    %Setting up Optic Flow Object
    opticFlow=opticalFlowLK('NoiseThreshold',0.001); %Increasing number means movement of object has less impact on calculation
    
    % Looping Through Video Frames
    
    m=1;
    while hasFrame(vidReader)
        frame=readFrame(vidReader);
        [centroids,Mask,radii]=detectObjects(frame);
        predictNewLocationsOfTracks();
        [assignments,unassignedTracks,unassignedDetections]=detectionToTrackAssignment();
        updateAssignedTracks();
        updateUnassignedTracks();
        deleteLostTracks();
        createNewTracks();
        
        disp(m); m=m+1;
    end
    
    
    function [centroids,Mask,radii]=detectObjects(frameRGB)
        % Parameters
        VelThresh=1; %Threshold for velocity magnitudes that are not considered movement
        radmin=8;
        radmax=30;
        HughSensit=0.8;

        CropRec=[1607.51,85.51,883.98,740.98];
        frameCropped=imcrop(frameRGB,CropRec); %Crops just the US portion of the image
        frameGray=rgb2gray(frameCropped);

        flow=estimateFlow(opticFlow,frameGray);
        Mask=imbinarize(flow.Magnitude,VelThresh);

        Mask=imopen(Mask,strel('disk',2));
        [centroids,radii,metric]=imfindcircles(Mask,[radmin radmax],'Sensitivity',HughSensit);
%         imshow(Mask);
%         viscircles(centroids,radii);
%         pause(0.01);
            
    end
    
    
    
    function predictNewLocationsOfTracks()
        %Note that we can play around with the Kalman Filter model to change the
        %Noise variance as well as the transition and observation matrix
        
        
            for i=1:length(tracks)
               predictedCentroid=predict(tracks(i).kalmanFilter);
%                viscircles(predictedCentroid,5,'Color','b');
%                pause(0.5)
        
               predictedCentroid=int32(predictedCentroid);
               tracks(i).Centroid=predictedCentroid;
        
            end
        
        
    end
    
    
    function [assignments,unassignedTracks,unassignedDetections]=...
            detectionToTrackAssignment()
        
            %Assign object detections in current frame to assigning tracks
            %This is done by minimizing the cost (which is the log-likelihood of a
            %detection corresponding to a track). 
            %Step1: compute cost of assigned every detection to each track it
            %accounts the Euclidean distance as well as the confidence of the
            %prediction
            %Step 2: Solve the assignment problem represented by cost matrix
        
            %Value of cost of not assigneding a detection to a track must be tuned
            %manually. Setting it too low increases likelihood of a new track,
            %setting too high may result in a single track representing multiple
            %moving objects
        
            nTracks=length(tracks);
            nDetections=size(centroids,1);
        
            cost=zeros(nTracks,nDetections); %Cost of assigning each detection to each track, where each row is cost for a single track
        
            for i=1:nTracks
                %------------<Maybe change tracks(i).kalmanFilter to
                %tracks(i).Centroid
                cost(i,:)=distance(tracks(i).kalmanFilter,centroids)*0.01; %This finds distance between centroids and tracks, we can adjust thie algorithm to go faster using weights (see the documentation for this)
%                 circlescenters=tracks(i).kalmanFilter.State([1,4],1);
%                 viscircles(circlescenters',5,'Color','g');
%                 pause(0.5)
            end
        
            %costOfNonAssignment=100000000;
            unassignedTrackCost=0.1; %Cost of not assigning detection to track
            unassignedDetectionCost=10000; %Cost of starting a new track for that detection
        
            %assigns detections to tracks using the James Munkres variant of the
            %Hungarian algorithm
            %Lowe the cost more likely a detection gets assigned to a track
            [assignments,unassignedTracks,unassignedDetections]=assignDetectionsToTracks(cost,unassignedTrackCost,unassignedDetectionCost);
        
            
    end
    
    
    function updateAssignedTracks()
            %Updates assigned tracks with the corresponding detections
        
            numAssignedTracks=size(assignments,1); 
            for i=1:numAssignedTracks
                trackIdx=assignments(i,1); %The index of the track from the assignment detection algorithm
                detectionIdx=assignments(i,2);
                centroid=centroids(detectionIdx,:);
                tracks(trackIdx).TotalTrack=[tracks(trackIdx).TotalTrack;centroid];
                tracks(trackIdx).framenum=[tracks(trackIdx).framenum;m];
                correct(tracks(trackIdx).kalmanFilter,centroid);
        
                %update track's age
                tracks(trackIdx).age=tracks(trackIdx).age+1;
        
                
                %Update the visibility
                tracks(trackIdx).totalVisibleCount=tracks(trackIdx).totalVisibleCount+1;
                tracks(trackIdx).consecutiveInvisibleCount=0;
                
%                 viscircles(tracks(1).TotalTrack(end,:),5,'Color','b');
%                 pause(0.5)
        
            end
        
    end
    
    function updateUnassignedTracks()
            %Mark each unassigned track as invisible and increase age by 1
            for i=1:length(unassignedTracks)
                ind=unassignedTracks(i);
                tracks(ind).age=tracks(ind).age+1;
                tracks(ind).consecutiveInvisibleCount=tracks(ind).consecutiveInvisibleCount+1;
                tracks(ind).TotalTrack=[tracks(ind).TotalTrack;[0,0]];
                tracks(ind).framenum=[tracks(ind).framenum;m];
            end
    end
    
    function deleteLostTracks()
            if isempty(tracks)
                return;
            end
            
            %Deletes tracks that have been invisible for too many consecutive
            %frames
            %Also deletes recently created tracks that have been invisible for too
            %many frames overall
        
            invisibleForTooLong=30;
            ageThreshold=8;
        
            %compute fraction of track's age for where it was visible
        
            ages=[tracks(:).age];
            totalVisibleCounts=[tracks(:).totalVisibleCount];
            visibility=totalVisibleCounts./ages;
        
            %Find indexes of lost tracks
            lostInds=(ages<ageThreshold & visibility<0.6) | ...
                ([tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong);
        
            %Delete the lost tracks
            tracks=tracks(~lostInds);
        end
    
    
    function createNewTracks()
            %We assign new tracks assuming unassigned detection is the start of a
            %new track
            centroids_new=centroids(unassignedDetections,:);
            
            for i=1:size(centroids_new,1)
                centroid=centroids_new(i,:);
                %Creat a Kalman filter object (we can change this)
                kalmanFilter=configureKalmanFilter('ConstantAcceleration',...
                centroid,[2,5.175e3,9.377e3],[10,25,50]*1e2,0.02);
%                   kalmanFilter=trackingEKF(@constacc,@cvmeas,centroid,...
%                       'StateTransitionJacobianFcn',@constaccjac,...
%                       'MeasurementJacobianFcn',@cvmeasjac,...
%                       'MeasurementNoise',0.02,'HasAdditiveMeasurementNoise',false,...
%                       'ProcessNoise',1000);

%                 kalmanFilter=configureKalmanFilter('ConstantAcceleration',...
%                 centroid,[2,5.175e3,9.377e3],[10,25,50]*1e3,0.001);
%                 kalmanFilter=configureKalmanFilter('ConstantVelocity',...
%                 centroid,[1,1]*1e5,[25,10],25);
        
                %Create a new track
                newTrack=struct('id',nextId,'Centroid',centroid,'kalmanFilter',kalmanFilter,...
                    'age',1,'totalVisibleCount',1,'consecutiveInvisibleCount',0,'framenum',m,'TotalTrack',centroid);
                %newTrack.TotalTrack{1}=centroid;
        
                tracks(end+1)=newTrack;
                nextId=nextId+1;
        
            end
    end

end


function [traj]=interpolator(traj)
    for k=1:size(traj,1)
        if traj(k,1)==0 
            if k==1
                traj(1,:)=checkahead(traj,k+1);
            elseif k==size(traj,1)
                traj(k,:)=traj(k-1,:);
            else
                traj(k,:)=(traj(k-1,:)+checkahead(traj,k+1))/2; %Simple straight line interp
            end

        end
    end

    function val=checkahead(traj,k)
        %Nested Function to return the value of the next element that is
        %not zero
        while traj(k,1)==0
            if k==size(traj,1)
                break
            end
            k=k+1;
        end
        val=traj(k,:);
    end

end




