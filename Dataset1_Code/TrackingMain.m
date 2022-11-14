clear 
clc
close all

% Reading Data

%datapath_robot=['C:\Users\playf\OneDrive\Documents\UBC\Alexandre_UNI_2022_2023\' ...
    %'Semester1\ELEC_523_MedImaging\Project\MooreBanks_Results\Trial4\Registration\data.csv'];
datapath_video=['C:/Users/playf/OneDrive/Documents/UBC/Alexandre_UNI_2022_2023/Semester1'...
    '/ELEC_523_MedImaging/Project/MooreBanks_Results/Trial4/Registration/Recorder_2_Nov11_20-01-30.mp4'];

%[time_vec,frame_vec,robot_data]=DaVinciToolData_Processing(datapath_robot); %Interpolates the da Vinci data, to match the US frames

%dvrk_xyz=robot_data(:,[1:3]); %End-effector xyz
first_frame=1;

% Finding Tracks
tracks=TrackDetect(datapath_video,first_frame);








function tracks=TrackDetect(datapath,first_frame)
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
    tracks=struct('id',{},'Centroid',{},'kalmanFilter',{},...
                    'age',{},'totalVisibleCount',{},'consecutiveInvisibleCount',{},'TotalTrack',{});
    nextId=1; %This is the ID of the next track
    
    %Setting up Optic Flow Object
     opticFlow=opticalFlowLK('NoiseThreshold',0.001); %Increasing number means movement of object has less impact on calculation
    
    % Looping Through Video Frames
    
    m=1;
    while hasFrame(vidReader)
        frame=readFrame(vidReader);
        if m>=first_frame
            [centroids,Mask,radii]=detectObjects(frame);
            predictNewLocationsOfTracks();
            [assignments,unassignedTracks,unassignedDetections]=detectionToTrackAssignment();
            updateAssignedTracks();
            updateUnassignedTracks();
            deleteLostTracks();
            createNewTracks();
        end
    
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
        imshow(Mask);
        viscircles(centroids,radii);
        pause(0.01);
            
    end
    
    
    
    function predictNewLocationsOfTracks()
        %Note that we can play around with the Kalman Filter model to change the
        %Noise variance as well as the transition and observation matrix
        
        
            for i=1:length(tracks)
               predictedCentroid=predict(tracks(i).kalmanFilter);
        
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
                cost(i,:)=distance(tracks(i).kalmanFilter,centroids); %This finds distance between centroids and tracks, we can adjust thie algorithm to go faster using weights (see the documentation for this)
            end
        
            costOfNonAssignment=200;
        
            %assigns detections to tracks using the James Munkres variant of the
            %Hungarian algorithm
            %Lowe the cost more likely a detection gets assigned to a track
            [assignments,unassignedTracks,unassignedDetections]=assignDetectionsToTracks(cost,costOfNonAssignment);
        
    end
    
    
    function updateAssignedTracks()
            %Updates assigned tracks with the corresponding detections
        
            numAssignedTracks=size(assignments,1); 
            for i=1:numAssignedTracks
                trackIdx=assignments(i,1); %The index of the track from the assignment detection algorithm
                detectionIdx=assignments(i,2);
                centroid=centroids(detectionIdx,:);
                tracks(trackIdx).TotalTrack=[tracks(trackIdx).TotalTrack;centroid];
                correct(tracks(trackIdx).kalmanFilter,centroid);
        
                %update track's age
                tracks(trackIdx).age=tracks(trackIdx).age+1;
        
                
                %Update the visibility
                tracks(trackIdx).totalVisibleCount=tracks(trackIdx).totalVisibleCount+1;
                tracks(trackIdx).consecutiveInvisibleCount=0;
        
        
            end
        
    end
    
    function updateUnassignedTracks()
            %Mark each unassigned track as invisible and increase age by 1
            for i=1:length(unassignedTracks)
                ind=unassignedTracks(i);
                tracks(ind).age=tracks(ind).age+1;
                tracks(ind).consecutiveInvisibleCount=tracks(ind).consecutiveInvisibleCount+1;
                tracks(ind).TotalTrack=[tracks(ind).TotalTrack;[0,0]];
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
        
            invisibleForTooLong=20;
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
            centroids=centroids(unassignedDetections,:);
            
            for i=1:size(centroids,1)
                centroid=centroids(i,:);
                %Creat a Kalman filter object (we can change this)
                kalmanFilter=configureKalmanFilter('ConstantAcceleration',...
                centroid,[1,1,1]*1e5,[25,10,10],25);
        
                %Create a new track
                newTrack=struct('id',nextId,'Centroid',centroid,'kalmanFilter',kalmanFilter,...
                    'age',1,'totalVisibleCount',1,'consecutiveInvisibleCount',0,'TotalTrack',centroid);
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




