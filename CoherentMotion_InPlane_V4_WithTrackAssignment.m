clear
clc

close all
CohMot();
function CohMot()
    % Reading in Video
    vidReader=VideoReader('BallsRolling_Trim.mp4'); %Create an object with the video
    
    % Creating a optical flow object using Lucas-Kanade Method
    opticFlow=opticalFlowLK('NoiseThreshold',0.001); %Increasing number means movement of object has less impact on calculation 
    
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

    %tracks=struct('id',{},'Centroid',{},'kalmanFilter',{},'age',{},'totalVisibleCount',{},...
     %       'consecutiveInvisibleCount',{});
    nextId=1; %This is the ID of the next track
    
    
    
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
    
    
    
        imshow(Mask);
        viscircles(centroids,radii,'EdgeColor','b');
        pause(0.001);
    
        
    
        disp(m); m=m+1;
    end
    
    
    function [centroids,Mask,radii]=detectObjects(frameRGB)
        % Parameters
        VelThresh=0.075; %Threshold for velocity magnitudes that are not considered movement
        MinObArea=25; %Used to remove all objects with less than this value of connected pixels
        radmin=30;
        radmax=100;
        HughSensit=0.8;
        
        frameGray=im2gray(frameRGB);
        flow=estimateFlow(opticFlow,frameGray);
        GrayNew=imopen(flow.Magnitude,strel('disk',1)); %Performs erosion then dilation (morph op)
        Mask=imbinarize(GrayNew,VelThresh); %Sets Pixels with velocity above threshold to 1
        
        Mask=imclose(Mask,strel('disk',3)); %Performs a dilation then an erosion (morph op)
        Mask=bwareaopen(Mask,MinObArea); %Any objects with area smaller than MinObArea are discarded
        [centroids,radii,metric]=imfindcircles(Mask,[radmin radmax],'Sensitivity',HughSensit);
    
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
        
            costOfNonAssignment=20;
        
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
                %CentroidStruct{trackIdx,m}=centroids(detectionIdx,:); %Adds the detection to the track
                %{
                if(m==1)
                    tracks(trackIdx).TotalTrack{1}=centroid;
                else
                %}
                %tracks(trackIdx).TotalTrack{tracks(trackIdx).age+1}=centroid;
                tracks(trackIdx).TotalTrack=[tracks(trackIdx).TotalTrack;centroid];
                %Correct the estimate of the object's location using detection
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
