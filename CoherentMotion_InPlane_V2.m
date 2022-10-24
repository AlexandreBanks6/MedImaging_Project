clear
clc
CohMotion()

function CohMotion()
    obj=setupSystemObjects();
    tracks=initializeTracks(); %Empty array of tracks to be filled
    
    nextId=1; %ID of the next track
    
    %Loops for all the frames of the video
    while hasFrame(obj.reader)
        frame=readFrame(obj.reader);
        [centroids, bboxes, mask] = detectObjects(frame);
    
        predictNewLocationsOfTracks();
        [assignments,unassignedTracks,unassignedDetections]=detectionToTrackAssignment();
        updateAssignedTracks();
        updateUnassignedTracks();
        deleteLostTracks();
        createNewTracks();
    
        displayTrackingResults();
    end
    
    
    
    function obj=setupSystemObjects()
        %Read video from a file
        %Draw the tracked objects in each frame
    
        %Video reader object
        obj.reader=VideoReader('SoccerVid_Good_Trim.avi');
    
        %Two video players, one to display forground mask and the
        %other to display the video
    
        obj.maskPlayer=vision.VideoPlayer('Position', [740, 400, 700, 400]);
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        
        %Create objects for forground detection and blob analysis
        
        obj.detector=vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
    
        %Use blob analysis to find connected regions of moving
        %objects in the video sequence
    
        obj.blobAnalyser=vision.BlobAnalysis('MinimumBlobArea',50); %Returns coordinates of bounding boxes
                                                %Returns the blob area
                                                %Returns the coordinates of
                                                %blob centroinds
    end
    
    function tracks=initializeTracks()
        %Creates an empty array of tracks
        %{
        ID=integer ID of the track
        bbox=the bounding box of hte object (this can be deleted after)
        kalmanFilter=Used for motion-based tracking
        age=the number of frames since the track was detected
        totalVisibleCount= # of frames where the track was detected
        consecutiveInvisibleCount=the number of consecutive frames where track
        was not detected
        %}
    
        tracks=struct('id',{},'bbox',{},'kalmanFilter',{},'age',{},'totalVisibleCount',{},...
            'consecutiveInvisibleCount',{});
    end
    
    function [centroids,bboxes,mask]=detectObjects(frame)
    
    %This function performs motion segmentation using the foreground detector
    %Performs morphological operation to remove noisy pixels
    %Returns centroids and bounding boxes, as well as the mask for that frame
    
    mask = obj.detector.step(frame);
    %Perform morphological operations which are summarized below:
    %{
    It performs a morphological opening, which is an erosion followed by a
    dilation of the eroded image and we use the same structuring elements for
    both operations. Useful for removing small objects nad thin lines from an
    image.
    
    Erosion: Value of the output pixel is the minimum of all pixels in the
    neighborhood
    Dilation: Value of the output pixel is the maximum of all pixels in the
    neighborhood
    The neighborhood is determined by the structuring element
    
    We can change this later to a different form of filtering
    %}
    mask=imopen(mask,strel('rectangle',[3,3]));
    
    
    %{
    imclose is essentially the opposite to imopen and is a dilation followed
    by an erosion
    I am not sure if it is necessary so I am going to comment it out for now
    %}
    mask=imclose(mask,strel('rectangle',[15,15]));
    mask=imfill(mask,'holes'); %Fills any holes in the mask
    
    %Perform blob analysis to find connected components
    [~,centroids,bboxes]=obj.blobAnalyser.step(mask);
    end
    
    function predictNewLocationsOfTracks()
    %Note that we can play around with the Kalman Filter model to change the
    %Noise variance as well as the transition and observation matrix
    
    
        for i=1:length(tracks)
           bbox=tracks(i).bbox;
           predictedCentroid=predict(tracks(i).kalmanFilter);
    
           predictedCentroid=int32(predictedCentroid)-bbox(3:4)/2;
           tracks(i).bbox=[predictedCentroid,bbox(3:4)];
    
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
            bbox=bboxes(detectionIdx,:);
    
            %Correct the estimate of the object's location using detection
            correct(tracks(trackIdx).kalmanFilter,centroid);
            tracks(trackIdx).bbox=bbox;
    
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
        bboxes=bboxes(unassignedDetections,:);
        
        for i=1:size(centroids,1)
            centroid=centroids(i,:);
            bbox=bboxes(i,:);
            %Creat a Kalman filter object (we can change this)
            kalmanFilter=configureKalmanFilter('ConstantAcceleration',...
            centroid,[1,1,1]*1e5,[25,10,10],25);
    
            %Create a new track
            newTrack=struct('id',nextId,'bbox',bbox,'kalmanFilter',kalmanFilter,...
                'age',1,'totalVisibleCount',1,'consecutiveInvisibleCount',0);
    
            tracks(end+1)=newTrack;
            nextId=nextId+1;
    
        end
    
    
    
    
    end
    
    
    
    function displayTrackingResults()
        %Convert frame and mask to uint8 RGB
        frame=im2uint8(frame);
        mask=uint8(repmat(mask,[1,1,3])).*225;
    
        minVisibleCount=8;
    
        if ~isempty(tracks)
    
            %noisy detections result in short-lived tracks, only displaying
            %those that have been visible for more than a minimum number of
            %frames
            reliableTrackInds=[tracks(:).totalVisibleCount]>minVisibleCount;
            reliableTracks=tracks(reliableTrackInds);
    
            %display the object
    
            if ~isempty(reliableTracks)
                %getting the bounding boxes
                bboxes=cat(1,reliableTracks.bbox);
    
                %Get ids
                ids=int32([reliableTracks(:).id]);
    
                labels=cellstr(int2str(ids'));
    
                predictedTrackInds=[reliableTracks(:).consecutiveInvisibleCount]>0;
                isPredicted=cell(size(labels));
                isPredicted(predictedTrackInds)={' predicted'};
                labels=strcat(labels,isPredicted);
    
                %Draw the objects on the frame
                frame=insertObjectAnnotation(frame,'rectangle',bboxes,labels);
    
                mask=insertObjectAnnotation(mask,'rectangle',bboxes,labels);
    
            end
    
        end
    
        obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
    
    end

end

